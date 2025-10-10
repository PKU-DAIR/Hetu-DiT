from transformers.models.t5.modeling_t5 import T5Attention
from torch import nn
from hetu_dit.core.distributed.parallel_state import (
    get_text_encoder_tensor_model_parallel_rank,
    get_text_encoder_tensor_model_parallel_world_size,
    get_text_encoder_tp_group,
)
import torch
from hetu_dit.model_executor.utils.register_warpper import hetuDiTLayerWrappersRegister
from hetu_dit.model_executor.textencoder_executor.layers.base_layer import (
    hetuDiTTextEncoderLayerBaseWrapper,
)


@hetuDiTLayerWrappersRegister.register(T5Attention)
class hetuDiTT5AttentionWrapper(hetuDiTTextEncoderLayerBaseWrapper):
    def __init__(self, attention: T5Attention):
        super(hetuDiTT5AttentionWrapper, self).__init__(module=attention)

        self.tp_degree = get_text_encoder_tensor_model_parallel_world_size()
        self.tp_rank = get_text_encoder_tensor_model_parallel_rank()
        for proj_name in ["q", "k", "v"]:
            proj_layer = getattr(self.module, proj_name)
            proj_weight = proj_layer.weight.data.chunk(self.tp_degree, dim=0)[
                self.tp_rank
            ].clone()
            proj_layer.weight.data = proj_weight

            if proj_layer.bias is not None:
                proj_bias = proj_layer.bias.data.chunk(self.tp_degree, dim=0)[
                    self.tp_rank
                ].clone()
                proj_layer.bias.data = proj_bias

        # Handle output projection
        proj_layer = self.module.o
        proj_layer.weight.data = proj_layer.weight.data.chunk(self.tp_degree, dim=1)[
            self.tp_rank
        ].clone()

        self.has_output_bias = False
        if proj_layer.bias is not None:
            self.register_parameter(
                "output_bias", nn.Parameter(proj_layer.bias.data.clone())
            )
            # print(f"module.to_out[0].bias {proj_layer.bias.shape}")
            proj_layer.bias = None
            self.has_output_bias = True

        torch.cuda.empty_cache()

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        n_heads_per_device = (
            self.module.n_heads // get_text_encoder_tensor_model_parallel_world_size()
        )
        mask = kwargs.get("mask", None)
        key_value_states = kwargs.get("key_value_states", None)
        position_bias = kwargs.get("position_bias", None)
        past_key_value = kwargs.get("past_key_value", None)
        layer_head_mask = kwargs.get("layer_head_mask", None)
        query_length = kwargs.get("query_length", None)
        use_cache = kwargs.get("use_cache", False)
        output_attentions = kwargs.get("output_attentions", False)
        cache_position = kwargs.get("cache_position", None)
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, seq_length, key_length) (causal decoder)
        batch_size, seq_length = hidden_states.shape[:2]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states)
        query_states = query_states.view(
            batch_size, -1, n_heads_per_device, self.key_value_proj_dim
        ).transpose(1, 2)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(
                batch_size, -1, n_heads_per_device, self.key_value_proj_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                batch_size, -1, n_heads_per_device, self.key_value_proj_dim
            ).transpose(1, 2)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"cache_position": cache_position},
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = (
                query_length if query_length is not None else cache_position[-1] + 1
            )
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, n_heads_per_device, seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length,
                    key_length,
                    device=scores.device,
                    cache_position=cache_position,
                )
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias
        # position_bias_masked = position_bias_masked.chunk(self.tp_degree, dim=1)[self.tp_rank]
        position_bias_masked = position_bias_masked.chunk(
            get_text_encoder_tensor_model_parallel_world_size(), dim=1
        )[get_text_encoder_tensor_model_parallel_rank()]
        scores += position_bias_masked

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(
            batch_size,
            -1,
            self.inner_dim // get_text_encoder_tensor_model_parallel_world_size(),
        )
        attn_output = self.o(attn_output)
        get_text_encoder_tp_group().all_reduce(attn_output)
        if self.has_output_bias:
            attn_output += self.output_bias

        outputs = (attn_output, past_key_value, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
