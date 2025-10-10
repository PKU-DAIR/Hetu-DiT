import inspect
from typing import Optional, Union, Tuple

import torch
from torch import nn
import torch.distributed
from torch.nn import functional as F
from diffusers.utils import deprecate
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0,
    HunyuanAttnProcessor2_0,
    CogVideoXAttnProcessor2_0,
)
from hetu_dit.envs import PACKAGES_CHECKER
from hetu_dit.logger import init_logger

logger = init_logger(__name__)

try:
    logger.debug("successfully import HunyuanVideoAttnProcessor2_0")
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoAttnProcessor2_0,
    )
except ImportError:
    logger.warning("failed to import HunyuanVideoAttnProcessor2_0")
    HunyuanVideoAttnProcessor2_0 = None

from hetu_dit.core.distributed import (
    get_sequence_parallel_world_size,
    get_pipeline_parallel_world_size,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tp_group,
)

from hetu_dit.core.resource_manager.cache_manager import get_cache_manager
from hetu_dit.core.distributed.runtime_state import get_runtime_state
from hetu_dit.model_executor.diffusion_executor.layers import (
    hetuDiTDiffusionLayerBaseWrapper,
)
from hetu_dit.model_executor.utils.register_warpper import hetuDiTLayerWrappersRegister
from hetu_dit.model_executor.utils.register_warpper import (
    hetuDiTAttentionProcessorRegister,
)

come_int_time = 0
env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]
HAS_FLASH_ATTN = env_info["has_flash_attn"]


def is_v100():
    if not torch.cuda.is_available():
        return False
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return "V100" in device_name


def torch_compile_disable_if_v100(func):
    if is_v100():
        return torch.compiler.disable(func)
    return func


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox,
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(
                -1
            )  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(
                -2
            )  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(
                f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
            )

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class hetuDiTAttentionBaseWrapper(hetuDiTDiffusionLayerBaseWrapper):
    def __init__(
        self,
        attention: Attention,
    ):
        super().__init__(module=attention)

        to_k = self.module.to_k
        to_v = self.module.to_v
        assert isinstance(to_k, nn.Linear)
        assert isinstance(to_v, nn.Linear)
        assert (to_k.bias is None) == (to_v.bias is None)
        assert to_k.weight.shape == to_v.weight.shape
        tp_degree = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for proj_name in [
            "to_q",
            "to_k",
            "to_v",
            "add_k_proj",
            "add_v_proj",
            "add_q_proj",
        ]:
            proj_layer = getattr(self.module, proj_name, None)
            if proj_layer is not None:
                proj_weight = proj_layer.weight.data.chunk(tp_degree, dim=0)[
                    tp_rank
                ].clone()
                proj_layer.weight.data = proj_weight

                if proj_layer.bias is not None:
                    proj_bias = proj_layer.bias.data.chunk(tp_degree, dim=0)[
                        tp_rank
                    ].clone()
                    proj_layer.bias.data = proj_bias

        # Handle output projection
        if not self.module.pre_only:
            proj_layer = self.module.to_out[0]
            proj_layer.weight.data = proj_layer.weight.data.chunk(tp_degree, dim=1)[
                tp_rank
            ].clone()

        self.has_output_bias = False
        if not self.module.pre_only and proj_layer.bias is not None:
            self.register_parameter(
                "output_bias", nn.Parameter(proj_layer.bias.data.clone())
            )
            proj_layer.bias = None
            self.has_output_bias = True

        # Handle output projection
        if (
            self.module.context_pre_only is not None
            and not self.module.context_pre_only
        ):
            proj_layer = getattr(self.module, "to_add_out", None)
            if proj_layer is not None:
                proj_layer.weight.data = proj_layer.weight.data.chunk(tp_degree, dim=1)[
                    tp_rank
                ].clone()

                self.has_encoder_output_bias = False
                if proj_layer.bias is not None:
                    self.register_parameter(
                        "encoder_output_bias",
                        nn.Parameter(proj_layer.bias.data.clone()),
                    )
                    proj_layer.bias = None
                    self.has_encoder_output_bias = True
        self.heads_per_device = self.module.heads // tp_degree


@hetuDiTLayerWrappersRegister.register(Attention)
class hetuDiTAttentionWrapper(hetuDiTAttentionBaseWrapper):
    def __init__(
        self,
        attention: Attention,
    ):
        super().__init__(attention=attention)
        self.processor = hetuDiTAttentionProcessorRegister.get_processor(
            attention.processor
        )()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        global come_int_time

        come_int_time += 1
        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k
            for k, _ in cross_attention_kwargs.items()
            if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {
            k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters
        }

        if type(self.processor) == hetuDiTCogVideoXAttnProcessor2_0:
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
            hidden_states = self.processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            if get_tensor_model_parallel_world_size() > 1:
                get_tp_group().all_reduce(hidden_states.contiguous())

            if self.has_output_bias:
                hidden_states += self.output_bias
            # dropout
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states, hidden_states = hidden_states.split(
                [
                    encoder_hidden_states.size(1),
                    hidden_states.size(1) - encoder_hidden_states.size(1),
                ],
                dim=1,
            )
            return hidden_states, encoder_hidden_states
        elif type(self.processor) == hetuDiTFluxAttnProcessor2_0:
            if encoder_hidden_states is not None:
                hidden_states, encoder_hidden_states = self.processor(
                    self,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                encoder_hidden_states = self.to_add_out(encoder_hidden_states)
                # TODO: Need to reduce the time of contiguous()
                if get_tensor_model_parallel_world_size() > 1:
                    get_tp_group().all_reduce(hidden_states.contiguous())
                    get_tp_group().all_reduce(encoder_hidden_states.contiguous())

                if self.has_output_bias:
                    hidden_states += self.output_bias
                # dropout
                hidden_states = self.to_out[1](hidden_states)
                return hidden_states, encoder_hidden_states
            else:
                hidden_states = self.processor(
                    self,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

                if get_tensor_model_parallel_world_size() > 1:
                    hidden_states = get_tp_group().all_gather(
                        hidden_states.contiguous(), dim=2
                    )
                return hidden_states
        elif (
            type(self.processor) == hetuDiTHunyuanVideoAttnProcessor2_0
            and hetuDiTHunyuanVideoAttnProcessor2_0 is not None
        ):
            hidden_states = self.processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            # 6. Output projection
            if encoder_hidden_states is not None and self.context_pre_only is False:
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, : -encoder_hidden_states.shape[1]],
                    hidden_states[:, -encoder_hidden_states.shape[1] :],
                )

                if getattr(self, "to_out", None) is not None:
                    hidden_states = self.to_out[0](hidden_states)
                    if get_tensor_model_parallel_world_size() > 1:
                        get_tp_group().all_reduce(hidden_states.contiguous())

                    if self.has_output_bias:
                        hidden_states += self.output_bias
                    hidden_states = self.to_out[1](hidden_states)

                if getattr(self, "to_add_out", None) is not None:
                    encoder_hidden_states = self.to_add_out(encoder_hidden_states)
                    if get_tensor_model_parallel_world_size() > 1:
                        get_tp_group().all_reduce(encoder_hidden_states.contiguous())

                return hidden_states, encoder_hidden_states
            elif self.pre_only is True:
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, : -encoder_hidden_states.shape[1]],
                    hidden_states[:, -encoder_hidden_states.shape[1] :],
                )
                if get_tensor_model_parallel_world_size() > 1:
                    hidden_states = get_tp_group().all_gather(
                        hidden_states.contiguous(), dim=2
                    )
                    encoder_hidden_states = get_tp_group().all_gather(
                        encoder_hidden_states.contiguous(), dim=2
                    )
                return hidden_states, encoder_hidden_states

        elif type(self.processor) == hetuDiTHunyuanAttnProcessor2_0:
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
            if encoder_hidden_states is not None:
                context_input_ndim = encoder_hidden_states.ndim
                if context_input_ndim == 4:
                    batch_size, channel, height, width = encoder_hidden_states.shape
            hidden_states, residual = self.processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            if get_tensor_model_parallel_world_size() > 1:
                get_tp_group().all_reduce(hidden_states.contiguous())

            if self.has_output_bias:
                hidden_states += self.output_bias
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        else:
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
            hidden_states, encoder_hidden_states = self.processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            if not self.context_pre_only and self.to_add_out:
                encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            # TODO: Need to reduce the time of contiguous()
            if get_tensor_model_parallel_world_size() > 1:
                get_tp_group().all_reduce(hidden_states.contiguous())
                get_tp_group().all_reduce(encoder_hidden_states.contiguous())

            if self.has_output_bias:
                hidden_states += self.output_bias
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )
            if context_input_ndim == 4:
                encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )

            if (
                self.module.context_pre_only is not None
                and not self.module.context_pre_only
                and self.has_encoder_output_bias
            ):
                encoder_hidden_states += self.encoder_output_bias

            return hidden_states, encoder_hidden_states


@hetuDiTAttentionProcessorRegister.register(AttnProcessor2_0)
class hetuDiTAttnProcessor2_0(AttnProcessor2_0):
    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            from hetu_dit.core.parallel import (
                hetuDiTLongContextAttention,
                hetuDiTUlyssesAttention,
            )

            if HAS_FLASH_ATTN:
                # self.hybrid_seq_parallel_attn = LongContextAttention()
                self.hybrid_seq_parallel_attn = hetuDiTLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache
                )
            else:
                self.hybrid_seq_parallel_attn = hetuDiTUlyssesAttention(
                    use_fa=False,
                    use_kv_cache=self.use_long_ctx_attn_kvcache,
                )
        else:
            self.hybrid_seq_parallel_attn = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        self.heads_per_device = attn.heads // get_tensor_model_parallel_world_size()
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, self.heads_per_device, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads_per_device

        query = query.view(batch_size, -1, self.heads_per_device, head_dim).transpose(
            1, 2
        )

        key = key.view(batch_size, -1, self.heads_per_device, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads_per_device, head_dim).transpose(
            1, 2
        )

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        #! ---------------------------------------- KV CACHE ----------------------------------------
        if not self.use_long_ctx_attn_kvcache:
            key, value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[key, value],
                layer=attn,
                slice_dim=2,
                layer_type="attn",
            )
        #! ---------------------------------------- KV CACHE ----------------------------------------

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            hidden_states = self.hybrid_seq_parallel_attn(
                attn,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                joint_strategy="none",
            )
            hidden_states = hidden_states.reshape(
                batch_size, -1, self.heads_per_device * head_dim
            )

        else:
            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(
                    batch_size, -1, self.heads_per_device * head_dim
                )

            else:
                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.module.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )

                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, self.heads_per_device * head_dim
                )
        #! ---------------------------------------- ATTENTION ----------------------------------------

        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@hetuDiTAttentionProcessorRegister.register(JointAttnProcessor2_0)
class hetuDiTJointAttnProcessor2_0(JointAttnProcessor2_0):
    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            from hetu_dit.core.parallel import (
                hetuDiTJointLongContextAttention,
                hetuDiTUlyssesAttention,
            )

            if HAS_FLASH_ATTN:
                self.hybrid_seq_parallel_attn = hetuDiTJointLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache
                )
            else:
                self.hybrid_seq_parallel_attn = hetuDiTUlyssesAttention(
                    use_fa=False,
                    use_kv_cache=self.use_long_ctx_attn_kvcache,
                )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1
        )
        global come_int_time  # TODO: DEL it
        residual = hidden_states
        self.heads_per_device = attn.heads // get_tensor_model_parallel_world_size()
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads_per_device

        #! ---------------------------------------- KV CACHE ----------------------------------------
        if not self.use_long_ctx_attn_kvcache:
            key, value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[key, value],
                layer=attn,
                slice_dim=1,
                layer_type="attn",
            )
        #! ---------------------------------------- KV CACHE ----------------------------------------

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            query = query.view(batch_size, -1, self.heads_per_device, head_dim)
            key = key.view(batch_size, -1, self.heads_per_device, head_dim)
            value = value.view(batch_size, -1, self.heads_per_device, head_dim)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, self.heads_per_device, head_dim
            )
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, self.heads_per_device, head_dim
            )
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, self.heads_per_device, head_dim
            )
            hidden_states = self.hybrid_seq_parallel_attn(
                attn,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                joint_tensor_query=encoder_hidden_states_query_proj,
                joint_tensor_key=encoder_hidden_states_key_proj,
                joint_tensor_value=encoder_hidden_states_value_proj,
                joint_strategy="rear",
            )
            hidden_states = hidden_states.reshape(
                batch_size, -1, self.heads_per_device * head_dim
            )

        else:
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.view(batch_size, -1, self.heads_per_device, head_dim)
                key = key.view(batch_size, -1, self.heads_per_device, head_dim)
                value = value.view(batch_size, -1, self.heads_per_device, head_dim)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(
                    batch_size, -1, self.heads_per_device * head_dim
                )

            else:
                query = query.view(
                    batch_size, -1, self.heads_per_device, head_dim
                ).transpose(1, 2)
                key = key.view(
                    batch_size, -1, self.heads_per_device, head_dim
                ).transpose(1, 2)
                value = value.view(
                    batch_size, -1, self.heads_per_device, head_dim
                ).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.module.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )

                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, self.heads_per_device * head_dim
                )
        #! ---------------------------------------- ATTENTION ----------------------------------------
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        return hidden_states, encoder_hidden_states


@hetuDiTAttentionProcessorRegister.register(FluxAttnProcessor2_0)
class hetuDiTFluxAttnProcessor2_0(FluxAttnProcessor2_0):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            from hetu_dit.core.parallel import (
                hetuDiTFluxLongContextAttention,
                hetuDiTUlyssesAttention,
            )

            if HAS_FLASH_ATTN:
                self.hybrid_seq_parallel_attn = hetuDiTFluxLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache
                )
            else:
                self.hybrid_seq_parallel_attn = hetuDiTUlyssesAttention(
                    use_fa=False,
                    use_kv_cache=self.use_long_ctx_attn_kvcache,
                )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1
        )
        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        self.heads_per_device = attn.heads // get_tensor_model_parallel_world_size()
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads_per_device

        query = query.view(batch_size, -1, self.heads_per_device, head_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, -1, self.heads_per_device, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads_per_device, head_dim).transpose(
            1, 2
        )

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, self.heads_per_device, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, self.heads_per_device, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, self.heads_per_device, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            num_encoder_hidden_states_tokens = encoder_hidden_states_query_proj.shape[2]
            num_query_tokens = query.shape[2]

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        else:
            num_encoder_hidden_states_tokens = (
                get_runtime_state().max_condition_sequence_length
            )
            num_query_tokens = query.shape[2] - num_encoder_hidden_states_tokens

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        #! ---------------------------------------- KV CACHE ----------------------------------------
        if not self.use_long_ctx_attn_kvcache:
            encoder_hidden_states_key_proj, key = key.split(
                [num_encoder_hidden_states_tokens, num_query_tokens], dim=2
            )
            encoder_hidden_states_value_proj, value = value.split(
                [num_encoder_hidden_states_tokens, num_query_tokens], dim=2
            )
            key, value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[key, value],
                layer=attn,
                slice_dim=2,
                layer_type="attn",
            )
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        #! ---------------------------------------- KV CACHE ----------------------------------------

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            encoder_hidden_states_query_proj, query = query.split(
                [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
            )
            encoder_hidden_states_key_proj, key = key.split(
                [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
            )
            encoder_hidden_states_value_proj, value = value.split(
                [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
            )
            hidden_states = self.hybrid_seq_parallel_attn(
                attn,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                joint_tensor_query=encoder_hidden_states_query_proj,
                joint_tensor_key=encoder_hidden_states_key_proj,
                joint_tensor_value=encoder_hidden_states_value_proj,
                joint_strategy="front",
            )
            hidden_states = hidden_states.reshape(
                batch_size, -1, self.heads_per_device * head_dim
            )

        else:
            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(
                    batch_size, -1, self.heads_per_device * head_dim
                )
            else:
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, self.heads_per_device * head_dim
                )
        #! ---------------------------------------- ATTENTION ----------------------------------------

        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


@hetuDiTAttentionProcessorRegister.register(CogVideoXAttnProcessor2_0)
class hetuDiTCogVideoXAttnProcessor2_0(CogVideoXAttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            from hetu_dit.core.parallel import (
                hetuDiTJointLongContextAttention,
                hetuDiTLongContextAttention,
            )

            if HAS_FLASH_ATTN and get_runtime_state().split_text_embed_in_sp:
                self.hybrid_seq_parallel_attn = hetuDiTLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache
                )
            else:
                self.hybrid_seq_parallel_attn = hetuDiTJointLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache
                )
        else:
            self.hybrid_seq_parallel_attn = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1
        )
        global come_int_time  # TODO: DEL it
        text_seq_length = encoder_hidden_states.size(1)
        latent_seq_length = hidden_states.size(1)
        self.heads_per_device = attn.heads // get_tensor_model_parallel_world_size()
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads_per_device

        query = query.view(batch_size, -1, self.heads_per_device, head_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, -1, self.heads_per_device, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads_per_device, head_dim).transpose(
            1, 2
        )

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if (
            get_pipeline_parallel_world_size() == 1
            and get_runtime_state().split_text_embed_in_sp
        ):
            hidden_states = self.hybrid_seq_parallel_attn(
                query, key, value, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads_per_device * head_dim
            )
        elif HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            if get_runtime_state().split_text_embed_in_sp:
                encoder_query = None
                encoder_key = None
                encoder_value = None
            else:
                encoder_query = query[:, :, :text_seq_length, :]
                query = query[:, :, text_seq_length:, :]
                encoder_key = key[:, :, :text_seq_length, :]
                key = key[:, :, text_seq_length:, :]
                encoder_value = value[:, :, :text_seq_length, :]
                value = value[:, :, text_seq_length:, :]

                encoder_query = encoder_query.transpose(1, 2)
                encoder_key = encoder_key.transpose(1, 2)
                encoder_value = encoder_value.transpose(1, 2)

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            hidden_states = self.hybrid_seq_parallel_attn(
                None,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                joint_tensor_query=encoder_query,
                joint_tensor_key=encoder_key,
                joint_tensor_value=encoder_value,
                joint_strategy="front",
            )

            hidden_states = hidden_states.reshape(
                batch_size, -1, self.heads_per_device * head_dim
            )
        else:
            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(
                    batch_size, -1, self.heads_per_device * head_dim
                )

            else:
                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, self.heads_per_device * head_dim
                )
        #! ---------------------------------------- ATTENTION ----------------------------------------

        assert text_seq_length + latent_seq_length == hidden_states.shape[1]
        return hidden_states


@hetuDiTAttentionProcessorRegister.register(HunyuanAttnProcessor2_0)
class hetuDiTHunyuanAttnProcessor2_0(HunyuanAttnProcessor2_0):
    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            from hetu_dit.core.parallel import (
                hetuDiTLongContextAttention,
                hetuDiTUlyssesAttention,
            )

            if HAS_FLASH_ATTN:
                self.hybrid_seq_parallel_attn = hetuDiTLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache
                )
            else:
                self.hybrid_seq_parallel_attn = hetuDiTUlyssesAttention(
                    use_fa=False,
                    use_kv_cache=self.use_long_ctx_attn_kvcache,
                )
        else:
            self.hybrid_seq_parallel_attn = None

    # NOTE() torch.compile dose not works for V100
    @torch_compile_disable_if_v100
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        latte_temporal_attention: Optional[bool] = False,
    ) -> torch.Tensor:
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1
        )
        self.heads_per_device = attn.heads // get_tensor_model_parallel_world_size()
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, self.heads_per_device, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads_per_device

        query = query.view(batch_size, -1, self.heads_per_device, head_dim).transpose(
            1, 2
        )

        key = key.view(batch_size, -1, self.heads_per_device, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads_per_device, head_dim).transpose(
            1, 2
        )

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        #! ---------------------------------------- KV CACHE ----------------------------------------
        if not self.use_long_ctx_attn_kvcache:
            key, value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[key, value],
                layer=attn,
                slice_dim=2,
                layer_type="attn",
            )
        #! ---------------------------------------- KV CACHE ----------------------------------------

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if (
            HAS_LONG_CTX_ATTN
            and get_sequence_parallel_world_size() > 1
            and not attn.is_cross_attention
            and not latte_temporal_attention
        ):
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            hidden_states = self.hybrid_seq_parallel_attn(
                attn,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                joint_strategy="none",
            )
            hidden_states = hidden_states.reshape(
                batch_size, -1, self.heads_per_device * head_dim
            )

        else:
            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(
                    batch_size, -1, self.heads_per_device * head_dim
                )

            else:
                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.module.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )

                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, self.heads_per_device * head_dim
                )
        #! ---------------------------------------- ATTENTION ----------------------------------------

        hidden_states = hidden_states.to(query.dtype)

        return hidden_states, residual


if HunyuanVideoAttnProcessor2_0 is not None:

    @hetuDiTAttentionProcessorRegister.register(HunyuanVideoAttnProcessor2_0)
    class hetuDiTHunyuanVideoAttnProcessor2_0(HunyuanVideoAttnProcessor2_0):
        def __init__(self):
            super().__init__()
            use_long_ctx_attn_kvcache = True
            self.use_long_ctx_attn_kvcache = (
                HAS_LONG_CTX_ATTN
                and use_long_ctx_attn_kvcache
                and get_sequence_parallel_world_size() > 1
            )
            if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
                from hetu_dit.core.parallel import (
                    hetuDiTJointLongContextAttention,
                    hetuDiTUlyssesAttention,
                )

                if HAS_FLASH_ATTN:
                    self.hybrid_seq_parallel_attn = hetuDiTJointLongContextAttention(
                        use_kv_cache=self.use_long_ctx_attn_kvcache
                    )
                else:
                    self.hybrid_seq_parallel_attn = hetuDiTUlyssesAttention(
                        use_fa=False,
                        use_kv_cache=self.use_long_ctx_attn_kvcache,
                    )
            else:
                self.hybrid_seq_parallel_attn = None

        def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
        ) -> torch.Tensor:
            self.use_long_ctx_attn_kvcache = (
                HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1
            )
            batch_size, _, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
            self.heads_per_device = attn.heads // get_tensor_model_parallel_world_size()
            # 1. QKV projections
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            query = query.unflatten(2, (self.heads_per_device, -1)).transpose(1, 2)
            key = key.unflatten(2, (self.heads_per_device, -1)).transpose(1, 2)
            value = value.unflatten(2, (self.heads_per_device, -1)).transpose(1, 2)

            # 2. QK normalization
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            # 3. Rotational positional embeddings applied to latent stream
            if image_rotary_emb is not None:
                if attn.add_q_proj is None and encoder_hidden_states is not None:
                    query = torch.cat(
                        [
                            apply_rotary_emb(
                                query[:, :, : -encoder_hidden_states.shape[1]],
                                image_rotary_emb,
                            ),
                            query[:, :, -encoder_hidden_states.shape[1] :],
                        ],
                        dim=2,
                    )
                    key = torch.cat(
                        [
                            apply_rotary_emb(
                                key[:, :, : -encoder_hidden_states.shape[1]],
                                image_rotary_emb,
                            ),
                            key[:, :, -encoder_hidden_states.shape[1] :],
                        ],
                        dim=2,
                    )
                else:
                    query = apply_rotary_emb(query, image_rotary_emb)
                    key = apply_rotary_emb(key, image_rotary_emb)

            # 4. Encoder condition QKV projection and normalization
            if attn.add_q_proj is not None and encoder_hidden_states is not None:
                encoder_query = attn.add_q_proj(encoder_hidden_states)
                encoder_key = attn.add_k_proj(encoder_hidden_states)
                encoder_value = attn.add_v_proj(encoder_hidden_states)

                encoder_query = encoder_query.unflatten(
                    2, (self.heads_per_device, -1)
                ).transpose(1, 2)
                encoder_key = encoder_key.unflatten(
                    2, (self.heads_per_device, -1)
                ).transpose(1, 2)
                encoder_value = encoder_value.unflatten(
                    2, (self.heads_per_device, -1)
                ).transpose(1, 2)

                if attn.norm_added_q is not None:
                    encoder_query = attn.norm_added_q(encoder_query)
                if attn.norm_added_k is not None:
                    encoder_key = attn.norm_added_k(encoder_key)

                query = torch.cat([query, encoder_query], dim=2)
                key = torch.cat([key, encoder_key], dim=2)
                value = torch.cat([value, encoder_value], dim=2)

            if encoder_hidden_states is not None:
                num_encoder_hidden_states_tokens = encoder_hidden_states.shape[1]
                num_query_tokens = query.shape[2] - num_encoder_hidden_states_tokens
            else:
                num_encoder_hidden_states_tokens = (
                    get_runtime_state().max_condition_sequence_length
                )
                num_query_tokens = query.shape[2] - num_encoder_hidden_states_tokens

            #! ---------------------------------------- ATTENTION ----------------------------------------
            if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
                if get_runtime_state().split_text_embed_in_sp:
                    encoder_query = None
                    encoder_key = None
                    encoder_value = None
                else:
                    query, encoder_query = query.split(
                        [num_query_tokens, num_encoder_hidden_states_tokens], dim=2
                    )
                    key, encoder_key = key.split(
                        [num_query_tokens, num_encoder_hidden_states_tokens], dim=2
                    )
                    value, encoder_value = value.split(
                        [num_query_tokens, num_encoder_hidden_states_tokens], dim=2
                    )

                    encoder_query = encoder_query.transpose(1, 2)
                    encoder_key = encoder_key.transpose(1, 2)
                    encoder_value = encoder_value.transpose(1, 2)

                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

                hidden_states = self.hybrid_seq_parallel_attn(
                    None,
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    causal=False,
                    joint_tensor_query=encoder_query,
                    joint_tensor_key=encoder_key,
                    joint_tensor_value=encoder_value,
                    joint_strategy="rear",
                )

                hidden_states = hidden_states.flatten(2, 3)
            else:
                if HAS_FLASH_ATTN:
                    from flash_attn import flash_attn_func

                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    hidden_states = flash_attn_func(
                        query, key, value, dropout_p=0.0, causal=False
                    )
                    hidden_states = hidden_states.flatten(2, 3)

                else:
                    # the output of sdp = (batch, num_heads, seq_len, head_dim)
                    # TODO: add support for attn.scale when we move to Torch 2.1
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, dropout_p=0.0, is_causal=False
                    )
                    hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

            hidden_states = hidden_states.to(query.dtype)

            return hidden_states
else:
    hetuDiTHunyuanVideoAttnProcessor2_0 = None
