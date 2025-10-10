from transformers.models.t5.modeling_t5 import (
    T5LayerFF,
    T5DenseGatedActDense,
    T5DenseActDense,
)
from torch import nn
from hetu_dit.core.distributed.parallel_state import (
    get_text_encoder_tensor_model_parallel_world_size,
    get_text_encoder_tensor_model_parallel_rank,
    get_text_encoder_tp_group,
)
import torch
from hetu_dit.model_executor.utils.register_warpper import hetuDiTLayerWrappersRegister
from hetu_dit.model_executor.textencoder_executor.layers.base_layer import (
    hetuDiTTextEncoderLayerBaseWrapper,
)


@hetuDiTLayerWrappersRegister.register(T5LayerFF)
class hetuDiTT5FFWrapper(hetuDiTTextEncoderLayerBaseWrapper):
    def __init__(self, feedforward: T5LayerFF):
        super(hetuDiTT5FFWrapper, self).__init__(module=feedforward)

        tp_degree = get_text_encoder_tensor_model_parallel_world_size()
        tp_rank = get_text_encoder_tensor_model_parallel_rank()

        if isinstance(self.module.DenseReluDense, T5DenseGatedActDense):
            # Split weights for wi_0
            self.module.DenseReluDense.wi_0.weight.data = (
                self.module.DenseReluDense.wi_0.weight.data.chunk(tp_degree, dim=0)[
                    tp_rank
                ]
            )
            if self.module.DenseReluDense.wi_0.bias is not None:
                self.module.DenseReluDense.wi_0.bias.data = (
                    self.module.DenseReluDense.wi_0.bias.data.chunk(tp_degree, dim=0)[
                        tp_rank
                    ]
                )

            # Split weights for wi_1
            self.module.DenseReluDense.wi_1.weight.data = (
                self.module.DenseReluDense.wi_1.weight.data.chunk(tp_degree, dim=0)[
                    tp_rank
                ]
            )
            if self.module.DenseReluDense.wi_1.bias is not None:
                self.module.DenseReluDense.wi_1.bias.data = (
                    self.module.DenseReluDense.wi_1.bias.data.chunk(tp_degree, dim=0)[
                        tp_rank
                    ]
                )

        elif isinstance(self.module.DenseReluDense, T5DenseActDense):
            # Split weights for wi
            self.module.DenseReluDense.wi.weight.data = (
                self.module.DenseReluDense.wi.weight.data.chunk(tp_degree, dim=0)[
                    tp_rank
                ]
            )
            if self.module.DenseReluDense.wi.bias is not None:
                self.module.DenseReluDense.wi.bias.data = (
                    self.module.DenseReluDense.wi.bias.data.chunk(tp_degree, dim=0)[
                        tp_rank
                    ]
                )

        else:
            raise TypeError(
                f"activation_fn {type(isinstance(self.module.net[0]))} not supported"
            )

        # Split weights for wo
        self.module.DenseReluDense.wo.weight.data = (
            self.module.DenseReluDense.wo.weight.data.chunk(tp_degree, dim=1)[tp_rank]
        )

        # Handle output bias for `wo`
        self.has_output_bias = False
        if self.module.DenseReluDense.wo.bias is not None:
            self.register_parameter(
                "output_bias",
                nn.Parameter(self.module.DenseReluDense.wo.bias.data.clone()),
            )
            self.module.DenseReluDense.wo.bias = None
            self.has_output_bias = True

        self.module.DenseReluDense.forward = self.new_DenseReluDense_fwd
        torch.cuda.empty_cache()

    def new_DenseReluDense_fwd(self, hidden_states):
        hidden_gelu = self.module.DenseReluDense.act(
            self.module.DenseReluDense.wi_0(hidden_states)
        )
        hidden_linear = self.module.DenseReluDense.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.module.DenseReluDense.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.module.DenseReluDense.wo.weight.dtype
            and self.module.DenseReluDense.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.module.DenseReluDense.wo.weight.dtype)

        hidden_states = self.module.DenseReluDense.wo(hidden_states)

        get_text_encoder_tp_group().all_reduce(hidden_states)
        if self.has_output_bias:
            hidden_states += self.output_bias
        return hidden_states

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # hidden_states = self.module(hidden_states, *args, **kwargs)
        forwarded_states = self.module.layer_norm(hidden_states)
        forwarded_states = self.module.DenseReluDense(forwarded_states)

        forwarded_states = self.module.DenseReluDense.dropout(forwarded_states)

        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states
