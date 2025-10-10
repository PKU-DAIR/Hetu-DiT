import torch
from torch import nn
from torch.nn import functional as F

from hetu_dit.core.distributed.parallel_state import get_sequence_parallel_world_size
from hetu_dit.core.distributed.runtime_state import get_runtime_state
from hetu_dit.model_executor.diffusion_executor.layers import (
    hetuDiTDiffusionLayerBaseWrapper,
)
from hetu_dit.logger import init_logger
from hetu_dit.model_executor.utils.register_warpper import hetuDiTLayerWrappersRegister
from hetu_dit.core.distributed import (
    get_pipeline_parallel_world_size,
)

logger = init_logger(__name__)


@hetuDiTLayerWrappersRegister.register(nn.Conv2d)
class hetuDiTConv2dWrapper(hetuDiTDiffusionLayerBaseWrapper):
    def __init__(
        self,
        conv2d: nn.Conv2d,
        *,
        is_first_layer: bool = True,
    ):
        super().__init__(
            module=conv2d,
        )
        self.is_first_layer = is_first_layer

    def naive_forward(self, x: torch.Tensor) -> torch.Tensor:
        #  x: [B, C, H, W]
        output = self.module(x)
        return output

    # TODO fix implementation problems in sliced_forward
    # only available for patchify process
    def sliced_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        stride = self.module.stride[0]
        padding = self.module.padding[0]

        idx = get_runtime_state().pipeline_patch_idx
        pp_patches_start_idx_local = get_runtime_state().pp_patches_start_idx_local
        h_begin = pp_patches_start_idx_local[idx] - padding
        h_end = pp_patches_start_idx_local[idx + 1] + padding
        final_padding = [padding, padding, 0, 0]
        if h_begin < 0:
            h_begin = 0
            final_padding[2] = padding
        if h_end > h:
            h_end = h
            final_padding[3] = padding
        sliced_input = x[:, :, h_begin:h_end, :]
        padded_input = F.pad(sliced_input, final_padding, mode="constant")
        result = F.conv2d(
            padded_input,
            self.module.weight,
            self.module.bias,
            stride=stride,
            padding="valid",
        )
        return result

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if (
            (
                get_pipeline_parallel_world_size() == 1
                and get_sequence_parallel_world_size() == 1
            )
            or self.module.kernel_size == (1, 1)
            or self.module.kernel_size == 1
        ):
            output = self.naive_forward(x)
        else:
            if self.is_first_layer:
                if (
                    not get_runtime_state().patch_mode
                    or get_runtime_state().num_pipeline_patch == 1
                ):
                    self.activation_cache = x
                    output = self.naive_forward(self.activation_cache)
                else:
                    if self.activation_cache is None:
                        self.activation_cache = torch.zeros(
                            [
                                x.shape[0],
                                x.shape[1],
                                get_runtime_state().pp_patches_start_idx_local[-1],
                                x.shape[3],
                            ],
                            dtype=x.dtype,
                            device=x.device,
                        )

                    self.activation_cache[
                        :,
                        :,
                        get_runtime_state().pp_patches_start_idx_local[
                            get_runtime_state().pipeline_patch_idx
                        ] : get_runtime_state().pp_patches_start_idx_local[
                            get_runtime_state().pipeline_patch_idx + 1
                        ],
                        :,
                    ] = x
                    output = self.sliced_forward(self.activation_cache)

            else:
                raise NotImplementedError

        return output
