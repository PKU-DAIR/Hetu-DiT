# Copyright 2024 The Hunyuan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Union, Type

import torch
import torch.nn as nn

from diffusers.models.transformers.transformer_hunyuan_video import (
    HunyuanVideoTransformer3DModel,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from diffusers.utils import (
    scale_lora_layers,
    USE_PEFT_BACKEND,
    unscale_lora_layers,
)

from hetu_dit.core.distributed import (
    get_world_group,
    get_runtime_state,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
)
from hetu_dit.core.distributed.runtime_state import get_runtime_state
from hetu_dit.logger import init_logger
from hetu_dit.model_executor.utils.register_warpper import (
    hetuDiTTransformerWrappersRegister,
)
from hetu_dit.model_executor.diffusion_executor.models.transformers.base_transformer import (
    hetuDiTDiffusionTransformerBaseWrapper,
)

logger = init_logger(__name__)
from diffusers.models.attention import FeedForward
from hetu_dit.model_executor.utils.register_warpper import hetuDiTLayerWrappersRegister
from hetu_dit.model_executor.diffusion_executor.layers import *


@hetuDiTTransformerWrappersRegister.register(HunyuanVideoTransformer3DModel)
class hetuDiTHunyuanVideoTransformer3DWrapper(hetuDiTDiffusionTransformerBaseWrapper):
    def __init__(
        self,
        transformer: HunyuanVideoTransformer3DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=([FeedForward]),
            submodule_name_to_wrap=["attn"],
            transformer_blocks_name=["transformer_blocks", "single_transformer_blocks"],
        )

    def _wrap_layers(
        self,
        model: Optional[nn.Module] = None,
        submodule_classes_to_wrap: List[Type] = [],
        submodule_name_to_wrap: List[str] = [],
        submodule_addition_args: Dict[str, Dict] = {},
    ) -> Union[nn.Module, None]:
        wrapped_layers = []
        wrap_self_module = False
        if model is None:
            wrap_self_module = True
            model = self.module

        for name, module in model.named_modules():
            if isinstance(module, hetuDiTDiffusionLayerBaseWrapper):
                continue

            for subname, submodule in module.named_children():
                need_wrap = subname in submodule_name_to_wrap
                for class_to_wrap in submodule_classes_to_wrap:
                    if isinstance(submodule, class_to_wrap):
                        need_wrap = True
                        break

                if need_wrap and "context_embedder" not in f"{name}.{subname}":
                    wrapper = hetuDiTLayerWrappersRegister.get_wrapper(submodule)
                    additional_args = submodule_addition_args.get(subname, {})
                    logger.info(
                        f"[RANK {get_world_group().rank}] "
                        f"Wrapping {name}.{subname} in model class "
                        f"{model.__class__.__name__} with "
                        f"{wrapper.__name__}"
                    )
                    if additional_args != {}:
                        setattr(
                            module,
                            subname,
                            wrapper(
                                submodule,
                                **additional_args,
                            ),
                        )
                    else:
                        setattr(
                            module,
                            subname,
                            wrapper(submodule),
                        )
                    wrapped_layers.append(getattr(module, subname))
        self.wrapped_layers = wrapped_layers
        if wrap_self_module:
            self.module = model
        else:
            return model

    @classmethod
    def _static_wrap_layers(
        cls,
        model: nn.Module,
        submodule_classes_to_wrap: List[Type] = [],
        submodule_name_to_wrap: List[str] = [],
        submodule_addition_args: Dict[str, Dict] = {},
    ) -> Union[nn.Module, None]:
        wrapped_layers = []

        for name, module in model.named_modules():
            if isinstance(module, hetuDiTDiffusionLayerBaseWrapper):
                continue

            for subname, submodule in module.named_children():
                need_wrap = subname in submodule_name_to_wrap
                for class_to_wrap in submodule_classes_to_wrap:
                    if isinstance(submodule, class_to_wrap):
                        need_wrap = True
                        break

                if need_wrap and "context_embedder" not in f"{name}.{subname}":
                    wrapper = hetuDiTLayerWrappersRegister.get_wrapper(submodule)
                    additional_args = submodule_addition_args.get(subname, {})
                    logger.info(
                        # f"[RANK {get_world_group().rank}] "
                        f"Wrapping {name}.{subname} in model class "
                        f"{model.__class__.__name__} with "
                        f"{wrapper.__name__}"
                    )
                    if additional_args != {}:
                        setattr(
                            module,
                            subname,
                            wrapper(
                                submodule,
                                **additional_args,
                            ),
                        )
                    else:
                        setattr(
                            module,
                            subname,
                            wrapper(submodule),
                        )
                    wrapped_layers.append(getattr(module, subname))
        return model

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        assert batch_size % get_classifier_free_guidance_world_size() == 0, (
            f"Cannot split dim 0 of hidden_states ({batch_size}) into {get_classifier_free_guidance_world_size()} parts."
        )

        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states, timestep, encoder_attention_mask
        )

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1
        )
        hidden_states = hidden_states.flatten(1, 3)

        hidden_states = torch.chunk(
            hidden_states, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(
            hidden_states, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]

        encoder_attention_mask = encoder_attention_mask[0].to(torch.bool)
        encoder_hidden_states_indices = torch.arange(
            encoder_hidden_states.shape[1], device=encoder_hidden_states.device
        )
        encoder_hidden_states_indices = encoder_hidden_states_indices[
            encoder_attention_mask
        ]
        encoder_hidden_states = encoder_hidden_states[
            ..., encoder_hidden_states_indices, :
        ]

        get_runtime_state().split_text_embed_in_sp = (
            False  # TODO: need to change, force to be false
        )
        encoder_hidden_states = torch.chunk(
            encoder_hidden_states, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(
                encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2
            )[get_sequence_parallel_rank()]

        freqs_cos, freqs_sin = image_rotary_emb

        def get_rotary_emb_chunk(freqs):
            freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=0)[
                get_sequence_parallel_rank()
            ]
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos)
        freqs_sin = get_rotary_emb_chunk(freqs_sin)
        image_rotary_emb = (freqs_cos, freqs_sin)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}

            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        None,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        None,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, None, image_rotary_emb
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, None, image_rotary_emb
                )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            -1,
            p_t,
            p,
            p,
        )

        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
