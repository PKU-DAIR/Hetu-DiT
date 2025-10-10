from .base_cache import BaseBlockCache
from typing import Tuple, List, Optional, Dict
import torch.nn as nn
from hetu_dit.logger import init_logger

logger = init_logger(__name__)


class TextEncoderBlockCache(BaseBlockCache):
    def __init__(
        self,
        global_blocks_limit: int = 0,
        worker_instance: "Worker" = None,
        all_worker_handles=[],
    ):
        super().__init__(global_blocks_limit, worker_instance, all_worker_handles)
        self.model_block_configs = {
            "t5_encoder": {"block_names": ["block"]},
            "default": {"block_names": ["block"]},
        }
        self.current_model_type: Optional[str] = None
        self.block_mapping: Dict[int, Tuple[str, int]] = {}

    def _detect_model_type(self, cpu_full_text_encoder: nn.Module) -> str:
        if hasattr(cpu_full_text_encoder, "block") and isinstance(
            getattr(cpu_full_text_encoder, "block"), nn.ModuleList
        ):
            return "t5_encoder"
        return "default"

    def _build_block_mapping(self, cpu_full_text_encoder: nn.Module):
        self.block_mapping = {}
        global_idx = 0
        if hasattr(cpu_full_text_encoder, "block"):
            for local_idx in range(len(cpu_full_text_encoder.block)):
                self.block_mapping[global_idx] = ("block", local_idx)
                global_idx += 1

    def _ensure_mapping(self, cpu_full_text_encoder: nn.Module):
        if self.current_model_type is None:
            self.current_model_type = self._detect_model_type(cpu_full_text_encoder)
            self._build_block_mapping(cpu_full_text_encoder)
        elif not self.block_mapping:
            self._build_block_mapping(cpu_full_text_encoder)

    def get_total_blocks(self, cpu_full_text_encoder: nn.Module) -> int:
        self._ensure_mapping(cpu_full_text_encoder)
        return len(self.block_mapping)

    def get_cpu_block_for_index(
        self, cpu_full_text_encoder: nn.Module, absolute_block_index: int
    ) -> nn.Module:
        self._ensure_mapping(cpu_full_text_encoder)
        if absolute_block_index not in self.block_mapping:
            raise IndexError(
                f"Block index {absolute_block_index} not found in text encoder block mapping"
            )
        block_type, local_idx = self.block_mapping[absolute_block_index]
        if not hasattr(cpu_full_text_encoder, block_type):
            raise AttributeError(
                f"CPU text encoder does not have attribute '{block_type}'"
            )
        block_list = getattr(cpu_full_text_encoder, block_type)
        if local_idx >= len(block_list):
            raise IndexError(
                f"Local index {local_idx} out of range for {block_type} (size {len(block_list)})"
            )
        return block_list[local_idx]

    def place_blocks_on_model(
        self,
        text_encoder: nn.Module,
        blocks_list: List[nn.Module],
        required_indices: List[int],
        cpu_full_text_encoder: nn.Module,
    ) -> None:
        self._ensure_mapping(cpu_full_text_encoder)
        config = self.model_block_configs.get(
            self.current_model_type, self.model_block_configs["default"]
        )
        blocks_by_type: Dict[str, List[nn.Module]] = {
            name: [] for name in config["block_names"]
        }
        for i, global_idx in enumerate(required_indices):
            block_type, _ = self.block_mapping[global_idx]
            blocks_by_type[block_type].append(blocks_list[i])
        for block_name, blocks in blocks_by_type.items():
            setattr(text_encoder, block_name, nn.ModuleList(blocks))

    def choose_tp_split_type(self, name: str) -> Optional[str]:
        adjust_names = {
            "layer.0.SelfAttention.module.q",
            "layer.0.SelfAttention.module.k",
            "layer.0.SelfAttention.module.v",
            "layer.0.SelfAttention.module.o",
            "layer.1.module.DenseReluDense.wi_0",
            "layer.1.module.DenseReluDense.wi_1",
            "layer.1.module.DenseReluDense.wo",
        }
        return (
            self.determine_tp_split_type_by_name(name) if name in adjust_names else None
        )

    def determine_tp_split_type_by_name(self, name: str) -> str:
        lower_name = name.lower()
        if any(x in lower_name for x in [".q", ".k", ".v"]):
            return "row"
        if ".o" in lower_name:
            return "column"
        if ".wi_1" in lower_name or ".wi_0" in lower_name:
            return "row"
        if ".wo" in lower_name:
            return "column"
        return "row"

    def get_block_from_cpu(
        self, block_index: int, cpu_full_text_encoder: nn.Module
    ) -> nn.Module:
        if self.current_model_type is None or not self.block_mapping:
            self.current_model_type = self._detect_model_type(cpu_full_text_encoder)
            self._build_block_mapping(cpu_full_text_encoder)

        total_blocks = self.get_total_blocks(cpu_full_text_encoder)
        if not (0 <= block_index < total_blocks):
            raise IndexError(
                f"Block index {block_index} out of range for cpu_full_text_encoder (total size {total_blocks})"
            )

        if block_index not in self.block_mapping:
            raise IndexError(
                f"Block index {block_index} not found in text encoder block mapping"
            )

        block_type, local_index = self.block_mapping[block_index]
        if not hasattr(cpu_full_text_encoder, block_type):
            raise AttributeError(
                f"CPU text encoder does not have attribute '{block_type}'"
            )

        block_list = getattr(cpu_full_text_encoder, block_type)
        if local_index >= len(block_list):
            raise IndexError(
                f"Local index {local_index} out of range for {block_type} (size {len(block_list)})"
            )

        cpu_block = block_list[local_index]
        return self.copy_cpu_block_to_gpu(cpu_block)
