from .base_cache import BaseBlockCache
from typing import List, Optional
from hetu_dit.logger import init_logger
import torch.nn as nn

logger = init_logger(__name__)


class TransformerBlockCache(BaseBlockCache):
    def __init__(
        self,
        global_blocks_limit: int = 0,
        worker_instance: "Worker" = None,
        all_worker_handles=[],
    ):
        super().__init__(global_blocks_limit, worker_instance, all_worker_handles)
        self.model_block_configs = {
            "flux": {
                "block_names": ["transformer_blocks", "single_transformer_blocks"],
                "block_counts": [19, 38],
            },
            "hunyuanvideo": {
                "block_names": ["transformer_blocks", "single_transformer_blocks"],
                "block_counts": [20, 40],
            },
            "hunyuandit": {"block_names": ["blocks"], "block_counts": [40]},
        }
        self.current_model_type = None
        self.block_mapping = {}

    def _detect_model_type(self, cpu_full_transformer: nn.Module) -> str:
        """Detect model type based on transformer structure."""
        if hasattr(cpu_full_transformer, "single_transformer_blocks"):
            if len(cpu_full_transformer.transformer_blocks) == 19:
                return "flux"
            elif len(cpu_full_transformer.transformer_blocks) == 20:
                return "hunyuanvideo"
        elif hasattr(cpu_full_transformer, "blocks"):
            if len(cpu_full_transformer.blocks) == 40:
                return "hunyuandit"
        return "default"

    def _build_block_mapping(self, cpu_full_transformer: nn.Module):
        """Build mapping from global index to (block_type, local_index)."""
        self.block_mapping = {}
        global_idx = 0

        if self.current_model_type in self.model_block_configs:
            config = self.model_block_configs[self.current_model_type]
            for block_name, block_count in zip(
                config["block_names"], config["block_counts"]
            ):
                for local_idx in range(block_count):
                    self.block_mapping[global_idx] = (block_name, local_idx)
                    global_idx += 1
        else:
            # Default behavior for models with only transformer_blocks
            if hasattr(cpu_full_transformer, "transformer_blocks"):
                for local_idx in range(len(cpu_full_transformer.transformer_blocks)):
                    self.block_mapping[global_idx] = ("transformer_blocks", local_idx)
                    global_idx += 1

    def _ensure_mapping(self, cpu_full_transformer: nn.Module):
        if self.current_model_type is None:
            self.current_model_type = self._detect_model_type(cpu_full_transformer)
            self._build_block_mapping(cpu_full_transformer)
        elif not self.block_mapping:
            self._build_block_mapping(cpu_full_transformer)

    def get_total_blocks(self, cpu_full_transformer: nn.Module) -> int:
        self._ensure_mapping(cpu_full_transformer)
        if self.current_model_type in self.model_block_configs:
            return sum(
                self.model_block_configs[self.current_model_type]["block_counts"]
            )
        return len(cpu_full_transformer.transformer_blocks)

    def get_cpu_block_for_index(
        self, cpu_full_transformer: nn.Module, absolute_block_index: int
    ) -> nn.Module:
        self._ensure_mapping(cpu_full_transformer)
        if absolute_block_index not in self.block_mapping:
            raise IndexError(
                f"Block index {absolute_block_index} not found in block mapping"
            )
        block_type, local_idx = self.block_mapping[absolute_block_index]
        block_list = getattr(cpu_full_transformer, block_type)
        if local_idx >= len(block_list):
            raise IndexError(
                f"Local index {local_idx} out of range for {block_type} (size {len(block_list)})"
            )
        return block_list[local_idx]

    def place_blocks_on_model(
        self,
        transformer: nn.Module,
        blocks_list: List[nn.Module],
        required_indices: List[int],
        cpu_full_transformer: nn.Module,
    ) -> None:
        self._ensure_mapping(cpu_full_transformer)
        if self.current_model_type in self.model_block_configs:
            config = self.model_block_configs[self.current_model_type]
            blocks_by_type = {name: [] for name in config["block_names"]}
            for i, global_idx in enumerate(required_indices):
                block_type, _ = self.block_mapping[global_idx]
                blocks_by_type[block_type].append(blocks_list[i])
            for block_name, blocks in blocks_by_type.items():
                setattr(transformer, block_name, nn.ModuleList(blocks))
        else:
            setattr(transformer, "transformer_blocks", nn.ModuleList(blocks_list))

    def choose_tp_split_type(self, name: str) -> Optional[str]:
        adjust_names = {
            "attn.module.to_q",
            "attn.module.to_k",
            "attn.module.to_v",
            "attn.module.add_k_proj",
            "attn.module.add_q_proj",
            "attn.module.add_v_proj",
            "attn.module.to_add_out",
            "attn.module.to_out.0",
            "attn1.module.to_q",
            "attn1.module.to_k",
            "attn1.module.to_v",
            "attn1.module.add_k_proj",
            "attn1.module.add_q_proj",
            "attn1.module.add_v_proj",
            "attn1.module.to_add_out",
            "attn1.module.to_out.0",
            "ff.module.net.0.proj",
            "ff.module.net.2",
            "ff_context.module.net.0.proj",
            "ff_context.module.net.2",
        }
        return (
            self.determine_tp_split_type_by_name(name) if name in adjust_names else None
        )

    def determine_tp_split_type_by_name(self, name: str) -> str:
        """Determine TP split type for transformer layers."""
        lower_name = name.lower()
        if any(
            x in lower_name
            for x in ["to_q", "to_k", "to_v", "add_k_proj", "add_q_proj", "add_v_proj"]
        ):
            return "row"
        if any(x in lower_name for x in ["to_out", "to_add_out"]):
            return "column"
        if "net.0.proj" in lower_name:
            return "row"
        if "net.2" in lower_name:
            return "column"
        return "row"

    def get_block_from_cpu(
        self, block_index: int, cpu_full_transformer: nn.Module
    ) -> nn.Module:
        """Get transformer block from CPU model."""
        # Detect model type and build mapping if not done
        if self.current_model_type is None:
            self.current_model_type = self._detect_model_type(cpu_full_transformer)
            self._build_block_mapping(cpu_full_transformer)

        total_blocks = self.get_total_blocks(cpu_full_transformer)
        if not (0 <= block_index < total_blocks):
            raise IndexError(
                f"Block index {block_index} out of range for cpu_full_transformer (total size {total_blocks})"
            )

        # Get block type and local index
        if block_index not in self.block_mapping:
            raise IndexError(f"Block index {block_index} not found in block mapping")

        block_type, local_index = self.block_mapping[block_index]

        # Get the appropriate block list
        if hasattr(cpu_full_transformer, block_type):
            block_list = getattr(cpu_full_transformer, block_type)
            if local_index < len(block_list):
                cpu_block = block_list[local_index]
                return self.copy_cpu_block_to_gpu(cpu_block)
            else:
                raise IndexError(
                    f"Local index {local_index} out of range for {block_type} (size {len(block_list)})"
                )
        else:
            raise AttributeError(
                f"CPU transformer does not have attribute {block_type}"
            )
