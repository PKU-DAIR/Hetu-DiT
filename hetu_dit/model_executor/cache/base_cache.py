from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import torch
import torch.nn as nn
import copy
from hetu_dit.logger import init_logger
from hetu_dit.core.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from hetu_dit.core.distributed.runtime_state import get_runtime_state
import asyncio

logger = init_logger(__name__)


@dataclass(eq=True)
class NeededPiece:
    """
    Describes a piece of data that needs to be reconstructed.
    """

    uid: int  # Globally unique ID within a single adjustment
    block_index: int
    layer_path: str  # Path within the module, e.g., "attn1.module.to_q"
    tensor_type: str  # "weight" or "bias"
    split_dim: int  # The dimension to split on
    full_size: int  # Size of the full tensor in the split dimension
    abs_start_idx: int  # Absolute start index in the full tensor
    abs_end_idx: int  # Absolute end index in the full tensor
    dest_offset: int  # Write offset in the destination GPU tensor

    source_rank: int = field(default=-1)
    status: str = field(default="needed")  # Status: needed, sourced, transferring, done
    component: Optional[str] = field(default=None, compare=False)


@dataclass
class DataReconstructionPlan:
    """Describes the reconstruction plan for a complete block."""

    block_index: int
    target_tp_range: Tuple[float, float]
    skeleton_block: nn.Module
    pieces: List[NeededPiece] = field(default_factory=list)


class BaseBlockCache(ABC):
    """Base class for GPU block cache with LRU eviction policy."""

    def __init__(
        self,
        global_blocks_limit: int = 0,
        worker_instance: "Worker" = None,
        all_worker_handles: List["Worker"] = [],
    ):
        self.global_blocks_limit = global_blocks_limit
        self.cache: OrderedDict[int, Tuple[nn.Module, Tuple[float, float]]] = (
            OrderedDict()
        )
        self.active_block_indices: List[int] = []
        self.worker = worker_instance
        self.all_worker_handles = all_worker_handles
        logger.info(
            f"{self.__class__.__name__} initialized with limit {global_blocks_limit}"
        )

    def get_total_gpu_blocks(self) -> int:
        """Returns the total number of blocks on the current GPU (active + cached)."""
        return len(self.cache)

    def get_active_blocks_count(self) -> int:
        """Returns the number of currently active blocks."""
        return len(self.active_block_indices)

    def get_cached_only_blocks_count(self) -> int:
        """Returns the number of cached-only (non-active) blocks."""
        return len(self.cache) - len(self.active_block_indices)

    def register_initial_blocks(
        self, initial_gpu_blocks: nn.ModuleList, absolute_start_index: int
    ):
        """Register initial GPU blocks into cache."""
        logger.info(
            f"Registering {len(initial_gpu_blocks)} initial blocks starting from absolute index {absolute_start_index}"
        )
        current_active_indices = []

        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size > 0:
            initial_tp_range = (tp_rank / tp_world_size, (tp_rank + 1) / tp_world_size)
        else:
            initial_tp_range = (0.0, 1.0)

        for i, block_module in enumerate(initial_gpu_blocks):
            absolute_index = absolute_start_index + i
            self._add_or_update(absolute_index, block_module, initial_tp_range)
            current_active_indices.append(absolute_index)

        self.active_block_indices = current_active_indices
        logger.info(
            f"Initial registration complete. Cache size: {len(self.cache)}, Active indices: {self.active_block_indices}"
        )

    def _evict(self):
        """
        Evicts the least recently used non-active block if the global limit is exceeded.
        No eviction occurs if all blocks are active, allowing active blocks to surpass the limit.
        """
        evicted_count = 0
        while len(self.cache) > self.global_blocks_limit:
            evicted_candidate_index = -1
            found_evictable = False
            for idx in list(self.cache.keys()):  # Iterate over keys to allow deletion
                if idx not in self.active_block_indices:
                    evicted_candidate_index = idx
                    found_evictable = True
                    break

            if found_evictable:
                # Retrieve module before deleting to potentially allow for cleanup if needed
                # module_to_evict, _ = self.cache[evicted_candidate_index]
                del self.cache[evicted_candidate_index]
                evicted_count += 1
                logger.info(
                    f"Evicted block {evicted_candidate_index} (LRU non-active) to meet limit."
                )
                # Explicitly clear cache might help here, but could be slow.
                # torch.cuda.empty_cache()
            else:
                logger.info(
                    f"All {len(self.cache)} blocks are active or cache within limit. No eviction performed."
                )
                break

    def _add_or_update(
        self,
        block_index: int,
        block_module: nn.Module,
        current_tp_range: Tuple[float, float],
    ):
        """Adds a block to the cache or updates its LRU state, also storing its TP range."""
        is_new = block_index not in self.cache
        if block_index in self.cache:
            # Update module and TP range, move to end
            self.cache[block_index] = (block_module, current_tp_range)
            self.cache.move_to_end(block_index, last=True)
        else:
            # Add new block with its TP range
            self.cache[block_index] = (block_module, current_tp_range)
            # Eviction check only relevant when adding *new* items that might exceed the limit
            if len(self.cache) > self.global_blocks_limit:
                self._evict()  # Eviction timing is logged inside _evict

        log_action = "Added" if is_new else "Updated LRU for"
        # Log duration including potential eviction
        logger.info(f"{log_action} block {block_index}.")

    @abstractmethod
    def get_block_from_cpu(
        self, block_index: int, cpu_full_model: nn.Module
    ) -> nn.Module:
        """Get a block from CPU model - subclass specific."""
        pass

    def get_block(
        self,
        block_index: int,
        cpu_full_model: nn.Module,
        target_tp_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[nn.Module, bool, Optional[Tuple[float, float]]]:
        """Gets the block for the specified index. Returns (block module, whether it was loaded from CPU, cached TP range or None)."""
        if block_index in self.cache:
            block_module, cached_tp_range = self.cache[block_index]
            self.cache.move_to_end(block_index, last=True)
            logger.info(f"Cache hit for block {block_index}.")
            return block_module, False, cached_tp_range
        # Miss -> get_block_from_cpu implemented by the subclass decides how to fetch the block
        gpu_block = self.get_block_from_cpu(block_index, cpu_full_model)
        loaded_tp_range = target_tp_range if target_tp_range is not None else (0.0, 1.0)
        self._add_or_update(block_index, gpu_block, loaded_tp_range)
        return gpu_block, True, None

    @abstractmethod
    def determine_tp_split_type_by_name(self, name: str) -> str:
        """Determine tensor parallel split type for layers."""
        pass

    @abstractmethod
    def get_total_blocks(self, cpu_full_model: nn.Module) -> int: ...

    @abstractmethod
    def get_cpu_block_for_index(
        self, cpu_full_model: nn.Module, absolute_block_index: int
    ) -> nn.Module: ...

    @abstractmethod
    def place_blocks_on_model(
        self,
        model: nn.Module,
        blocks_list: List[nn.Module],
        required_indices: List[int],
        cpu_full_model: nn.Module,
    ) -> None: ...

    @abstractmethod
    def choose_tp_split_type(self, name: str) -> Optional[str]:
        """Returns 'row'/'column' or None for layers that do not require adjustment."""

    def range_to_indices(
        self, rng: Tuple[float, float], total_size: int
    ) -> Tuple[int, int]:
        """Converts a proportional range to absolute indices."""
        if total_size == 0:
            return 0, 0
        start_prop, end_prop = rng
        start_idx = int(start_prop * total_size)
        end_idx = int(end_prop * total_size)
        start_idx = max(0, min(start_idx, total_size))
        end_idx = max(0, min(end_idx, total_size))
        if start_idx > end_idx:
            logger.warning(
                f"Calculated start_idx {start_idx} > end_idx {end_idx} for range {rng} and total_size {total_size}. Adjusting end_idx."
            )
            end_idx = start_idx
        elif end_prop == 1.0 and end_idx < total_size:
            logger.info(
                f"Adjusting end_idx from {end_idx} to {total_size} for end_prop=1.0"
            )
            end_idx = total_size

        return start_idx, end_idx

    def copy_cpu_block_to_gpu(self, cpu_block: nn.Module) -> nn.Module:
        """Copies a CPU block to the GPU."""
        gpu_block = copy.deepcopy(cpu_block)
        gpu_block = gpu_block.to("cuda")
        return gpu_block

    async def adjust_pipeline(
        self,
        transformer: Optional[nn.Module] = None,
        cpu_full_transformer: Optional[nn.Module] = None,
        old_pp_range: tuple = None,
        new_pp_range: tuple = None,
        old_tp_range: tuple = None,
        new_tp_range: tuple = None,
        slice_bias: bool = True,
        text_encoder: Optional[nn.Module] = None,
        cpu_full_text_encoder: Optional[nn.Module] = None,
        model: Optional[nn.Module] = None,
        cpu_full_model: Optional[nn.Module] = None,
        **kwargs,
    ) -> nn.Module:
        if model is None or cpu_full_model is None:
            if transformer is not None and cpu_full_transformer is not None:
                model = transformer
                cpu_full_model = cpu_full_transformer
                component_name = "Transformer"
            elif text_encoder is not None and cpu_full_text_encoder is not None:
                model = text_encoder
                cpu_full_model = cpu_full_text_encoder
                component_name = "TextEncoder"
            else:
                raise TypeError(
                    "async_adjust_pipeline requires either (transformer, cpu_full_transformer) or (text_encoder, cpu_full_text_encoder) or (model, cpu_full_model)"
                )

        result = await self._adjust_pipeline_legacy(
            model=model,
            cpu_full_model=cpu_full_model,
            old_pp_range=old_pp_range,
            new_pp_range=new_pp_range,
            old_tp_range=old_tp_range,
            new_tp_range=new_tp_range,
            slice_bias=slice_bias,
            component_name=component_name,
        )
        return result

    async def _adjust_pipeline_legacy(
        self,
        model: nn.Module,
        cpu_full_model: nn.Module,
        old_pp_range: tuple,
        new_pp_range: tuple,
        old_tp_range: tuple,
        new_tp_range: tuple = None,
        slice_bias: bool = True,
        component_name: Optional[str] = None,
    ) -> nn.Module:
        use_p2p = (
            getattr(get_runtime_state().runtime_config, "adjust_strategy", None)
            == "p2p"
            and hasattr(self, "worker")
            and self.worker is not None
            and hasattr(self.worker, "nixl_manager")
            and self.worker.nixl_manager is not None
        )

        total_blocks_cpu = self.get_total_blocks(cpu_full_model)
        new_pp_start, new_pp_end = self.range_to_indices(new_pp_range, total_blocks_cpu)
        new_required_indices = list(range(new_pp_start, new_pp_end))
        new_required_indices_set = set(new_required_indices)

        target_tp = new_tp_range if new_tp_range is not None else old_tp_range
        pp_indices_unchanged = (
            set(self.active_block_indices) == new_required_indices_set
        )
        if pp_indices_unchanged and old_tp_range == target_tp:
            logger.info(
                "Parallel configuration and required block indices unchanged, skipping adjustment."
            )
            return model

        logger.info(
            f"Adjusting pipeline. Old PP: {old_pp_range}, New PP: {new_pp_range}. Old TP: {old_tp_range}, Target TP: {target_tp}."
        )
        if not pp_indices_unchanged:
            logger.info(
                f"Required indices changing from {self.active_block_indices} to {new_required_indices}"
            )
        else:
            logger.info(
                f"Required indices unchanged, but TP range changing from {old_tp_range} to {target_tp}."
            )

        pre_plans: Dict[int, DataReconstructionPlan] = {}
        all_pieces: List[NeededPiece] = []
        piece_uid = 0
        reusable_tp_by_block: Dict[int, Tuple[float, float]] = {}

        # Calculate reusable tp_range (global proportion, independent of layer dimension) for each block by intersecting proportional ranges.
        for idx in new_required_indices:
            cached = self.cache.get(idx, None)
            cached_tp_range = cached[1] if cached is not None else None
            if cached_tp_range is not None:
                reuse_tp = (
                    max(cached_tp_range[0], target_tp[0]),
                    min(cached_tp_range[1], target_tp[1]),
                )
                if reuse_tp[1] > reuse_tp[0]:
                    reusable_tp_by_block[idx] = reuse_tp

        # For each Linear layer in each block, convert "non-reusable" sections into NeededPieces (weight/bias) based on dimension size.
        for idx in new_required_indices:
            cpu_block = self.get_cpu_block_for_index(cpu_full_model, idx)
            cached = self.cache.get(idx, None)
            cached_tp_range = cached[1] if cached is not None else None

            plan = DataReconstructionPlan(
                block_index=idx,
                target_tp_range=target_tp,
                skeleton_block=None,
                pieces=[],
            )

            for name, cpu_submodule in dict(cpu_block.named_modules()).items():
                if not isinstance(cpu_submodule, nn.Linear):
                    continue
                tp_split_type = self.choose_tp_split_type(name)
                if tp_split_type is None:
                    continue

                if tp_split_type == "row":
                    full_size = cpu_submodule.weight.size(0)
                    split_dim = 0
                else:  # "column"
                    full_size = cpu_submodule.weight.size(1)
                    split_dim = 1

                new_start, new_end = self.range_to_indices(target_tp, full_size)
                if new_end <= new_start:
                    continue

                if cached_tp_range is not None:
                    old_start, old_end = self.range_to_indices(
                        cached_tp_range, full_size
                    )
                    overlap_start = max(new_start, old_start)
                    overlap_end = min(new_end, old_end)
                else:
                    overlap_start, overlap_end = (
                        new_start,
                        new_start,
                    )  # Indicates no overlap

                # Calculate gaps: prefix part + suffix part
                gaps: List[Tuple[int, int]] = []
                if overlap_start > new_start:
                    gaps.append((new_start, overlap_start))
                if overlap_end < new_end:
                    gaps.append((overlap_end, new_end))

                # Convert gaps to NeededPiece (weight)
                for gap_start, gap_end in gaps:
                    if gap_end <= gap_start:
                        continue
                    piece_uid += 1
                    plan.pieces.append(
                        NeededPiece(
                            uid=piece_uid,
                            block_index=idx,
                            layer_path=name,
                            tensor_type="weight",
                            split_dim=split_dim,
                            full_size=full_size,
                            abs_start_idx=gap_start,
                            abs_end_idx=gap_end,
                            dest_offset=gap_start - new_start,
                            component=component_name,
                        )
                    )

                    # For row-split, also generate corresponding pieces for bias (aligned with out_features)
                    if (
                        slice_bias
                        and tp_split_type == "row"
                        and cpu_submodule.bias is not None
                    ):
                        bias_full = cpu_submodule.bias.size(0)
                        # Generate interval for bias dimension similarly using target_tp
                        bias_new_start, bias_new_end = self.range_to_indices(
                            target_tp, bias_full
                        )
                        # The absolute interval of the corresponding gap on the bias
                        bias_gap_start = bias_new_start + (gap_start - new_start)
                        bias_gap_end = bias_gap_start + (gap_end - gap_start)
                        if bias_gap_end > bias_gap_start:
                            piece_uid += 1
                            plan.pieces.append(
                                NeededPiece(
                                    uid=piece_uid,
                                    block_index=idx,
                                    layer_path=name,
                                    tensor_type="bias",
                                    split_dim=split_dim,  # Descriptive only; bias is logically 1D
                                    full_size=bias_full,
                                    abs_start_idx=bias_gap_start,
                                    abs_end_idx=bias_gap_end,
                                    dest_offset=bias_gap_start - bias_new_start,
                                    component=component_name,  # New: Write component name
                                )
                            )

            pre_plans[idx] = plan
            all_pieces.extend(plan.pieces)

        # Archive (for direct reuse in subsequent P2P execution)
        self._last_p2p_preplans = pre_plans
        self._last_p2p_all_pieces = all_pieces
        self._last_reusable_tp_by_block = reusable_tp_by_block

        if use_p2p and all_pieces:
            try:
                await self._source_needed_pieces_from_peers(all_pieces, component_name)
                sourced_cnt = sum(1 for p in all_pieces if p.status == "sourced")
                logger.debug(
                    f"P2P sourcing complete: {sourced_cnt}/{len(all_pieces)} pieces can be fetched from other GPUs."
                )
            except Exception as e:
                logger.warning(
                    f"Exception during P2P sourcing phase, continuing with local/CPU fallback: {e}"
                )

        # Build skeletons for "newly cached blocks" -> P2P -> CPU backfill
        blocks_to_fetch: List[int] = [
            idx for idx in new_required_indices if idx not in self.cache
        ]
        plans_for_recv: Dict[int, DataReconstructionPlan] = {}
        pieces_for_recv: List[NeededPiece] = []

        logger.debug("Starting to build skeletons")
        for idx in blocks_to_fetch:
            sourced = [
                p
                for p in pre_plans.get(
                    idx, DataReconstructionPlan(idx, target_tp, None, [])
                ).pieces
                if p.status == "sourced"
            ]
            # First, build the skeleton (even if this block has no sourced pieces, it must be placed in the cache for later CPU backfill)
            sk = self._create_skeleton_block(
                idx, cpu_full_model, target_tp, slice_bias=slice_bias
            )
            pre_plans[idx].skeleton_block = sk
            # Put it in the cache, mark it with the target TP, so adjust_linears_in_layers will skip this block later
            self._add_or_update(idx, sk, target_tp)

            if sourced:
                plans_for_recv[idx] = pre_plans[idx]
                pieces_for_recv.extend(sourced)

        logger.debug(
            f"Starting P2P reconstruction, blocks to reconstruct: {blocks_to_fetch}, pieces to reconstruct: {pieces_for_recv}"
        )
        if use_p2p and pieces_for_recv:
            try:
                logger.debug("Starting nixl P2P reconstruction")
                await self.worker.nixl_manager.execute_reconstruction_plan(
                    pieces_for_recv, plans_for_recv, component_name
                )
                logger.debug(
                    f"[NIXL][RECV] P2P reconstruction complete: blocks={list(plans_for_recv.keys())}, pieces={len(pieces_for_recv)}"
                )
            except Exception as e:
                logger.warning(
                    f"Exception during P2P reconstruction phase (new blocks), continuing: {e}"
                )

        # Perform CPU backfill for all new blocks (only filling uncovered intervals; column-split bias was copied to GPU in its entirety during the skeleton stage)
        for idx in blocks_to_fetch:
            try:
                self._fill_plan_cpu_gaps(
                    pre_plans[idx], cpu_full_model, slice_bias=slice_bias
                )
            except Exception as e:
                logger.warning(f"Block {idx} CPU backfill failed: {e}")

        # Then continue the original process: fetch blocks uniformly, place them back on the model, and re-slice for TP if necessary
        new_active_blocks_list, loaded_from_cpu_flags, actual_cached_tp_ranges = (
            [],
            [],
            [],
        )
        for idx in new_required_indices:
            block_module, loaded, cached_tp = self.get_block(idx, cpu_full_model)
            new_active_blocks_list.append(block_module)
            loaded_from_cpu_flags.append(loaded)
            actual_cached_tp_ranges.append(cached_tp)

        self.place_blocks_on_model(
            model, new_active_blocks_list, new_required_indices, cpu_full_model
        )
        self.active_block_indices = new_required_indices

        if new_tp_range is not None:
            logger.info(
                f"Adjusting Tensor Parallelism from {old_tp_range} to target {new_tp_range}"
            )
            layers_actually_adjusted = self.adjust_linears_in_layers(
                layers=new_active_blocks_list,
                cpu_full_model=cpu_full_model,
                new_pp_range=new_pp_range,
                new_tp_range=new_tp_range,
                loaded_from_cpu_flags=loaded_from_cpu_flags,
                actual_cached_tp_ranges=actual_cached_tp_ranges,
                slice_bias=slice_bias,
                required_indices=new_required_indices,
                pre_plans=pre_plans,
                component_name=component_name,
            )
            if layers_actually_adjusted > 0:
                for idx in self.active_block_indices:
                    if idx in self.cache:
                        module, _ = self.cache[idx]
                        self.cache[idx] = (module, new_tp_range)
                    else:
                        logger.warning(
                            f"Block {idx} was active but not found in cache after TP adjustment."
                        )
            else:
                logger.info(
                    "adjust_linears_in_layers reported no blocks needed adjustment."
                )

        # Execute P2P for sourced pieces of "already cached blocks", then fill gaps with CPU
        index_to_block = {
            idx: blk for idx, blk in zip(new_required_indices, new_active_blocks_list)
        }
        existing_indices = [
            idx for idx in new_required_indices if idx not in blocks_to_fetch
        ]
        pieces_for_recv_existing, plans_for_recv_existing = [], {}
        if use_p2p and pre_plans:
            for idx in existing_indices:
                plan = pre_plans.get(idx)
                if not plan:
                    continue
                # Only collect sourced pieces
                sourced = [p for p in plan.pieces if p.status == "sourced"]
                if not sourced:
                    continue
                # Ensure skeleton_block points to the latest GPU module (shape reconstruction is complete)
                plan.skeleton_block = index_to_block[idx]
                plans_for_recv_existing[idx] = plan
                pieces_for_recv_existing.extend(sourced)

            if pieces_for_recv_existing:
                try:
                    await self.worker.nixl_manager.execute_reconstruction_plan(
                        pieces_for_recv_existing,
                        plans_for_recv_existing,
                        component_name,
                    )
                    logger.debug(
                        f"[NIXL][RECV] P2P reconstruction complete (for cached blocks): blocks={list(plans_for_recv_existing.keys())}, pieces={len(pieces_for_recv_existing)}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Exception during P2P reconstruction phase (for cached blocks), continuing: {e}"
                    )

                # CPU backfill for uncovered intervals (for cached blocks)
                for idx in plans_for_recv_existing.keys():
                    try:
                        self._fill_plan_cpu_gaps(
                            pre_plans[idx], cpu_full_model, slice_bias=slice_bias
                        )
                    except Exception as e:
                        logger.warning(f"Block {idx} CPU backfill failed: {e}")

        logger.info(
            f"Running eviction check after pipeline adjustment. Cache size: {len(self.cache)}, Limit: {self.global_blocks_limit}, Active blocks: {len(self.active_block_indices)}"
        )
        self._evict()

        logger.debug(
            f"Adjusted pipeline. Cache Status: Total GPU={self.get_total_gpu_blocks()}, Active={self.get_active_blocks_count()}, Cached Only={self.get_cached_only_blocks_count()}"
        )
        logger.debug(f"Active indices: {self.active_block_indices}")
        logger.debug(
            f"Cache keys (LRU order): {[f'{k}:{v[1]}' for k, v in self.cache.items()]}"
        )
        return model

    def adjust_linears_in_layers(
        self,
        layers: List[nn.Module],
        cpu_full_model: nn.Module,
        new_pp_range: tuple,
        new_tp_range: tuple,
        loaded_from_cpu_flags: List[bool],
        actual_cached_tp_ranges: List[Optional[Tuple[float, float]]],
        slice_bias: bool = True,
        required_indices: Optional[List[int]] = None,
        pre_plans: Optional[Dict[int, DataReconstructionPlan]] = None,  # New
        component_name: Optional[str] = None,  # New (for logging)
    ) -> int:
        total_blocks = self.get_total_blocks(cpu_full_model)
        new_pp_start, _ = self.range_to_indices(new_pp_range, total_blocks)
        if len(layers) != len(loaded_from_cpu_flags) or len(layers) != len(
            actual_cached_tp_ranges
        ):
            msg = (
                f"Mismatch counts: layers({len(layers)}), "
                f"loaded_flags({len(loaded_from_cpu_flags)}), "
                f"cached_tp_ranges({len(actual_cached_tp_ranges)})"
            )
            logger.error(msg)
            raise ValueError(
                "Layer count and associated flags/ranges count mismatch during TP adjustment."
            )

        num_layers_adjusted = 0
        for i, gpu_block in enumerate(layers):
            absolute_block_index = (
                required_indices[i] if required_indices else (new_pp_start + i)
            )
            was_loaded_from_cpu = loaded_from_cpu_flags[i]
            actual_cached_tp = actual_cached_tp_ranges[i]

            if was_loaded_from_cpu:
                tp_range_before_adjustment = (0.0, 1.0)
            elif actual_cached_tp is not None:
                tp_range_before_adjustment = actual_cached_tp
            else:
                logger.error(
                    f"Block {absolute_block_index} from cache but no TP range found. Assuming (0.0, 1.0)."
                )
                tp_range_before_adjustment = (0.0, 1.0)

            if tp_range_before_adjustment == new_tp_range:
                continue

            logger.info(
                f"Block {absolute_block_index}: Adjusting TP layers from {tp_range_before_adjustment} to {new_tp_range}."
            )
            num_layers_adjusted += 1

            # Bind skeleton to the plan of existing blocks (for subsequent P2P writes)
            if pre_plans and absolute_block_index in pre_plans:
                pre_plans[absolute_block_index].skeleton_block = gpu_block

            cpu_block = self.get_cpu_block_for_index(
                cpu_full_model, absolute_block_index
            )
            cpu_named_modules = dict(cpu_block.named_modules())
            for name, gpu_submodule in dict(gpu_block.named_modules()).items():
                if (
                    isinstance(gpu_submodule, nn.Linear)
                    and name in cpu_named_modules
                    and isinstance(cpu_named_modules[name], nn.Linear)
                ):
                    tp_split_type = self.choose_tp_split_type(name)
                    if tp_split_type is None:
                        continue

                    # Collect the absolute intervals of weight/bias for this layer that are "covered by P2P" (on the target TP)
                    skip_w, skip_b = None, None
                    if pre_plans and absolute_block_index in pre_plans:
                        plan = pre_plans[absolute_block_index]
                        sw = [
                            (p.abs_start_idx, p.abs_end_idx)
                            for p in plan.pieces
                            if p.layer_path == name
                            and p.tensor_type == "weight"
                            and p.status == "sourced"
                        ]
                        sb = [
                            (p.abs_start_idx, p.abs_end_idx)
                            for p in plan.pieces
                            if p.layer_path == name
                            and p.tensor_type == "bias"
                            and p.status == "sourced"
                        ]
                        skip_w = sw if sw else None
                        skip_b = sb if sb else None

                    try:
                        self.adjust_linear_tensor_parallel(
                            linear_module=gpu_submodule,
                            cpu_module=cpu_named_modules[name],
                            old_tp_range=tp_range_before_adjustment,
                            new_tp_range=new_tp_range,
                            tp_split_type=tp_split_type,
                            slice_bias=slice_bias,
                            skip_weight_intervals=skip_w,
                            skip_bias_intervals=skip_b,
                        )
                    except Exception as e:
                        logger.error(
                            f"Error adjusting {name} in block {absolute_block_index}: {e}"
                        )
                        raise RuntimeError(
                            f"Error adjusting {name} in block {absolute_block_index} "
                            f"from TP {tp_range_before_adjustment} to {new_tp_range}"
                        ) from e
                elif (
                    isinstance(gpu_submodule, nn.Linear)
                    and name not in cpu_named_modules
                ):
                    logger.warning(
                        f"Layer {name} found in GPU block {absolute_block_index} but not in corresponding CPU block."
                    )
                elif isinstance(gpu_submodule, nn.Linear) and not isinstance(
                    cpu_named_modules.get(name), nn.Linear
                ):
                    logger.warning(
                        f"Layer {name} in GPU block {absolute_block_index} is Linear, but CPU counterpart is {type(cpu_named_modules.get(name))}."
                    )
        return num_layers_adjusted

    def adjust_linear_tensor_parallel(
        self,
        linear_module: nn.Linear,
        cpu_module: nn.Linear,
        old_tp_range: tuple,
        new_tp_range: tuple,
        tp_split_type: str = "row",
        slice_bias: bool = True,
        skip_weight_intervals: Optional[List[Tuple[int, int]]] = None,
        skip_bias_intervals: Optional[List[Tuple[int, int]]] = None,
    ) -> nn.Linear:
        """
        Adjusts the weights and bias of a linear layer for tensor parallelism,
        optimizing CPU->GPU copies using an intermediate pinned CPU buffer
        when necessary.

        If skip_weight_intervals/skip_bias_intervals provided, the function will
        skip CPU copies for those absolute intervals in the new TP slice, leaving
        them for P2P to fill later.
        """
        if old_tp_range == new_tp_range:
            return linear_module

        def _merge(iv):
            if not iv:
                return []
            iv = sorted(iv)
            res = []
            for a, b in iv:
                if not res or a > res[-1][1]:
                    res.append([a, b])
                else:
                    res[-1][1] = max(res[-1][1], b)
            return [(a, b) for a, b in res]

        def _clip_to(base, ivs):
            a0, b0 = base
            out = []
            for a, b in ivs:
                a = max(a0, min(a, b0))
                b = max(a0, min(b, b0))
                if b > a:
                    out.append((a, b))
            return _merge(out)

        def _complement(base, covered):
            covered = _merge(covered)
            a0, b0 = base
            cur = a0
            res = []
            for a, b in covered:
                if a > cur:
                    res.append((cur, a))
                cur = max(cur, b)
            if cur < b0:
                res.append((cur, b0))
            return res

        layer_name_for_log = f"Layer (Type: {tp_split_type}, Bias: {slice_bias}, OldTP: {old_tp_range}, NewTP: {new_tp_range})"

        # Calculations
        if tp_split_type == "row":
            tensor_param_total_size = cpu_module.weight.size(0)
            tensor_split_dimension = 0
            bias_total_size = (
                cpu_module.bias.size(0) if cpu_module.bias is not None else 0
            )
        else:  # "column"
            tensor_param_total_size = cpu_module.weight.size(1)
            tensor_split_dimension = 1
            bias_total_size = 0

        old_tp_start, old_tp_end = self.range_to_indices(
            old_tp_range, tensor_param_total_size
        )
        new_tp_start, new_tp_end = self.range_to_indices(
            new_tp_range, tensor_param_total_size
        )

        expected_old_len = old_tp_end - old_tp_start
        if cpu_module.weight.dim() < 2:
            raise ValueError(
                f"CPU weight dimension is {cpu_module.weight.dim()}, expected at least 2."
            )
        if tensor_split_dimension == 0:
            expected_old_shape = (expected_old_len, cpu_module.weight.size(1))
        else:
            expected_old_shape = (cpu_module.weight.size(0), expected_old_len)

        weight = linear_module.weight.data
        cpu_weight = cpu_module.weight.data.cpu()
        cpu_bias = cpu_module.bias.data.cpu() if cpu_module.bias is not None else None
        gpu_device = weight.device

        use_gpu_optimization = True
        if weight.shape != expected_old_shape:
            logger.warning(
                f"{layer_name_for_log}: Cached weight shape {weight.shape} mismatch for its actual TP range {old_tp_range} (expected {expected_old_shape}). Discarding GPU cache optimization for this layer."
            )
            use_gpu_optimization = False

        needed_length = new_tp_end - new_tp_start
        if needed_length <= 0:
            result_weight = torch.empty(
                (0,) + weight.shape[1:], dtype=weight.dtype, device=gpu_device
            )
            logger.info(
                f"{layer_name_for_log} New weight slice is empty. Setting empty tensor."
            )
        else:
            if tensor_split_dimension == 0:
                result_shape = (needed_length, cpu_weight.size(1))
            else:
                result_shape = (cpu_weight.size(0), needed_length)

            result_weight = torch.empty(
                result_shape, dtype=weight.dtype, device=gpu_device
            )

            # Normalize skip intervals to the new slice range
            skip_w = _clip_to((new_tp_start, new_tp_end), skip_weight_intervals or [])

            if use_gpu_optimization:
                overlap_start = max(old_tp_start, new_tp_start)
                overlap_end = min(old_tp_end, new_tp_end)
                overlap_len = overlap_end - overlap_start

                if overlap_len > 0:
                    old_rel_start = overlap_start - old_tp_start
                    new_rel_start = overlap_start - new_tp_start
                    if tensor_split_dimension == 0:
                        result_weight[
                            new_rel_start : new_rel_start + overlap_len, :
                        ] = weight[old_rel_start : old_rel_start + overlap_len, :]
                    else:
                        result_weight[
                            :, new_rel_start : new_rel_start + overlap_len
                        ] = weight[:, old_rel_start : old_rel_start + overlap_len]

                    # Intervals needing CPU fill = [new_start, overlap_start) U [overlap_end, new_end) minus skip_w
                    cpu_need = []
                    if new_tp_start < overlap_start:
                        cpu_need.append((new_tp_start, overlap_start))
                    if overlap_end < new_tp_end:
                        cpu_need.append((overlap_end, new_tp_end))
                    cpu_need = (
                        _complement((new_tp_start, new_tp_end), skip_w)
                        if skip_w
                        else cpu_need
                    )
                    if cpu_need:
                        for a, b in cpu_need:
                            length = b - a
                            if length <= 0:
                                continue
                            if tensor_split_dimension == 0:
                                cpu_slice = torch.narrow(
                                    cpu_weight, tensor_split_dimension, a, length
                                )
                                pinned_buffer = torch.empty_like(
                                    cpu_slice, device="cpu", pin_memory=True
                                )
                                pinned_buffer.copy_(cpu_slice)
                                result_weight[
                                    a - new_tp_start : a - new_tp_start + length, :
                                ] = pinned_buffer
                            else:
                                cpu_slice = torch.narrow(
                                    cpu_weight, tensor_split_dimension, a, length
                                )
                                pinned_buffer = torch.empty_like(
                                    cpu_slice, device="cpu", pin_memory=True
                                )
                                pinned_buffer.copy_(cpu_slice)
                                result_weight[
                                    :, a - new_tp_start : a - new_tp_start + length
                                ] = pinned_buffer
                    else:
                        logger.debug(
                            f"{layer_name_for_log} All non-overlap weight segments will be filled by P2P."
                        )
                else:
                    # No overlap at all: only fill sub-intervals not covered by P2P
                    cpu_need = (
                        _complement((new_tp_start, new_tp_end), skip_w)
                        if skip_w
                        else [(new_tp_start, new_tp_end)]
                    )
                    for a, b in cpu_need:
                        length = b - a
                        if length <= 0:
                            continue
                        if tensor_split_dimension == 0:
                            cpu_slice = torch.narrow(
                                cpu_weight, tensor_split_dimension, a, length
                            )
                            pinned_buffer = torch.empty_like(
                                cpu_slice, device="cpu", pin_memory=True
                            )
                            pinned_buffer.copy_(cpu_slice)
                            result_weight[
                                a - new_tp_start : a - new_tp_start + length, :
                            ] = pinned_buffer
                        else:
                            cpu_slice = torch.narrow(
                                cpu_weight, tensor_split_dimension, a, length
                            )
                            pinned_buffer = torch.empty_like(
                                cpu_slice, device="cpu", pin_memory=True
                            )
                            pinned_buffer.copy_(cpu_slice)
                            result_weight[
                                :, a - new_tp_start : a - new_tp_start + length
                            ] = pinned_buffer
            else:
                # Fallback path: also perform incremental filling according to skip_w
                cpu_need = (
                    _complement((new_tp_start, new_tp_end), skip_w)
                    if skip_w
                    else [(new_tp_start, new_tp_end)]
                )
                for a, b in cpu_need:
                    length = b - a
                    if length <= 0:
                        continue
                    cpu_slice = torch.narrow(
                        cpu_weight, tensor_split_dimension, a, length
                    )
                    pinned_buffer = torch.empty_like(
                        cpu_slice, device="cpu", pin_memory=True
                    )
                    pinned_buffer.copy_(cpu_slice)
                    if tensor_split_dimension == 0:
                        result_weight[
                            a - new_tp_start : a - new_tp_start + length, :
                        ] = pinned_buffer
                    else:
                        result_weight[
                            :, a - new_tp_start : a - new_tp_start + length
                        ] = pinned_buffer

            result_weight_contiguous = result_weight.contiguous()
        linear_module.weight = nn.Parameter(result_weight_contiguous)

        # Bias
        result_bias = None
        bias = linear_module.bias.data if linear_module.bias is not None else None

        if slice_bias and cpu_bias is not None and tp_split_type == "row":
            old_bias_tp_start, old_bias_tp_end = self.range_to_indices(
                old_tp_range, bias_total_size
            )
            new_bias_tp_start, new_bias_tp_end = self.range_to_indices(
                new_tp_range, bias_total_size
            )
            needed_bias_length = new_bias_tp_end - new_bias_tp_start

            if needed_bias_length > 0:
                can_optimize_bias = use_gpu_optimization and (bias is not None)
                if bias is not None:
                    expected_old_bias_len = old_bias_tp_end - old_bias_tp_start
                    if bias.shape != (expected_old_bias_len,):
                        logger.warning(
                            f"{layer_name_for_log} Cached bias shape {bias.shape} mismatch for its actual TP range {old_tp_range} (expected ({expected_old_bias_len},)). Discarding GPU cache optimization for bias."
                        )
                        can_optimize_bias = False
                else:
                    can_optimize_bias = False

                # Normalize bias skip
                skip_b = _clip_to(
                    (new_bias_tp_start, new_bias_tp_end), skip_bias_intervals or []
                )

                result_bias = torch.empty(
                    (needed_bias_length,), dtype=cpu_bias.dtype, device=gpu_device
                )

                if can_optimize_bias:
                    bias_overlap_start = max(old_bias_tp_start, new_bias_tp_start)
                    bias_overlap_end = min(old_bias_tp_end, new_bias_tp_end)
                    bias_overlap_len = bias_overlap_end - bias_overlap_start

                    if bias_overlap_len > 0:
                        old_rel_start = bias_overlap_start - old_bias_tp_start
                        new_rel_start = bias_overlap_start - new_bias_tp_start
                        result_bias[
                            new_rel_start : new_rel_start + bias_overlap_len
                        ] = bias[old_rel_start : old_rel_start + bias_overlap_len]

                    cpu_need_b = (
                        _complement((new_bias_tp_start, new_bias_tp_end), skip_b)
                        if skip_b
                        else []
                    )
                    if not skip_b:
                        # No P2P: fill based on overlap
                        if bias_overlap_len > 0:
                            # If there is overlap, fill the gaps around it
                            if new_bias_tp_start < bias_overlap_start:
                                cpu_need_b.append((new_bias_tp_start, bias_overlap_start))
                            if bias_overlap_end < new_bias_tp_end:
                                cpu_need_b.append((bias_overlap_end, new_bias_tp_end))
                        else:
                            # If there is NO overlap, the entire new range needs to be filled from CPU
                            cpu_need_b.append((new_bias_tp_start, new_bias_tp_end))

                    for a, b in cpu_need_b:
                        length = b - a
                        if length <= 0:
                            continue
                        cpu_slice = cpu_bias[a:b]
                        pinned_buffer = torch.empty_like(
                            cpu_slice, device="cpu", pin_memory=True
                        )
                        pinned_buffer.copy_(cpu_slice)
                        result_bias[
                            a - new_bias_tp_start : a - new_bias_tp_start + length
                        ] = pinned_buffer
                else:
                    # Fallback: also perform incremental filling according to skip_b
                    cpu_need_b = (
                        _complement((new_bias_tp_start, new_bias_tp_end), skip_b)
                        if skip_b
                        else [(new_bias_tp_start, new_bias_tp_end)]
                    )
                    for a, b in cpu_need_b:
                        length = b - a
                        if length <= 0:
                            continue
                        cpu_slice = cpu_bias[a:b]
                        pinned_buffer = torch.empty_like(
                            cpu_slice, device="cpu", pin_memory=True
                        )
                        pinned_buffer.copy_(cpu_slice)
                        result_bias[
                            a - new_bias_tp_start : a - new_bias_tp_start + length
                        ] = pinned_buffer

            elif needed_bias_length == 0 and not (
                slice_bias and cpu_bias is not None and tp_split_type == "row"
            ):
                result_bias = torch.empty(
                    (0,),
                    dtype=cpu_bias.dtype if cpu_bias is not None else weight.dtype,
                    device=gpu_device,
                )
                logger.info(
                    f"{layer_name_for_log} New bias slice is empty. Setting empty tensor."
                )

        elif cpu_bias is not None and tp_split_type == "column":
            if bias is None or bias.shape != cpu_bias.shape:
                result_bias = torch.empty_like(cpu_bias, device=gpu_device)
                pinned_buffer = torch.empty_like(
                    cpu_bias, device="cpu", pin_memory=True
                )
                pinned_buffer.copy_(cpu_bias)
                result_bias.copy_(pinned_buffer, non_blocking=True)
            else:
                result_bias = bias
                logger.info(
                    f"{layer_name_for_log} Column split, existing bias found and kept."
                )

        if result_bias is not None:
            if result_bias.numel() > 0:
                result_bias_contiguous = result_bias.contiguous()
            else:
                result_bias_contiguous = result_bias
            linear_module.bias = nn.Parameter(result_bias_contiguous)
        else:
            linear_module.bias = None
        return linear_module

    async def _source_needed_pieces_from_peers(
        self, all_needed_pieces: List[NeededPiece], component_name: Optional[str] = None
    ):
        """Sourcing phase: broadcast needs and select a data source for each piece."""
        if not all_needed_pieces:
            logger.info("No sourcing needed, there are no missing data pieces.")
            return

        my_rank = self.worker.rank
        peer_handles = {
            r: h for r, h in self.worker.all_worker_handles.items() if r != my_rank
        }

        # Convert NeededPiece objects to a list of serializable dictionaries for RPC
        serializable_pieces = []
        for piece in all_needed_pieces:
            d = piece.__dict__.copy()
            if component_name:
                d["component"] = component_name
            serializable_pieces.append(d)

        logger.info(
            f"Broadcasting sourcing request for {len(serializable_pieces)} data pieces to {len(peer_handles)} peers."
        )
        # Only pass one argument here to maintain signature consistency with ray_utils/worker
        tasks = [
            h.rpc_find_sources.remote(serializable_pieces)
            for h in peer_handles.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        piece_map = {p.uid: p for p in all_needed_pieces}

        for i, res in enumerate(results):
            peer_rank = list(peer_handles.keys())[i]
            if isinstance(res, Exception):
                logger.warning(f"Error while sourcing from Rank {peer_rank}: {res}")
                continue

            for piece_uid in res:
                if piece_map[piece_uid].source_rank == -1:
                    piece_map[piece_uid].source_rank = peer_rank
                    piece_map[piece_uid].status = "sourced"

        sourced_count = sum(1 for p in all_needed_pieces if p.status == "sourced")
        logger.info(
            f"Sourcing complete. Found P2P data sources for {sourced_count} / {len(all_needed_pieces)} pieces."
        )

    def _create_skeleton_block(
        self,
        block_index: int,
        cpu_full_model: nn.Module,
        target_tp_range: Tuple[float, float],
        slice_bias: bool = True,
    ) -> nn.Module:
        cpu_block = self.get_cpu_block_for_index(cpu_full_model, block_index)
        sk = copy.deepcopy(cpu_block)

        cpu_named = dict(cpu_block.named_modules())
        for name, m in sk.named_modules():
            if not isinstance(m, nn.Linear) or name not in cpu_named:
                continue
            cpu_lin = cpu_named[name]
            tp_type = self.choose_tp_split_type(name)
            if tp_type is None:
                continue

            if tp_type == "row":
                full = cpu_lin.weight.size(0)
                new_s, new_e = self.range_to_indices(target_tp_range, full)
                need = max(0, new_e - new_s)
                in_features = cpu_lin.weight.size(1)
                w = torch.empty(
                    (need, in_features), dtype=cpu_lin.weight.dtype, device="cuda"
                )
                m.weight = nn.Parameter(w)
                if slice_bias and cpu_lin.bias is not None:
                    b = torch.empty((need,), dtype=cpu_lin.bias.dtype, device="cuda")
                    m.bias = nn.Parameter(b)
                else:
                    m.bias = (
                        nn.Parameter(cpu_lin.bias.detach().to("cuda"))
                        if cpu_lin.bias is not None
                        else None
                    )
            else:  # column
                full = cpu_lin.weight.size(1)
                new_s, new_e = self.range_to_indices(target_tp_range, full)
                need = max(0, new_e - new_s)
                out_features = cpu_lin.weight.size(0)
                w = torch.empty(
                    (out_features, need), dtype=cpu_lin.weight.dtype, device="cuda"
                )
                m.weight = nn.Parameter(w)
                # column split doesn't slice bias -> copy the entire bias to GPU directly (no P2P for this part)
                m.bias = (
                    nn.Parameter(cpu_lin.bias.detach().to("cuda"))
                    if cpu_lin.bias is not None
                    else None
                )

        # Migrate non-parameter buffers
        sk = sk.to("cuda", non_blocking=True)
        return sk

    def _fill_plan_cpu_gaps(
        self,
        plan: DataReconstructionPlan,
        cpu_full_model: nn.Module,
        slice_bias: bool = True,
    ) -> None:
        # Calculate the covered intervals for each layer based on completed pieces (status == "done") in the plan, and accurately backfill the rest using the CPU
        def merge(iv):
            iv = sorted(iv)
            res = []
            for a, b in iv:
                if not res or a > res[-1][1]:
                    res.append([a, b])
                else:
                    res[-1][1] = max(res[-1][1], b)
            return [(a, b) for a, b in res]

        def complement(base, cov):
            cov = merge(cov)
            res = []
            cur = base[0]
            for a, b in cov:
                if a > cur:
                    res.append((cur, a))
                cur = max(cur, b)
            if cur < base[1]:
                res.append((cur, base[1]))
            return res

        sk = plan.skeleton_block
        cpu_block = self.get_cpu_block_for_index(cpu_full_model, plan.block_index)
        cpu_named = dict(cpu_block.named_modules())

        # Aggregate completed coverage by layer
        covered_w = {}
        covered_b = {}
        for p in plan.pieces:
            if p.status != "done":
                continue
            if p.tensor_type == "weight":
                covered_w.setdefault(p.layer_path, []).append(
                    (p.abs_start_idx, p.abs_end_idx)
                )
            else:
                covered_b.setdefault(p.layer_path, []).append(
                    (p.abs_start_idx, p.abs_end_idx)
                )

        for name, m in sk.named_modules():
            if not isinstance(m, nn.Linear) or name not in cpu_named:
                continue
            cpu_lin = cpu_named[name]
            tp_type = self.choose_tp_split_type(name)
            if tp_type is None:
                continue

            if tp_type == "row":
                # weight gaps
                full = cpu_lin.weight.size(0)
                new_s, new_e = self.range_to_indices(plan.target_tp_range, full)
                need = max(0, new_e - new_s)
                if need > 0:
                    gaps = complement((new_s, new_e), covered_w.get(name, []))
                    for a, b in gaps:
                        ofs = a - new_s
                        m.weight.data[ofs : ofs + (b - a), :].copy_(
                            cpu_lin.weight.data[a:b, :], non_blocking=True
                        )

                # bias gaps
                if slice_bias and cpu_lin.bias is not None:
                    bfull = cpu_lin.bias.size(0)
                    bs, be = self.range_to_indices(plan.target_tp_range, bfull)
                    if be > bs:
                        gaps_b = complement((bs, be), covered_b.get(name, []))
                        for a, b in gaps_b:
                            ofs = a - bs
                            m.bias.data[ofs : ofs + (b - a)].copy_(
                                cpu_lin.bias.data[a:b], non_blocking=True
                            )

            else:  # column
                # weight gaps
                full = cpu_lin.weight.size(1)
                new_s, new_e = self.range_to_indices(plan.target_tp_range, full)
                need = max(0, new_e - new_s)
                if need > 0:
                    gaps = complement((new_s, new_e), covered_w.get(name, []))
                    for a, b in gaps:
                        ofs = a - new_s
                        m.weight.data[:, ofs : ofs + (b - a)].copy_(
                            cpu_lin.weight.data[:, a:b], non_blocking=True
                        )
