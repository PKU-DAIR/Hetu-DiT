import asyncio
import pickle
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import torch
import torch.nn as nn
from ray.actor import ActorHandle

from hetu_dit.logger import init_logger
from nixl._api import nixl_agent, nixl_agent_config

if TYPE_CHECKING:
    from hetu_dit.model_executor.cache.base_cache import (
        DataReconstructionPlan,
        NeededPiece,
    )

logger = init_logger(__name__)


class NixlP2PManager:
    """
    Communication manager for zero-copy P2P tensor transfer using NIXL.
    """

    def __init__(self, rank: int, all_worker_handles: Dict[int, ActorHandle]):
        self.rank = rank
        self.agent_name = f"agent_{self.rank}"
        self.agent = nixl_agent(self.agent_name, nixl_agent_config(backends=["UCX"]))
        self.all_worker_handles = all_worker_handles
        self.registered_blocks: Dict[int, List[torch.Tensor]] = {}
        self._xfer_lock = asyncio.Lock()

        logger.info(f"Rank {self.rank}: NixlP2PManager initialized successfully.")

    async def _execute_transfer_for_source(
        self,
        source_rank: int,
        pieces: List["NeededPiece"],
        dest_descriptors: List[tuple],
        tensors_to_register: List[torch.Tensor],
    ):
        """
        Executes the complete transfer flow for a single source: registration, RPC call, and deregistration.
        This is a standalone, concurrently executable task.
        """
        reg_desc_src = None
        try:
            # 1. Register memory for this source
            if tensors_to_register:
                reg_desc_src = self.agent.register_memory(tensors_to_register)
            else:
                reg_desc_src = self.agent.get_reg_descs([])

            logger.debug(
                f"[NIXL][RECV][REG] rank={self.rank} src={source_rank} uniq_tensors={len(tensors_to_register)}"
            )

            # 2. Generate partial metadata and transfer descriptors
            partial_md_src = self.agent.get_partial_agent_metadata(
                reg_desc_src, inc_conn_info=True
            )
            pm_head = (
                bytes(partial_md_src)[:64]
                if hasattr(partial_md_src, "__iter__")
                else b""
            )
            logger.debug(
                f"[NIXL][RECV][PM] rank={self.rank} src={source_rank} partial_md_len={len(partial_md_src)} partial_md_head_hash={hash(pm_head)}"
            )

            remote_xfer_desc = self.agent.get_xfer_descs(
                dest_descriptors, mem_type="cuda", is_sorted=True
            )
            remote_xfer_desc_bytes = pickle.dumps(remote_xfer_desc)

            tuples = len(dest_descriptors)
            bytes_sum_dst = sum(int(d[1]) for d in dest_descriptors)
            logger.debug(
                f"[NIXL][RECV][XFER] rank={self.rank} src={source_rank} tuples={tuples} bytes_sum={bytes_sum_dst} first={dest_descriptors[:1]} last={dest_descriptors[-1:]}"
            )

            # 3. Initiate the RPC call
            source_worker_handle = self.all_worker_handles[source_rank]
            logger.debug(
                f"[NIXL][RECV] rank={self.rank} -> RPC to source={source_rank}, pieces={len(pieces)}"
            )

            await source_worker_handle.rpc_nixl_send_data.remote(
                pieces, remote_xfer_desc_bytes, partial_md_src, self.rank
            )
            logger.debug(
                f"[NIXL][RECV] rank={self.rank} transfer from source={source_rank} finished."
            )

        finally:
            # 4. Deregister memory after the transfer is complete (regardless of success or failure)
            if reg_desc_src and tensors_to_register:
                try:
                    self.agent.deregister_memory(reg_desc_src)
                    logger.debug(
                        f"[NIXL][RECV][REG] rank={self.rank} src={source_rank} deregistered"
                    )
                except Exception as e:
                    logger.debug(
                        f"[NIXL][RECV][REG] rank={self.rank} src={source_rank} deregister error: {e}"
                    )

    def init_peers(self, all_metadata: List[Tuple[int, bytes]]):
        """Establishes P2P connections for NIXL using metadata from all workers."""
        logger.info(f"Rank {self.rank}: Initializing NIXL inter-node connections...")
        for r, meta_bytes in all_metadata:
            if r != self.rank:
                remote_name = self.agent.add_remote_agent(meta_bytes)
                logger.debug(
                    f"Rank {self.rank}: Added remote agent {remote_name} (Rank {r})"
                )

        # Pre-establish connections to optimize first-transfer latency
        for r in self.all_worker_handles:
            if r != self.rank:
                remote_agent_name = f"agent_{r}"
                self.agent.make_connection(remote_agent_name)
        logger.info(f"Rank {self.rank}: All NIXL node connections established.")

    def get_metadata(self) -> bytes:
        """Gets the NIXL metadata for this node."""
        return self.agent.get_agent_metadata()

    def register_block(self, block_index: int, block_module: nn.Module):
        """Registers all parameters and buffers of a Block with NIXL."""
        if block_index in self.registered_blocks:
            self.deregister_block(
                block_index
            )  # First, deregister the old one, just in case

        tensors_to_reg = [p for p in block_module.parameters() if p.numel() > 0]

        if not tensors_to_reg:
            return

        reg_desc = self.agent.get_reg_descs(tensors_to_reg)
        self.agent.register_memory(reg_desc)
        self.registered_blocks[block_index] = tensors_to_reg
        logger.debug(
            f"Rank {self.rank}: Successfully registered memory for Block {block_index}."
        )

    def deregister_block(self, block_index: int):
        """Deregisters all memory for a Block from NIXL."""
        if block_index in self.registered_blocks:
            tensors_to_dereg = self.registered_blocks.pop(block_index)
            if not tensors_to_dereg:
                return
            try:
                reg_desc = self.agent.get_reg_descs(tensors_to_dereg)
                self.agent.deregister_memory(reg_desc)
                logger.debug(
                    f"Rank {self.rank}: Successfully deregistered memory for Block {block_index}."
                )
            except Exception as e:
                # Log as a fallback to avoid affecting subsequent processes
                logger.warning(
                    f"Rank {self.rank}: Failed to deregister memory for Block {block_index}: {e}"
                )

    async def execute_reconstruction_plan(
        self,
        all_needed_pieces: List["NeededPiece"],
        reconstruction_plans: Dict[int, "DataReconstructionPlan"],
        component_name: Optional[str] = None,
    ):
        # 1) Group by source
        grouped_pieces: Dict[int, List["NeededPiece"]] = {}
        for piece in all_needed_pieces:
            if piece.source_rank != -1 and piece.status == "sourced":
                grouped_pieces.setdefault(piece.source_rank, []).append(piece)

        if not grouped_pieces:
            logger.debug(
                f"[NIXL][RECV] rank={self.rank} comp={component_name} no pieces to transfer; return"
            )
            return

        # 2) Pre-calculate the destination segment descriptors for each source and collect the target tensors to be written to for each source
        per_source_desc: Dict[int, List[tuple]] = {}
        per_source_tensors: Dict[int, List[torch.Tensor]] = {}

        for source_rank, pieces_for_this_source in grouped_pieces.items():
            dest_descriptors: List[tuple] = []  # (addr, length_bytes, dev_id)
            dst_tensors: List[torch.Tensor] = []

            for piece in pieces_for_this_source:
                plan = reconstruction_plans[piece.block_index]
                skeleton_module = plan.skeleton_block

                target_tensor = skeleton_module
                for part in piece.layer_path.split("."):
                    target_tensor = getattr(target_tensor, part)
                target_tensor = (
                    target_tensor.bias
                    if piece.tensor_type == "bias"
                    else target_tensor.weight
                )

                base_addr = target_tensor.data_ptr()
                elem_size = target_tensor.element_size()
                dev_id = target_tensor.device.index
                span = piece.abs_end_idx - piece.abs_start_idx
                if span <= 0:
                    logger.debug(
                        f"[NIXL][RECV] rank={self.rank} skip zero-length piece uid={piece.uid}"
                    )
                    continue

                if piece.tensor_type == "bias":
                    start_addr = base_addr + piece.dest_offset * elem_size
                    length_bytes = span * elem_size
                    dest_descriptors.append((start_addr, length_bytes, dev_id))
                    dst_tensors.append(target_tensor)
                else:
                    if piece.split_dim == 0:
                        row_stride_bytes = target_tensor.stride(0) * elem_size
                        start_addr = base_addr + piece.dest_offset * row_stride_bytes
                        length_bytes = span * row_stride_bytes
                        dest_descriptors.append((start_addr, length_bytes, dev_id))
                        dst_tensors.append(target_tensor)
                    else:
                        num_rows = target_tensor.shape[0]
                        row_stride_bytes = target_tensor.stride(0) * elem_size
                        for r in range(num_rows):
                            start_addr = (
                                base_addr
                                + r * row_stride_bytes
                                + piece.dest_offset * elem_size
                            )
                            length_bytes = span * elem_size
                            dest_descriptors.append((start_addr, length_bytes, dev_id))
                        dst_tensors.append(target_tensor)

            dest_descriptors = [d for d in dest_descriptors if d[1] > 0]
            bytes_sum_dst = sum(int(d[1]) for d in dest_descriptors)
            dev_ids = (
                sorted(set(int(d[2]) for d in dest_descriptors))
                if dest_descriptors
                else []
            )
            logger.debug(
                f"[NIXL][RECV][PRE] rank={self.rank} comp={component_name} src={source_rank} dst_desc_cnt={len(dest_descriptors)} bytes_sum={bytes_sum_dst} dev_ids={dev_ids}"
            )

            per_source_desc[source_rank] = dest_descriptors
            per_source_tensors[source_rank] = dst_tensors

        import asyncio

        async with self._xfer_lock:
            # 3) Concurrently prepare and execute transfers for each source
            tasks = []
            for source_rank, pieces_for_this_source in grouped_pieces.items():
                dest_descriptors = per_source_desc.get(source_rank)
                if not dest_descriptors:
                    logger.debug(
                        f"[NIXL][RECV] rank={self.rank} comp={component_name} source={source_rank} no valid dst descriptors; skip"
                    )
                    continue

                # Deduplicate target tensors for this source
                uniq_tensors_src, seen_ids = [], set()
                for t in per_source_tensors.get(source_rank, []):
                    if id(t) not in seen_ids:
                        seen_ids.add(id(t))
                        uniq_tensors_src.append(t)

                # Create a concurrent task
                task = asyncio.create_task(
                    self._execute_transfer_for_source(
                        source_rank=source_rank,
                        pieces=pieces_for_this_source,
                        dest_descriptors=dest_descriptors,
                        tensors_to_register=uniq_tensors_src,
                    )
                )
                tasks.append(task)

            # Wait for all transfer tasks to complete
            if tasks:
                await asyncio.gather(*tasks)

        # 4) Mark as done
        done_cnt = 0
        for piece in all_needed_pieces:
            if piece.status in ("sourced", "transferring"):
                piece.status = "done"
                done_cnt += 1
        logger.debug(
            f"[NIXL][RECV] rank={self.rank} comp={component_name} all transfers done; pieces_marked_done={done_cnt}"
        )
