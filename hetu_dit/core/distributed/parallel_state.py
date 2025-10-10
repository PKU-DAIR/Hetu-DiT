# Copyright 2024 PKUDAIR team.
# Adapted from
# https://github.com/xdit-project/xDiT/blob/main/xfuser/core/distributed/parallel_state.py
# Copyright 2024 xDiT team.
# Copyright 2023 The vLLM team.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
from typing import List, Optional, Dict, Tuple
import torch

import numpy as np

import torch.distributed
import hetu_dit.envs as envs
from itertools import combinations
from hetu_dit.logger import init_logger
from .group_coordinator import (
    GroupCoordinator,
    PipelineGroupCoordinator,
    SequenceParallelGroupCoordinator,
)
from .utils import RankGenerator

env_info = envs.PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]
HAS_FLASH_ATTN = env_info["has_flash_attn"]

logger = init_logger(__name__)
from yunchang.globals import PROCESS_GROUP

_WORLD: Optional[GroupCoordinator] = None
_TP: Optional[GroupCoordinator] = None
_SP: Optional[SequenceParallelGroupCoordinator] = None
_PP: Optional[PipelineGroupCoordinator] = None
_CFG: Optional[GroupCoordinator] = None
_DP: Optional[GroupCoordinator] = None


# * QUERY
def get_world_group(is_serving=True) -> GroupCoordinator:
    if is_serving:
        return get_parallel_groups().activate_world_groups
    else:
        return _WORLD


def get_world_group_world_size():
    """Return world size for the tensor model parallel group."""
    if get_world_group() is None:
        return 1
    else:
        return get_world_group().world_size


def get_world_group_rank():
    """Return my rank for the tensor model parallel group."""
    if get_world_group() is None:
        return 0
    else:
        return get_world_group().rank_in_group


# dp_last
def get_dp_last_group() -> GroupCoordinator:
    # assert _WORLD is not None, "world group is not initialized"
    return get_parallel_groups().activate_dp_last_groups


# TP
def get_tp_group() -> GroupCoordinator:
    return get_parallel_groups().activate_tp_groups


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    if get_tp_group() is None:
        return 1
    else:
        return get_tp_group().world_size


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    if get_tp_group() is None:
        return 0
    else:
        return get_tp_group().rank_in_group


# Text_encoder_TP
def get_text_encoder_tp_group() -> GroupCoordinator:
    return get_parallel_groups().activate_text_encoder_tp_groups


def get_text_encoder_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    if get_text_encoder_tp_group() is None:
        return 1
    else:
        return get_text_encoder_tp_group().world_size


def get_text_encoder_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    if get_text_encoder_tp_group() is None:
        return 0
    else:
        return get_text_encoder_tp_group().rank_in_group


# SP
def get_sp_group() -> SequenceParallelGroupCoordinator:
    return get_parallel_groups().activate_sp_groups


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    if get_sp_group() is None:
        return 1
    else:
        return get_sp_group().world_size


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    if get_sp_group() is None:
        return 0
    else:
        return get_sp_group().rank_in_group


def get_ulysses_parallel_world_size():
    if get_sp_group() is None:
        return 1
    else:
        return get_sp_group().ulysses_world_size


def get_ulysses_parallel_rank():
    if get_sp_group() is None:
        return 0
    else:
        return get_sp_group().ulysses_rank


def get_ring_parallel_world_size():
    if get_sp_group() is None:
        return 1
    else:
        return get_sp_group().ring_world_size


def get_ring_parallel_rank():
    if get_sp_group() is None:
        return 0
    else:
        return get_sp_group().ring_rank


# PP
def get_pp_group() -> PipelineGroupCoordinator:
    return get_parallel_groups().activate_pp_groups


def get_pipeline_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    if get_pp_group() is None:
        return 1
    else:
        return get_pp_group().world_size


def get_pipeline_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    if get_pp_group() is None:
        return 0
    else:
        return get_pp_group().rank_in_group


def is_pipeline_first_stage():
    """Return True if in the first pipeline model parallel stage, False otherwise."""
    return get_pipeline_parallel_rank() == 0


def is_pipeline_last_stage():
    """Return True if in the last pipeline model parallel stage, False otherwise."""
    return get_pipeline_parallel_rank() == (get_pipeline_parallel_world_size() - 1)


# CFG
def get_cfg_group() -> GroupCoordinator:
    return get_parallel_groups().activate_cfg_groups


def get_classifier_free_guidance_world_size():
    """Return world size for the classifier_free_guidance parallel group."""
    return get_cfg_group().world_size


def get_classifier_free_guidance_rank():
    """Return my rank for the classifier_free_guidance parallel group."""
    return get_cfg_group().rank_in_group


# DP
def get_dp_group() -> GroupCoordinator:
    return get_parallel_groups().activate_dp_groups


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return get_dp_group().world_size


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return get_dp_group().rank_in_group


def is_dp_last_group():
    """Return True if in the last data parallel group, False otherwise."""
    return (
        get_sequence_parallel_rank() == (get_sequence_parallel_world_size() - 1)
        and get_classifier_free_guidance_rank()
        == (get_classifier_free_guidance_world_size() - 1)
        and get_pipeline_parallel_rank() == (get_pipeline_parallel_world_size() - 1)
    )


def init_world_group(
    ranks: List[int], local_rank: int, backend: str
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
    )


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
):
    logger.info(
        "world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment"
        )
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )
    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        if distributed_init_method == "env://":
            local_rank = envs.LOCAL_RANK
        else:
            local_rank = rank
    global _WORLD
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
    else:
        assert _WORLD.world_size == torch.distributed.get_world_size(), (
            "world group already initialized with a different world size"
        )


def init_world(
    ranks: List[int],
    local_rank: int,
    backend: str,
    is_serving: bool = False,
    rank: int = -1,
):
    global _WORLD
    if _WORLD is None:
        _WORLD = init_world_group(ranks, local_rank, backend)
    else:
        assert _WORLD.world_size == len(ranks), (
            "world group already initialized with a different world size"
        )


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return (
        _DP is not None
        and _CFG is not None
        and _SP is not None
        and _PP is not None
        and _TP is not None
    )


def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    parallel_mode: str,
    is_serving: bool = False,
    comm_pools=None,
    **kwargs,
) -> GroupCoordinator:
    assert parallel_mode in [
        "data",
        "pipeline",
        "tensor",
        "sequence",
        "classifier_free_guidance",
    ], f"parallel_mode {parallel_mode} is not supported"
    if parallel_mode == "pipeline":
        return PipelineGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            is_serving=is_serving,
            comm_pools=comm_pools,
        )
    elif parallel_mode == "sequence":
        return SequenceParallelGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            is_serving=is_serving,
            comm_pools=comm_pools,
            **kwargs,
        )
    else:
        return GroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            is_serving=is_serving,
            comm_pools=comm_pools,
        )


def initialize_model_parallel(
    data_parallel_degree: int = 1,
    classifier_free_guidance_degree: int = 1,
    sequence_parallel_degree: int = 1,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    tensor_parallel_degree: int = 1,
    pipeline_parallel_degree: int = 1,
    backend: Optional[str] = None,
    is_serving: bool = False,
    world_size: Optional[int] = None,
    ranks: List[int] = [],
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        data_parallel_degree: number of data parallelism groups.
        classifier_free_guidance_degree: number of GPUs used for Classifier Free Guidance (CFG)
        sequence_parallel_degree: number of GPUs used for sequence parallelism.
        ulysses_degree: number of GPUs used for ulysses sequence parallelism.
        ring_degree: number of GPUs used for ring sequence parallelism.
        tensor_parallel_degree: number of GPUs used for tensor parallelism.
        pipeline_parallel_degree: number of GPUs used for pipeline parallelism.
        backend: distributed backend of pytorch collective comm.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 groups to parallelize the batch dim(dp), 2 groups to parallelize
    splited batch caused by CFG, and 2 GPUs to parallelize sequence.

    dp_degree (2) * cfg_degree (2) * sp_degree (2) * pp_degree (2) = 16.

    The present function will create 2 data parallel-groups,
    8 CFG group, 8 pipeline-parallel group, and
    8 sequence-parallel groups:
        2 data-parallel groups:
            [g0, g1, g2, g3, g4, g5, g6, g7],
            [g8, g9, g10, g11, g12, g13, g14, g15]
        8 CFG-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7],
            [g8, g12], [g9, g13], [g10, g14], [g11, g15]
        8 sequence-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7],
            [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        8 pipeline-parallel groups:
            [g0, g2], [g4, g6], [g8, g10], [g12, g14],
            [g1, g3], [g5, g7], [g9, g11], [g13, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    if not is_serving:
        world_size: int = torch.distributed.get_world_size()
    else:
        assert world_size is not None, "world_size must be provided for serving"
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    if (
        world_size
        != data_parallel_degree
        * classifier_free_guidance_degree
        * sequence_parallel_degree
        * tensor_parallel_degree
        * pipeline_parallel_degree
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_parallel_degree ({tensor_parallel_degree}) x "
            f"pipeline_parallel_degree ({pipeline_parallel_degree}) x"
            f"sequence_parallel_degree ({sequence_parallel_degree}) x"
            f"classifier_free_guidance_degree "
            f"({classifier_free_guidance_degree}) x"
            f"data_parallel_degree ({data_parallel_degree})"
        )

    rank_generator: RankGenerator = RankGenerator(
        tensor_parallel_degree,
        sequence_parallel_degree,
        pipeline_parallel_degree,
        classifier_free_guidance_degree,
        data_parallel_degree,
        "tp-sp-pp-cfg-dp",
    )
    if is_serving:
        rank_map = {i: rank for i, rank in enumerate(sorted(ranks))}
    else:
        rank_map = {i: rank for i, rank in enumerate(range(len(ranks)))}
    global _DP
    assert _DP is None, "data parallel group is already initialized"
    if not is_serving:
        _DP = init_model_parallel_group(
            group_ranks=rank_generator.get_ranks("dp"),
            local_rank=get_world_group().local_rank,
            backend=backend,
            parallel_mode="data",
        )
    else:
        base_dp_groups = rank_generator.get_ranks("dp")
        _DP = init_model_parallel_group(
            group_ranks=[[rank_map[r] for r in group] for group in base_dp_groups],
            local_rank=get_world_group().local_rank,
            backend=backend,
            parallel_mode="data",
        )

    global _CFG
    assert _CFG is None, "classifier_free_guidance group is already initialized"
    if not is_serving:
        _CFG = init_model_parallel_group(
            group_ranks=rank_generator.get_ranks("cfg"),
            local_rank=get_world_group().local_rank,
            backend=backend,
            parallel_mode="classifier_free_guidance",
        )
    else:
        base_cfg_groups = rank_generator.get_ranks("cfg")
        _CFG = init_model_parallel_group(
            group_ranks=[[rank_map[r] for r in group] for group in base_cfg_groups],
            local_rank=get_world_group().local_rank,
            backend=backend,
            parallel_mode="classifier_free_guidance",
        )

    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    if not is_serving:
        _PP = init_model_parallel_group(
            group_ranks=rank_generator.get_ranks("pp"),
            local_rank=get_world_group().local_rank,
            backend=backend,
            parallel_mode="pipeline",
        )
    else:
        base_pp_groups = rank_generator.get_ranks("pp")
        _PP = init_model_parallel_group(
            group_ranks=[[rank_map[r] for r in group] for group in base_pp_groups],
            local_rank=get_world_group().local_rank,
            backend=backend,
            parallel_mode="pipeline",
        )

    global _SP
    assert _SP is None, "sequence parallel group is already initialized"

    # if HAS_LONG_CTX_ATTN and sequence_parallel_degree > 1:
    if HAS_LONG_CTX_ATTN:
        from yunchang import set_seq_parallel_pg
        from yunchang.globals import PROCESS_GROUP

        if not is_serving:
            set_seq_parallel_pg(
                sp_ulysses_degree=ulysses_degree,
                sp_ring_degree=ring_degree,
                rank=get_world_group().rank_in_group,
                world_size=get_world_group().world_size,
            )
            _SP = init_model_parallel_group(
                group_ranks=rank_generator.get_ranks("sp"),
                local_rank=get_world_group().local_rank,
                backend=backend,
                parallel_mode="sequence",
                ulysses_group=PROCESS_GROUP.ULYSSES_PG,
                ring_group=PROCESS_GROUP.RING_PG,
            )
        else:
            rank = torch.distributed.get_rank()
            sp_degree = ring_degree * ulysses_degree
            dp_degree = world_size // sp_degree

            assert world_size % sp_degree == 0, (
                f"world_size {world_size} % sp_degree {ulysses_degree} == 0"
            )

            num_ulysses_pgs = ring_degree  # world_size // sp_ulysses_degree
            num_ring_pgs = ulysses_degree  # world_size // sp_ring_degree

            for dp_rank in range(dp_degree):
                offset = dp_rank * sp_degree
                for i in range(num_ulysses_pgs):
                    ulysses_ranks = list(
                        range(
                            i * ulysses_degree + offset,
                            (i + 1) * ulysses_degree + offset,
                        )
                    )
                    ulysses_ranks_new = [rank_map[r] for r in ulysses_ranks]
                    group = torch.distributed.new_group(ulysses_ranks_new)
                    if rank in ulysses_ranks_new:
                        ulysses_pg = group

                for i in range(num_ring_pgs):
                    ring_ranks = list(
                        range(i + offset, sp_degree + offset, num_ring_pgs)
                    )
                    ring_ranks_new = [rank_map[r] for r in ring_ranks]
                    group = torch.distributed.new_group(ring_ranks_new)
                    if rank in ring_ranks_new:
                        ring_pg = group

            base_sp_groups = rank_generator.get_ranks("sp")
            _SP = init_model_parallel_group(
                group_ranks=[[rank_map[r] for r in group] for group in base_sp_groups],
                local_rank=get_world_group().local_rank,
                backend=backend,
                parallel_mode="sequence",
                ulysses_group=ulysses_pg,
                ring_group=ring_pg,
            )

    else:
        if not is_serving:
            _SP = init_model_parallel_group(
                group_ranks=rank_generator.get_ranks("sp"),
                local_rank=get_world_group().local_rank,
                backend=backend,
                parallel_mode="sequence",
            )
        else:
            base_sp_groups = rank_generator.get_ranks("sp")
            _SP = init_model_parallel_group(
                group_ranks=[[rank_map[r] for r in group] for group in base_sp_groups],
                local_rank=get_world_group().local_rank,
                backend=backend,
                parallel_mode="sequence",
            )

    global _TP
    assert _TP is None, "Tensor parallel group is already initialized"
    if not is_serving:
        _TP = init_model_parallel_group(
            group_ranks=rank_generator.get_ranks("tp"),
            local_rank=get_world_group().local_rank,
            backend=backend,
            parallel_mode="tensor",
        )
    else:
        base_tp_groups = rank_generator.get_ranks("tp")
        _TP = init_model_parallel_group(
            group_ranks=[[rank_map[r] for r in group] for group in base_tp_groups],
            local_rank=get_world_group().local_rank,
            backend=backend,
            parallel_mode="tensor",
        )


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _DP
    if _DP:
        _DP.destroy()
    _DP = None

    global _CFG
    if _CFG:
        _CFG.destroy()
    _CFG = None

    global _SP
    if _SP:
        _SP.destroy()
    _SP = None

    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None


def destroy_distributed_environment():
    global _WORLD
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def set_seq_parallel_pg(
    sp_ulysses_degree,
    sp_ring_degree,
    rank,
    world_size,
    ranks: List[int],  # Add the list of actual used ranks
    use_ulysses_low=True,
    is_serving=False,
    comm_pools=None,
):
    """
    sp_ulysses_degree x sp_ring_degree = seq_parallel_degree

    Args:
        sp_ulysses_degree: Ulysses parallelism degree
        sp_ring_degree: Ring parallelism degree
        rank: Current rank
        world_size: Total world size
        ranks: List of actual used ranks, e.g., [0, 3, 4, 7]
        use_ulysses_low: Whether to use Ulysses low mode
    """
    sp_degree = sp_ring_degree * sp_ulysses_degree
    dp_degree = world_size // sp_degree
    assert world_size % sp_degree == 0, (
        f"world_size {world_size} % sp_degree {sp_degree} == 0"
    )

    num_ulysses_pgs = sp_ring_degree
    num_ring_pgs = sp_ulysses_degree

    # create rank map: 0->ranks[0], 1->ranks[1], ...
    rank_map = {i: rank for i, rank in enumerate(sorted(ranks))}
    ulysses_pg = None
    ring_pg = None
    if use_ulysses_low:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ulysses_pgs):
                # generate basic ranks
                base_ulysses_ranks = list(
                    range(
                        i * sp_ulysses_degree + offset,
                        (i + 1) * sp_ulysses_degree + offset,
                    )
                )
                # map to actual ranks
                ulysses_ranks = [rank_map[r] for r in base_ulysses_ranks]
                if is_serving:
                    ulysses_ranks_key = tuple(sorted(ulysses_ranks))
                    group = comm_pools[ulysses_ranks_key]
                else:
                    group = torch.distributed.new_group(ulysses_ranks, backend="nccl")

                if rank in ulysses_ranks:
                    ulysses_pg = group

            for i in range(num_ring_pgs):
                # generate basic ranks
                base_ring_ranks = list(
                    range(i + offset, sp_degree + offset, num_ring_pgs)
                )
                # map to actual ranks
                ring_ranks = [rank_map[r] for r in base_ring_ranks]
                if is_serving:
                    ring_ranks_key = tuple(sorted(ring_ranks))
                    group = comm_pools[ring_ranks_key]
                else:
                    group = torch.distributed.new_group(ring_ranks, backend="nccl")

                if rank in ring_ranks:
                    ring_pg = group

    else:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ring_pgs):
                # generate basic ranks
                base_ring_ranks = list(
                    range(
                        i * sp_ring_degree + offset, (i + 1) * sp_ring_degree + offset
                    )
                )
                # map to actual ranks
                ring_ranks = [rank_map[r] for r in base_ring_ranks]
                if is_serving:
                    ring_ranks_key = tuple(sorted(ring_ranks))
                    group = comm_pools[ring_ranks_key]
                else:
                    group = torch.distributed.new_group(ring_ranks, backend="nccl")
                if rank in ring_ranks:
                    ring_pg = group

            for i in range(num_ulysses_pgs):
                # generate basic ranks
                base_ulysses_ranks = list(
                    range(i + offset, sp_degree + offset, num_ulysses_pgs)
                )
                # map to actual ranks
                ulysses_ranks = [rank_map[r] for r in base_ulysses_ranks]
                if is_serving:
                    ulysses_ranks_key = tuple(sorted(ulysses_ranks))
                    group = comm_pools[ulysses_ranks_key]
                else:
                    group = torch.distributed.new_group(ulysses_ranks, backend="nccl")
                if rank in ulysses_ranks:
                    ulysses_pg = group

    return ulysses_pg, ring_pg


class ParallelGroups:
    """
    Manage all possible communication groups
    """

    def __init__(
        self,
        world_size: int = -1,
        rank: int = -1,
        distributed_init_method: str = "env://",
        local_rank: int = -1,
        backend: str = "nccl",
    ):
        logger.info("-----init parallel group------")
        self.activate_world_groups = None
        self.activate_tp_groups = None
        self.activate_sp_groups = None
        self.activate_pp_groups = None
        self.activate_cfg_groups = None
        self.activate_dp_groups = None
        self.activate_text_encoder_tp_groups = None

        # For vae parallel
        self.activate_dp_last_groups = None

    def lazy_init(
        self,
        world_size: int = -1,
        rank: int = -1,
        distributed_init_method: str = "env://",
        local_rank: int = -1,
        backend: str = "nccl",
        is_serving: bool = False,
    ):
        # Initialize the distributed environment
        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                distributed_init_method=distributed_init_method,
                local_rank=local_rank,
                backend=backend,
            )  # Initialized _WORLD, to be modified

        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
        num_gpus_per_node = 8
        local_rank = global_rank % num_gpus_per_node
        logger.info(
            f"in lazy_init, world_size = {world_size}, global_rank = {global_rank}, local_rank = {local_rank}, num_gpu_per_node = {num_gpus_per_node}"
        )
        self._comm_pools = {}
        # store all possible parallel groups
        # key format: (degree, ranks_tuple, group_ranks_tuple), sp_group is (degree, ranks_tuple, group_ranks_tuple, ulysses_degree, ring_degree)
        self._tp_groups: Dict[
            Tuple[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]], GroupCoordinator
        ] = {}
        self._sp_groups: Dict[
            Tuple[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
            SequenceParallelGroupCoordinator,
        ] = {}
        self._pp_groups: Dict[
            Tuple[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
            PipelineGroupCoordinator,
        ] = {}
        self._cfg_groups: Dict[
            Tuple[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]], GroupCoordinator
        ] = {}
        self._dp_groups: Dict[
            Tuple[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]], GroupCoordinator
        ] = {}
        self._world_groups: Dict[
            Tuple[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]], GroupCoordinator
        ] = {}

        self.valid_degrees = [1, 2, 4, 8]

        # Create all possible communication groups at initialization
        self._generate_comm_pool(local_rank, backend)
        self._initialize_all_groups(local_rank, backend, is_serving)

    def _generate_rank_combinations(
        self, world_size: int, base_num=0
    ) -> List[List[int]]:
        """Generate all possible rank combinations (one-dimensional list)"""
        combinations_list = []
        for degree in self.valid_degrees:
            if degree > world_size:
                continue
            degree_combinations = [
                list(c) for c in combinations(range(world_size), degree)
            ]
            degree_combinations_inter = np.array(degree_combinations) + base_num
            degree_combinations = degree_combinations_inter.tolist()
            combinations_list.extend(degree_combinations)
        return combinations_list

    def _generate_group_ranks(self, ranks: List[int]) -> List[List[List[int]]]:
        """Generate all possible grouping methods for a rank combination (two-dimensional list)

        For example: ranks=[0,1,2,3]
        return: [
            [[0], [1], [2], [3]],          # 1+1+1+1
            [[0,1], [2,3]], [[0,2], [1,3]], [[0,3], [1,2]],  # 2+2
            [[0,1,2,3]]                     # 4
        ]
        Note: [[0,1], [2,3]] and [[2,3], [0,1]] are regarded as the same group
        """

        def get_all_partitions(n: int) -> List[List[int]]:
            """Obtain all 2-power partitions of the number n"""
            if n == 1:
                return [[1]]
            partitions = []
            for i in range(n.bit_length()):
                part = 1 << i  # 2‘s power
                if part > n:
                    break
                if n % part == 0:
                    partitions.append([part] * (n // part))
            return partitions

        def generate_groupings(
            ranks: List[int], partition: List[int]
        ) -> List[List[List[int]]]:
            """According to a partition, generate all possible grouping methods."""
            if len(partition) == 1:
                return [[ranks]]

            result = []
            first_group_size = partition[0]

            # Always include the first rank in the first group, so as to avoid generating equivalent groups.
            first_rank = ranks[0]
            remaining_ranks = ranks[1:]

            # From the remaining ranks, select first_group_size - 1 elements
            for other_members in combinations(remaining_ranks, first_group_size - 1):
                first_group = [first_rank] + list(other_members)
                remaining = [r for r in ranks if r not in first_group]

                # Recursively generate the grouping of the remaining part
                sub_groupings = generate_groupings(remaining, partition[1:])
                for sub_grouping in sub_groupings:
                    result.append([first_group] + sub_grouping)

            return result

        n = len(ranks)
        all_partitions = get_all_partitions(n)
        all_groupings = []

        for partition in all_partitions:
            groupings = generate_groupings(ranks, partition)
            all_groupings.extend(groupings)

        return all_groupings

    def get_tp_group(
        self, degree: int, ranks: List[int], group_ranks: List[List[int]]
    ) -> GroupCoordinator:
        key = (degree, tuple(sorted(ranks)), tuple(map(tuple, group_ranks)))
        group = self._tp_groups.get(key)
        assert group is not None, (
            f"tensor parallel group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}"
        )
        return group

    def get_world_group(
        self, degree: int, ranks: List[int], group_ranks: List[List[int]]
    ) -> GroupCoordinator:
        key = (degree, tuple(sorted(ranks)), tuple(map(tuple, group_ranks)))
        group = self._world_groups.get(key)
        assert group is not None, (
            f"tensor parallel group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}"
        )
        return group

    def get_sp_group(
        self,
        degree: int,
        ranks: List[int],
        group_ranks: List[List[int]],
        ulysses_degree: int,
        ring_degree: int,
    ) -> SequenceParallelGroupCoordinator:
        key = (
            degree,
            tuple(sorted(ranks)),
            tuple(map(tuple, group_ranks)),
            ulysses_degree,
            ring_degree,
        )
        group = self._sp_groups.get(key)
        assert group is not None, (
            f"sequence parallel group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}"
        )
        return group

    def get_pp_group(
        self, degree: int, ranks: List[int], group_ranks: List[List[int]]
    ) -> PipelineGroupCoordinator:
        key = (degree, tuple(sorted(ranks)), tuple(map(tuple, group_ranks)))
        group = self._pp_groups.get(key)
        assert group is not None, (
            f"pipeline parallel group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}"
        )
        return group

    def get_cfg_group(
        self, degree: int, ranks: List[int], group_ranks: List[List[int]]
    ) -> GroupCoordinator:
        key = (degree, tuple(sorted(ranks)), tuple(map(tuple, group_ranks)))
        group = self._cfg_groups.get(key)
        assert group is not None, (
            f"cfg group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}"
        )
        return group

    def get_dp_group(
        self, degree: int, ranks: List[int], group_ranks: List[List[int]]
    ) -> GroupCoordinator:
        key = (degree, tuple(sorted(ranks)), tuple(map(tuple, group_ranks)))
        group = self._dp_groups.get(key)
        assert group is not None, (
            f"data parallel group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}"
        )
        return group

    def set_activate_tp_group(
        self, degree: int, ranks: List[int], group_ranks: List[List[int]]
    ) -> GroupCoordinator:
        key = (degree, tuple(sorted(ranks)), tuple(map(tuple, group_ranks)))
        group = self._tp_groups.get(key)
        assert group is not None, (
            f"tensor parallel group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}"
        )
        self.activate_tp_groups = group

    def set_activate_text_encoder_tp_group(
        self, degree: int, ranks: List[int], group_ranks: List[List[int]]
    ) -> GroupCoordinator:
        key = (degree, tuple(sorted(ranks)), tuple(map(tuple, group_ranks)))
        group = self._tp_groups.get(key)
        assert group is not None, (
            f"tensor parallel group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}"
        )
        self.activate_text_encoder_tp_groups = group

    def set_activate_world_group(
        self, degree: int, ranks: List[int], group_ranks: List[List[int]]
    ) -> GroupCoordinator:
        key = (degree, tuple(sorted(ranks)), tuple(map(tuple, group_ranks)))
        group = self._world_groups.get(key)
        assert group is not None, (
            f"tensor parallel group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}"
        )
        self.activate_world_groups = group

    def set_activate_dp_last_group(
        self, degree: int, ranks: List[int], group_ranks: List[List[int]]
    ) -> GroupCoordinator:
        key = (degree, tuple(sorted(ranks)), tuple(map(tuple, group_ranks)))
        group = self._world_groups.get(key)
        assert group is not None, (
            f"tensor parallel group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}"
        )
        self.activate_dp_last_groups = group

    def set_activate_sp_group(
        self,
        degree: int,
        ranks: List[int],
        group_ranks: List[List[int]],
        ulysses_degree: int = -1,
        ring_degree: int = -1,
    ) -> SequenceParallelGroupCoordinator:
        key = (
            degree,
            tuple(sorted(ranks)),
            tuple(map(tuple, group_ranks)),
            ulysses_degree,
            ring_degree,
        )
        group = self._sp_groups.get(key)
        assert group is not None, (
            f"sequence parallel group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}, ulysses_degree={ulysses_degree}, ring_degree={ring_degree}"
        )
        self.activate_sp_groups = group
        # Must set PROCESS_GROUP.ULYSSES_PG and PROCESS_GROUP.RING_PG because the constrain of yunchang
        global PROCESS_GROUP
        PROCESS_GROUP.ULYSSES_PG = group.ulysses_group
        PROCESS_GROUP.RING_PG = group.ring_group

    def set_activate_pp_group(
        self, degree: int, ranks: List[int], group_ranks: List[List[int]]
    ) -> PipelineGroupCoordinator:
        key = (degree, tuple(sorted(ranks)), tuple(map(tuple, group_ranks)))
        group = self._pp_groups.get(key)
        assert group is not None, (
            f"pipeline parallel group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}"
        )
        self.activate_pp_groups = group

    def set_activate_cfg_group(
        self, degree: int, ranks: List[int], group_ranks: List[List[int]]
    ) -> GroupCoordinator:
        key = (degree, tuple(sorted(ranks)), tuple(map(tuple, group_ranks)))
        group = self._cfg_groups.get(key)
        assert group is not None, (
            f"cfg group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}"
        )
        self.activate_cfg_groups = group

    def set_activate_dp_group(
        self, degree: int, ranks: List[int], group_ranks: List[List[int]]
    ) -> GroupCoordinator:
        key = (degree, tuple(sorted(ranks)), tuple(map(tuple, group_ranks)))
        group = self._dp_groups.get(key)
        assert group is not None, (
            f"data parallel group not found for degree={degree}, ranks={ranks}, group_ranks={group_ranks}"
        )
        self.activate_dp_groups = group

    def _generate_sp_group_ranks(
        self, ranks: List[int]
    ) -> List[Tuple[List[List[int]], int, int]]:
        """Generate all possible combinations of group_ranks for sequence parallel

        Args:
            ranks: The ranks list to participate in sequence parallel

        Returns:
            List of (group_ranks, ulysses_degree, ring_degree)
        """
        n = len(ranks)
        sp_combinations = []

        # get all possible ulysses_degree and ring_degree combinations
        for i in range(n.bit_length()):
            ulysses_degree = 1 << i  # 2^i
            if ulysses_degree > n:
                break

            for j in range(n.bit_length()):
                ring_degree = 1 << j  # 2^i
                if ring_degree > n:
                    break

                if ulysses_degree * ring_degree == n:
                    # Use the logic of set_seq_parallel_pg to generate group_ranks
                    num_ulysses_pgs = ring_degree
                    num_ring_pgs = ulysses_degree

                    # generate ulysses groups
                    ulysses_groups = []
                    for i in range(num_ulysses_pgs):
                        ulysses_ranks = ranks[
                            i * ulysses_degree : (i + 1) * ulysses_degree
                        ]
                        ulysses_groups.append(ulysses_ranks)

                    # generate ring groups
                    ring_groups = []
                    for i in range(num_ring_pgs):
                        ring_ranks = ranks[i::num_ring_pgs]
                        ring_groups.append(ring_ranks)

                    # combine as one group_ranks
                    group_ranks = ulysses_groups + ring_groups
                    sp_combinations.append((group_ranks, ulysses_degree, ring_degree))

        return sp_combinations

    def _initialize_all_groups(
        self, local_rank: int, backend: str, is_serving: bool = False
    ):
        """init all possible grups"""
        world_size = torch.distributed.get_world_size()
        if world_size <= 8:
            # 1. Generate all possible rank combinations
            rank_combinations = self._generate_rank_combinations(world_size)

            # 2. Generate all possible grouping methods for each rank combination.
            for ranks in rank_combinations:
                degree = len(ranks)
                ranks_key = tuple(sorted(ranks))

                # Special processing for sequence parallel
                def generate_ulysses_ring_degree(n):
                    """
                    Returns all unique permutations of two factors of the given integer n.

                    Parameters:
                        n (int): The integer to be factored.

                    Returns:
                        list: A list of tuples where each tuple contains a pair of factors of n.
                    """
                    factors = []
                    for i in range(1, int(n**0.5) + 1):
                        if n % i == 0:
                            factors.append((i, n // i))  # Add (i, n // i)
                            if i != n // i:
                                factors.append(
                                    (n // i, i)
                                )  # Add (n // i, i) if it's not a duplicate
                    return factors

                # generate all possible group ranks
                all_group_ranks = self._generate_group_ranks(ranks)
                for group_ranks in all_group_ranks:
                    group_ranks_key = tuple(map(tuple, group_ranks))

                    torch.distributed.barrier()
                    self._tp_groups[(degree, ranks_key, group_ranks_key)] = (
                        init_model_parallel_group(
                            group_ranks=group_ranks,
                            local_rank=local_rank,
                            backend=backend,
                            parallel_mode="tensor",
                            is_serving=is_serving,
                            comm_pools=self._comm_pools,
                        )
                    )
                    torch.distributed.barrier()

                    # create sequence parallel group if HAS_LONG_CTX_ATTN is True
                    if HAS_LONG_CTX_ATTN:

                        def generate_powers_of_two(max_value):
                            power = 1
                            while power <= max_value:
                                yield power
                                power *= 2

                        for sp_parallel_degree in generate_powers_of_two(degree):
                            degree_combine = generate_ulysses_ring_degree(
                                sp_parallel_degree
                            )
                            for ulysses_degree, ring_degree in degree_combine:
                                group_ranks_key = tuple(map(tuple, group_ranks))
                                # set sequence parallel's process groups
                                ulysses_group, ring_group = set_seq_parallel_pg(
                                    sp_ulysses_degree=ulysses_degree,
                                    sp_ring_degree=ring_degree,
                                    rank=torch.distributed.get_rank(),
                                    world_size=len(ranks),
                                    ranks=ranks,  # actual ranks
                                    is_serving=is_serving,
                                    comm_pools=self._comm_pools,
                                )

                                torch.distributed.barrier()

                                # create sequence parallel group

                                self._sp_groups[
                                    (
                                        degree,
                                        ranks_key,
                                        group_ranks_key,
                                        ulysses_degree,
                                        ring_degree,
                                    )
                                ] = init_model_parallel_group(
                                    group_ranks=group_ranks,
                                    local_rank=local_rank,
                                    backend=backend,
                                    parallel_mode="sequence",
                                    is_serving=is_serving,
                                    comm_pools=self._comm_pools,
                                    ulysses_group=ulysses_group,
                                    ring_group=ring_group,
                                )

                    else:
                        self._sp_groups[
                            (degree, ranks_key, group_ranks_key, -1, -1)
                        ] = init_model_parallel_group(
                            group_ranks=group_ranks,
                            local_rank=local_rank,
                            backend=backend,
                            parallel_mode="sequence",
                            is_serving=is_serving,
                            comm_pools=self._comm_pools,
                        )
                        torch.distributed.barrier()

                    self._pp_groups[(degree, ranks_key, group_ranks_key)] = (
                        init_model_parallel_group(
                            group_ranks=group_ranks,
                            local_rank=local_rank,
                            backend=backend,
                            parallel_mode="pipeline",
                            is_serving=is_serving,
                            comm_pools=self._comm_pools,
                        )
                    )
                    torch.distributed.barrier()

                    self._cfg_groups[(degree, ranks_key, group_ranks_key)] = (
                        init_model_parallel_group(
                            group_ranks=group_ranks,
                            local_rank=local_rank,
                            backend=backend,
                            parallel_mode="classifier_free_guidance",
                            is_serving=is_serving,
                            comm_pools=self._comm_pools,
                        )
                    )
                    torch.distributed.barrier()

                    self._dp_groups[(degree, ranks_key, group_ranks_key)] = (
                        init_model_parallel_group(
                            group_ranks=group_ranks,
                            local_rank=local_rank,
                            backend=backend,
                            parallel_mode="data",
                            is_serving=is_serving,
                            comm_pools=self._comm_pools,
                        )
                    )
                    torch.distributed.barrier()

                    self._world_groups[(degree, ranks_key, group_ranks_key)] = (
                        GroupCoordinator(
                            group_ranks=group_ranks,
                            local_rank=local_rank,
                            torch_distributed_backend=backend,
                            is_serving=is_serving,
                            comm_pools=self._comm_pools,
                        )
                    )
                    torch.distributed.barrier()
        else:
            if world_size % 8 != 0:
                raise ValueError(
                    f"world_size {world_size} must be a multiple of 8 for Hetu-dit"
                )
            num_iter = world_size // 8
            for iter in range(num_iter):
                # 1. Generate all possible rank combinations
                rank_combinations = self._generate_rank_combinations(
                    8, base_num=iter * 8
                )

                # 2. Generate all possible grouping methods for each rank combination.
                for ranks in rank_combinations:
                    degree = len(ranks)
                    ranks_key = tuple(sorted(ranks))

                    # Special processing for sequence parallel
                    def generate_ulysses_ring_degree(n):
                        """
                        Returns all unique permutations of two factors of the given integer n.

                        Parameters:
                            n (int): The integer to be factored.

                        Returns:
                            list: A list of tuples where each tuple contains a pair of factors of n.
                        """
                        factors = []
                        for i in range(1, int(n**0.5) + 1):
                            if n % i == 0:
                                factors.append((i, n // i))  # Add (i, n // i)
                                if i != n // i:
                                    factors.append(
                                        (n // i, i)
                                    )  # Add (n // i, i) if it's not a duplicate
                        return factors

                    # generate all possible group ranks
                    all_group_ranks = self._generate_group_ranks(ranks)
                    for group_ranks in all_group_ranks:
                        group_ranks_key = tuple(map(tuple, group_ranks))

                        torch.distributed.barrier()
                        self._tp_groups[(degree, ranks_key, group_ranks_key)] = (
                            init_model_parallel_group(
                                group_ranks=group_ranks,
                                local_rank=local_rank,
                                backend=backend,
                                parallel_mode="tensor",
                                is_serving=is_serving,
                                comm_pools=self._comm_pools,
                            )
                        )
                        torch.distributed.barrier()

                        # create sequence parallel group if HAS_LONG_CTX_ATTN is True
                        if HAS_LONG_CTX_ATTN:

                            def generate_powers_of_two(max_value):
                                power = 1
                                while power <= max_value:
                                    yield power
                                    power *= 2

                            for sp_parallel_degree in generate_powers_of_two(degree):
                                degree_combine = generate_ulysses_ring_degree(
                                    sp_parallel_degree
                                )

                                for ulysses_degree, ring_degree in degree_combine:
                                    group_ranks_key = tuple(map(tuple, group_ranks))
                                    # set sequence parallel's process groups
                                    ulysses_group, ring_group = set_seq_parallel_pg(
                                        sp_ulysses_degree=ulysses_degree,
                                        sp_ring_degree=ring_degree,
                                        rank=torch.distributed.get_rank(),
                                        world_size=len(ranks),
                                        ranks=ranks,  # actual ranks
                                        is_serving=is_serving,
                                        comm_pools=self._comm_pools,
                                    )

                                    torch.distributed.barrier()

                                    # create sequence parallel group

                                    self._sp_groups[
                                        (
                                            degree,
                                            ranks_key,
                                            group_ranks_key,
                                            ulysses_degree,
                                            ring_degree,
                                        )
                                    ] = init_model_parallel_group(
                                        group_ranks=group_ranks,
                                        local_rank=local_rank,
                                        backend=backend,
                                        parallel_mode="sequence",
                                        is_serving=is_serving,
                                        comm_pools=self._comm_pools,
                                        ulysses_group=ulysses_group,
                                        ring_group=ring_group,
                                    )

                        else:
                            self._sp_groups[
                                (degree, ranks_key, group_ranks_key, -1, -1)
                            ] = init_model_parallel_group(
                                group_ranks=group_ranks,
                                local_rank=local_rank,
                                backend=backend,
                                parallel_mode="sequence",
                                is_serving=is_serving,
                                comm_pools=self._comm_pools,
                            )
                            torch.distributed.barrier()

                        self._pp_groups[(degree, ranks_key, group_ranks_key)] = (
                            init_model_parallel_group(
                                group_ranks=group_ranks,
                                local_rank=local_rank,
                                backend=backend,
                                parallel_mode="pipeline",
                                is_serving=is_serving,
                                comm_pools=self._comm_pools,
                            )
                        )
                        torch.distributed.barrier()

                        self._cfg_groups[(degree, ranks_key, group_ranks_key)] = (
                            init_model_parallel_group(
                                group_ranks=group_ranks,
                                local_rank=local_rank,
                                backend=backend,
                                parallel_mode="classifier_free_guidance",
                                is_serving=is_serving,
                                comm_pools=self._comm_pools,
                            )
                        )
                        torch.distributed.barrier()

                        self._dp_groups[(degree, ranks_key, group_ranks_key)] = (
                            init_model_parallel_group(
                                group_ranks=group_ranks,
                                local_rank=local_rank,
                                backend=backend,
                                parallel_mode="data",
                                is_serving=is_serving,
                                comm_pools=self._comm_pools,
                            )
                        )
                        torch.distributed.barrier()

                        self._world_groups[(degree, ranks_key, group_ranks_key)] = (
                            GroupCoordinator(
                                group_ranks=group_ranks,
                                local_rank=local_rank,
                                torch_distributed_backend=backend,
                                is_serving=is_serving,
                                comm_pools=self._comm_pools,
                            )
                        )
                        torch.distributed.barrier()

    def _generate_comm_pool(self, local_rank: int, backend: str):
        """init all possible grups"""
        world_size = torch.distributed.get_world_size()
        if world_size <= 8:
            # 1. Generate all possible rank combinations
            rank_combinations = self._generate_rank_combinations(world_size)
            logger.debug(f"rank_combinations={rank_combinations}")
            logger.debug("begin to generate comm pool")
            # 2. Generate all possible grouping methods for each rank combination.
            for ranks in rank_combinations:
                # generate all possible group ranks
                all_group_ranks = self._generate_group_ranks(ranks)
                for group_ranks in all_group_ranks:
                    group_ranks_key = tuple(map(tuple, group_ranks))

                    torch.distributed.barrier()
                    rank = torch.distributed.get_rank()

                    for sub_ranks in group_ranks:
                        sub_ranks_key = tuple(sorted(sub_ranks))
                        device_group = torch.distributed.new_group(
                            sub_ranks, backend=backend
                        )
                        # if rank in ranks:
                        self._comm_pools[sub_ranks_key] = device_group
                    torch.distributed.barrier()
            logger.debug("end to generate comm pool")
        else:
            if world_size % 8 != 0:
                raise ValueError(
                    f"world_size {world_size} must be a multiple of 8 for Hetu-dit"
                )
            num_iter = world_size // 8
            for i in range(num_iter):
                # 1. Generate all possible rank combinations
                rank_combinations = self._generate_rank_combinations(8, base_num=i * 8)
                logger.debug(f"rank_combinations={rank_combinations}")
                logger.debug("begin to generate comm pool")
                # 2. Generate all possible grouping methods for each rank combination.
                for ranks in rank_combinations:
                    # generate all possible group ranks
                    all_group_ranks = self._generate_group_ranks(ranks)
                    for group_ranks in all_group_ranks:
                        group_ranks_key = tuple(map(tuple, group_ranks))

                        torch.distributed.barrier()
                        logger.debug(f"before comm_pools group_ranks={group_ranks}")
                        rank = torch.distributed.get_rank()

                        for sub_ranks in group_ranks:
                            sub_ranks_key = tuple(sorted(sub_ranks))
                            device_group = torch.distributed.new_group(
                                sub_ranks, backend=backend
                            )
                            # if rank in ranks:
                            self._comm_pools[sub_ranks_key] = device_group
                        torch.distributed.barrier()
                logger.debug("end to generate comm pool")

    def destroy(self):
        """destroy all possible groups"""

        for group in self._tp_groups.values():
            group.destroy()
        self._tp_groups.clear()

        for group in self._sp_groups.values():
            group.destroy()
        self._sp_groups.clear()

        for group in self._pp_groups.values():
            group.destroy()
        self._pp_groups.clear()

        for group in self._cfg_groups.values():
            group.destroy()
        self._cfg_groups.clear()

        for group in self._dp_groups.values():
            group.destroy()
        self._dp_groups.clear()


# global instance
_PARALLEL_GROUPS = ParallelGroups()


def get_parallel_groups() -> ParallelGroups:
    pass
    global _PARALLEL_GROUPS
    return _PARALLEL_GROUPS
