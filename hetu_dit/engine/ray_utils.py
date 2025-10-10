from typing import Dict, List, Optional, Tuple
import os
from hetu_dit.config import ParallelConfig
import hetu_dit.envs as envs
from hetu_dit.logger import init_logger

import inspect
import asyncio
from concurrent.futures import ThreadPoolExecutor
from hetu_dit.utils import get_ip, get_open_port
from hetu_dit.config.config import EngineConfig

logger = init_logger(__name__)

PG_WAIT_TIMEOUT = 1800

try:
    import ray

    class RayWorkerHetudit:
        """Ray wrapper for hetu_dit.worker.Worker, allowing Worker to be
        lazliy initialized after Ray sets CUDA_VISIBLE_DEVICES."""

        def __init__(self, global_rank=0, work_dir=None, machine_id=0) -> None:
            self.state = "idle"  # Possible states: 'idle', 'ready', 'busy'
            self.machine_id = machine_id
            if work_dir is not None:
                os.chdir(work_dir)
                self.work_dir = work_dir
            logger.debug(
                f"in rayworkerhetudit init, machine_id is {machine_id}, work_dir = {os.getcwd()},"
            )
            self.global_rank = global_rank
            self.local_rank = global_rank % 8
            self.sync_task_executor = ThreadPoolExecutor(
                max_workers=8, thread_name_prefix=f"RayWorkerHetudit_{global_rank}"
            )

        def init_worker(self, worker_init_fn):
            logger.debug(f"init_worker's pid = {os.getpid()}")
            self.worker = worker_init_fn()
            logger.debug(
                f"in rayworkerhetudit init_worker, rank = {self.global_rank}, local_rank = {self.local_rank}"
            )
            logger.debug(f"worker's type is {type(self.worker)}")

        def init_worker_env(
            self,
            engine_config: EngineConfig,
            rank: int,
            distributed_init_method: Optional[str] = None,
        ):
            ray.get(
                self.worker.init_worker_distributed_environment.remote(
                    engine_config, rank, distributed_init_method
                )
            )

        def __getattr__(self, name):
            return getattr(self.worker, name)

        async def execute_method(self, method, *args, **kwargs):
            # self.set_state("busy")
            try:
                executor_func = getattr(self, method)

                if inspect.iscoroutinefunction(
                    executor_func
                ) or inspect.isasyncgenfunction(executor_func):
                    logger.debug(f"Executing async method: {method}")
                    return await executor_func(*args, **kwargs)
                else:
                    logger.debug(f"Executing sync method in thread pool: {method}")
                    loop = asyncio.get_running_loop()

                    return await loop.run_in_executor(
                        self.sync_task_executor, lambda: executor_func(*args, **kwargs)
                    )
            finally:
                pass
                # self.set_state("ready")

        def get_states(self):
            executor = getattr(self, "get_worker_state")
            states = executor()
            return {
                "state": self.state,
                **states,
            }

        def get_node_ip(self) -> str:
            return get_ip()

        def get_node_open_port(self):
            return get_open_port()

        def get_node_and_gpu_ids(self) -> Tuple[str, List[int]]:
            logger.debug(f"get_node_and_gpu_ids's pid = {os.getpid()}")
            node_id = ray.get_runtime_context().get_node_id()
            gpu_ids = ray.get_gpu_ids()
            return node_id, gpu_ids

        def set_cuda_visible_devices(self, device_ids) -> None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

        def get_state(self):
            return self.state

        def set_state(self, new_state: str):
            if new_state not in {"idle", "ready", "busy"}:
                raise ValueError(f"Invalid state: {new_state}")

            self.state = new_state
            logger.debug(
                f"worker is {self}, rank is {self.global_rank}, set state to {new_state}, self.state is {self.get_state()}"
            )

        def get_machine_id(self):
            return self.machine_id

        def get_global_rank(self):
            return self.global_rank

        def get_local_rank(self):
            return self.local_rank

        def set_worker_handles(self, all_worker_handles: dict):
            if hasattr(self, "worker") and self.worker is not None:
                self.worker.all_worker_handles = all_worker_handles
                logger.info(
                    f"Rank {self.worker.rank}: Successfully set {len(all_worker_handles)} worker handles: {list(all_worker_handles.keys())}"
                )
                return True
            else:
                logger.error(
                    "Warning: Worker not initialized yet. Handles will not be set."
                )
                return False

        async def get_nixl_metadata(self) -> Tuple[int, bytes]:
            return await self.worker.get_nixl_metadata()

        async def init_nixl_peers(self, all_metadata_list: List[Tuple[int, bytes]]):
            return await self.worker.init_nixl_peers(all_metadata_list)

        async def register_existing_cache_with_nixl(self):
            return await self.worker.register_existing_cache_with_nixl()

        async def rpc_find_sources(self, serializable_pieces: List[Dict]) -> List[int]:
            return await self.worker.rpc_find_sources(serializable_pieces)

        async def rpc_nixl_send_data(
            self,
            pieces_to_send: List["NeededPiece"],
            remote_xfer_desc_bytes: bytes,
            remote_partial_metadata: bytes,
            requester_rank: int,
        ):
            return await self.worker.rpc_nixl_send_data(
                pieces_to_send,
                remote_xfer_desc_bytes,
                remote_partial_metadata,
                requester_rank,
            )

        async def create_nixl_manager(self):
            return await self.worker.create_nixl_manager()

        async def nixl_preflight_get_md(self):
            return await self.worker.nixl_preflight_get_md()

        async def nixl_preflight_connect(self, all_metadata_list):
            return await self.worker.nixl_preflight_connect(all_metadata_list)

except ImportError as e:
    logger.warning(
        f"Failed to import Ray with {e!r}. "
        "For distributed inference, please install Ray with "
        "`pip install ray`."
    )
    ray = None
    RayWorkerVllm = None


def initialize_ray_cluster(
    parallel_config: ParallelConfig,
    ray_address: Optional[str] = None,
):
    """Initialize the distributed cluster with Ray.

    It will connect to the Ray cluster and create a placement group
    for the all workers, which includes the specification of the resources
    for each distributed worker.
    It only called once in the static placement, and when we need to use small comm group, we will not create new placement group, we just need to create new torch comm group.

    Args:
        parallel_config: The configurations for parallel execution.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.
    """
    if ray is None:
        raise ImportError(
            "Ray is not installed. Please install Ray to use distributed serving."
        )

    ray.init(
        address=ray_address,
        ignore_reinit_error=True,
        runtime_env={"env_vars": dict(os.environ)},
    )

    if parallel_config.placement_group:
        # Placement group is already set.
        return

    # Create placement group for worker processes
    current_placement_group = ray.util.get_current_placement_group()
    if current_placement_group:
        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        gpu_bundles = 0
        for bundle in bundles:
            bundle_gpus = bundle.get("GPU", 0)
            if bundle_gpus > 1:
                raise ValueError("Placement group bundle cannot have more than 1 GPU.")
            if bundle_gpus:
                gpu_bundles += 1
        if parallel_config.world_size > gpu_bundles:
            raise ValueError(
                "The number of required GPUs exceeds the total number of "
                "available GPUs in the placement group."
            )
    else:
        num_gpus_in_cluster = ray.cluster_resources().get("GPU", 0)
        if parallel_config.world_size > num_gpus_in_cluster:
            raise ValueError(
                "The number of required GPUs exceeds the total number of "
                "available GPUs in the cluster."
            )
        # Create a new placement group with 2 CPUs and 1 GPU per bundle
        placement_group_specs = [{"GPU": 1, "CPU": 10}] * int(num_gpus_in_cluster)
        current_placement_group = ray.util.placement_group(placement_group_specs)
        # Wait until PG is ready - this will block until all
        # requested resources are available, and will timeout
        # if they cannot be provisioned.
        ray.get(current_placement_group.ready(), timeout=1800)

    # Set the placement group in the parallel config
    parallel_config.placement_group = current_placement_group
