from hetu_dit.utils import Counter
from collections import defaultdict
from typing import Dict, List, Type, Any
from hetu_dit.config.config import EngineConfig
from hetu_dit.core.request_manager.request_manager import RequestManager
from hetu_dit.config.config import ServingConfig
from hetu_dit.executor.executor_base import ExecutorBase
from .ray_utils import initialize_ray_cluster, RayWorkerHetudit
import os
import ray
import copy
from hetu_dit.executor.gpu_executor import RayGPUExecutor
from hetu_dit.utils import (
    get_ip,
    set_cuda_visible_devices,
    get_distributed_init_method,
    get_open_port,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from hetu_dit.worker.worker import Worker
from hetu_dit.logger import init_logger

logger = init_logger(__name__)


class ServingEngine:
    """
    A serving engine retains a singleton CPU model, allowing for direct uploads from CPU to GPU. At the same time, it retains information about all global workers,
    binding a fixed GPU to each worker. It also keeps references to multiple RayGPUExecutors, responsible for passing the composed workers for execution.
    When a request comes in, it can decide whether to execute directly based on whether this RayGPUExecutor can be reused.
    If it cannot be reused, we may create a new RayGPUExecutor for execution. When a GPU combination is dismantled,
    the RayGPUExecutor responsible for that group of GPUs will also be destroyed.
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        model_class: Type,
        executor_class: Type[ExecutorBase],
    ) -> None:
        self.engine_config = engine_config
        self.model_config = engine_config.model_config
        self.runtime_config = engine_config.runtime_config
        self.parallel_config = engine_config.parallel_config
        self.model_class = model_class
        self.executor_class = executor_class
        self.reqs_counter = Counter()
        self.driver_dummy_worker = None
        self.all_workers: List[
            RayWorkerHetudit
        ] = []  # manage all the workers globally,and dispatch it to different RayGPUExecutor

        self.all_executors: List[
            RayGPUExecutor
        ] = []  # manage all the RayGPUExecutor dynamically

        self.scheduler = RequestManager()

        # init all the possible workers
        self._init_workers_ray(self.parallel_config.placement_group)

        self.global_executor = RayGPUExecutor(
            workers=self.all_workers, engine_config=self.engine_config
        )
        self.global_executor._run_workers(
            "init_worker_distributed_environment",
            self.engine_config,
            len(self.all_workers),
        )
        # init the static placement, and dispatch all the init RayGPUExecutor
        self.static_placement_init()

        for executor in self.all_executors:
            executor._run_workers(
                "init_static_env",
                world_size=len(executor.workers),
                ranks=executor.global_ranks,
            )
            executor._run_workers("init_instance_model")
            executor._run_workers("execute_model")

    @classmethod
    def from_engine_args(cls, serving_config: ServingConfig):
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = serving_config.engine_config
        parallel_config = engine_config.parallel_config

        # Initialize the cluster and specify the executor class.
        initialize_ray_cluster(
            parallel_config
        )  # ray.init, and set the placement_group field in parallel_config

        executor_class = RayGPUExecutor

        # Create the LLM engine.
        engine = cls(
            engine_config=engine_config,
            model_class=serving_config.model_class,
            executor_class=executor_class,
        )
        return engine

    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
        num_gpus = 1
        logger.debug(f"init_workers_ray's pid = {os.getpid()}")
        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        # self.driver_dummy_worker: RayWorkerHetudit = None
        # The remaining workers are the actual ray actors.

        # Create the workers.
        driver_ip = get_ip()
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )
            worker = ray.remote(
                num_cpus=2,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                **ray_remote_kwargs,
            )(RayWorkerHetudit).remote()

            worker_ip = ray.get(worker.get_node_ip.remote())
            if worker_ip == driver_ip and self.driver_dummy_worker is None:
                # If the worker is on the same node as the driver, we use it
                # as the resource holder for the driver process.
                self.driver_dummy_worker = worker

            self.all_workers.append(worker)

        if self.driver_dummy_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "GPU node."
            )

        # Get the set of GPU IDs used on each node.
        driver_node_id, driver_gpu_ids = ray.get(
            self.driver_dummy_worker.get_node_and_gpu_ids.remote()
        )
        worker_node_and_gpu_ids = ray.get(
            [worker.get_node_and_gpu_ids.remote() for worker in self.all_workers[:]]
        )

        node_workers = defaultdict(list)
        node_gpus = defaultdict(list)

        node_workers[driver_node_id].append(0)
        node_gpus[driver_node_id].extend(driver_gpu_ids)
        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids, start=0):
            node_workers[node_id].append(i)
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        # Set CUDA_VISIBLE_DEVICES for the driver and workers.
        set_cuda_visible_devices(
            node_gpus[driver_node_id]
        )  # Set the visible device for the main thread (driver_worker) to the gpu_id of all workers. For example, if 4 are used, then it would be 0, 1, 2, 3.
        for worker, (node_id, _) in zip(self.all_workers, worker_node_and_gpu_ids):
            worker.set_cuda_visible_devices.remote(
                node_gpus[node_id]
            )  # Each remote worker can see all the GPUs on this node, so in the future, numbering can start directly from 0.

        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port()
        )

        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        engine_config = copy.deepcopy(self.engine_config)
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)

        # Initialize the actual workers with the Worker class.
        for rank, (worker, (node_id, _)) in enumerate(
            zip(self.all_workers, worker_node_and_gpu_ids),
            start=0,
        ):
            local_rank = node_workers[node_id].index(rank)
            worker.init_worker.remote(
                lambda rank=rank, local_rank=local_rank: Worker(
                    engine_config,
                    local_rank,
                    rank,
                    distributed_init_method,
                )
            )

        # Initialize the driver worker with the Worker class.
        driver_rank = 0
        driver_local_rank = node_workers[driver_node_id].index(driver_rank)
        self.driver_worker = self.driver_worker = Worker(
            engine_config,
            driver_local_rank,
            driver_rank,
            distributed_init_method,
            is_driver_worker=True,
        )

    def add_request(self, request: Dict[str, Any]) -> None:
        """Add a request to the serving engine."""
        pass

    def execute(self) -> Dict[str, Any]:
        """Execute one diffusion process."""
        pass

    def search_best_static_placement(self):
        """
        Search the best static placement for the model, and return the best placement
        """
        return {
            "instance0": [0, 3],
            "instance1": [5, 6],
            "instance2": [1, 2],
            "instance3": [4, 7],
        }

    def static_placement_init(self):
        """
        Really do the best static placement, place the model;s para in corresponding GPU
        """
        instance_dict = self.search_best_static_placement()
        for name, gpu_ids in instance_dict.items():
            workers = [self.all_workers[i] for i in gpu_ids]
            self.all_executors.append(
                RayGPUExecutor(workers, self.engine_config, global_ranks=gpu_ids)
            )

    def reconfigure_executors(self, new_config: Dict[str, List[int]]):
        """
        Reconfigure executors while respecting worker states.
        """
        affected_executors = []
        for executor in self.all_executors:
            if any(
                worker in new_config["new_instance"] for worker in executor.global_ranks
            ):
                affected_executors.append(executor)

        # Gracefully shutdown affected executors
        for executor in affected_executors:
            executor.shutdown(prepare_for_reuse=True)
            self.all_executors.remove(executor)

        # Create new executor
        new_workers = []
        for worker_id in new_config["new_instance"]:
            worker = self.all_workers[worker_id]
            if ray.get(worker.get_state.remote()) == "busy":
                raise RuntimeError(
                    f"Worker {worker_id} is busy and cannot be reconfigured."
                )
            ray.get(worker.set_state.remote("ready"))
            new_workers.append(worker)

        new_executor = RayGPUExecutor(
            new_workers, self.engine_config, global_ranks=new_config["new_instance"]
        )
        self.all_executors.append(new_executor)
