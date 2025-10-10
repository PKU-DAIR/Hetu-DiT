from .executor_base import ExecutorBase, ExecutorAsyncBase
from hetu_dit.engine.ray_utils import RayWorkerHetudit, ray
from hetu_dit.logger import init_logger
from hetu_dit.config.config import EngineConfig
from typing import Any, Optional, List, Dict
import asyncio

logger = init_logger(__name__)


class GPUExecutor(ExecutorBase):
    def __init__(self, engine_config: EngineConfig):
        super().__init__(engine_config)
        self.engine_config = engine_config

    def _init_executor(self):
        pass


class RayGPUExecutor(GPUExecutor):
    """
    All the method is called by _run_workers
    """

    def __init__(
        self,
        workers: List[RayWorkerHetudit],
        engine_config: EngineConfig,
        global_ranks: List[int] = [],
        name="RayGPUExecutor",
    ):
        super().__init__(engine_config)
        self.workers = []
        # self.global_ranks = sorted(global_ranks)
        self.global_ranks = []
        self.local_ranks = []
        self.executor_state = "ready"
        self.name = name
        self.machine_id = engine_config.machine_id
        for worker in workers:
            self.global_ranks.append(ray.get(worker.get_global_rank.remote()))
            self.local_ranks.append(ray.get(worker.get_local_rank.remote()))
            if ray.get(worker.get_state.remote()) != "busy":
                ray.get(worker.set_state.remote("ready"))
                self.workers.append(worker)
            else:
                raise ValueError(
                    f"Worker {worker} state is {ray.get(worker.get_state.remote())} and cannot be added to the executor."
                )
        self.global_ranks = sorted(self.global_ranks)
        self.local_ranks = sorted(self.local_ranks)
        logger.info(
            f"in raygpuexecutor, {self.name}, global_ranks = {self.global_ranks}, local_ranks = {self.local_ranks}"
        )

    def _init_executor(self):
        return super()._init_executor()

    def execute_model(
        self,
    ):
        all_outputs = self._run_workers("execute_model")

        # Only the driver worker returns the sampling results.
        output = all_outputs[0]
        return output

    def _run_workers(
        self,
        method: str,
        *args,
        driver_args: Optional[List[Any]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrent_workers: Optional[int] = None,
        use_ray_compiled_dag: bool = False,
        **kwargs,
    ) -> Any:
        """
        Runs the given method on all workers with state management.
        """
        if max_concurrent_workers:
            raise NotImplementedError("max_concurrent_workers is not supported yet.")

        # Check that all workers are in the "ready" state
        for worker in self.workers:
            if ray.get(worker.get_state.remote()) != "ready":
                raise RuntimeError(
                    f"Worker {worker} is not ready to execute the method: {method}"
                )

        # Set all workers to "busy"
        for worker in self.workers:
            ray.get(worker.set_state.remote("busy"))
        self.executor_state = "busy"
        # Start the ray workers
        try:
            logger.debug(f"len(self.workers) = {len(self.workers)}")
            ray_worker_outputs = [
                worker.execute_method.remote(method, *args, **kwargs)
                for worker in self.workers
            ]

            # Get the results of the ray workers
            if self.workers:
                ray_worker_outputs = ray.get(ray_worker_outputs)

            return ray_worker_outputs
        except Exception as e:
            # Log or handle exceptions if necessary
            raise RuntimeError(
                f"Error during execution of method {method}: {str(e)}"
            ) from e
        finally:
            # Reset all workers to "ready" after execution, even if an exception occurs
            for worker in self.workers:
                ray.get(worker.set_state.remote("ready"))
            self.execute_state = "ready"

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        self._check_if_any_actor_is_dead()

    def _check_if_any_actor_is_dead(self):
        if not self.workers:
            return

        dead_actors = []
        for actor in self.workers:
            actor_state = ray.state.actors(actor._ray_actor_id.hex())  # pylint: disable=protected-access
            if actor_state["State"] == "DEAD":
                dead_actors.append(actor)
        if dead_actors:
            raise RuntimeError(
                f"At least one Worker is dead. Dead Workers: {dead_actors}. "
            )

    def shutdown(self, prepare_for_reuse: bool = True):
        """
        Gracefully shuts down the executor without killing workers.
        Workers can optionally be prepared for reuse.
        """
        if not self.workers:
            logger.info("No workers to shut down.")
            return

        logger.info("Dissolving executor and preparing workers for reuse...")
        if prepare_for_reuse:
            for worker in self.workers:
                try:
                    # Reset the worker's state if a reset method is available
                    logger.debug(f"in shutdown before set {worker} to idle")
                    ray.get(worker.set_state.remote("idle"))
                    logger.debug(
                        f"after shutdown, {worker} state is {ray.get(worker.get_state.remote())}"
                    )
                except Exception as e:
                    logger.error(f"Failed to reset worker {worker}: {e}")

        # Clear internal references
        self.workers = []
        self.global_ranks = []
        self.executor_state = "ready"
        logger.info("Executor shut down successfully, workers are ready for reuse.")

    def set_state(self, state: str):
        self.executor_state = state

    def get_state(self):
        return self.executor_state

    def get_machine_id(self):
        return self.machine_id


class RayGPUExecutorAsync(RayGPUExecutor, ExecutorAsyncBase):
    async def _run_workers_async(
        self,
        method: str,
        *args,
        driver_args: Optional[List[Any]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        logger.debug(
            f"in _run_workers_async executor is {self}, the len of workers is {len(self.workers)}, workers is {self.workers}"
        )
        # Set all workers to "busy"
        for worker in self.workers:
            ray.get(worker.set_state.remote("busy"))
        self.executor_state = "busy"
        logger.debug(
            f"in _run_workers_async executor is {self}, set executor_state to busy"
        )

        async def _task_runner():
            coros = [
                worker.execute_method.remote(method, *args, **kwargs)
                for worker in self.workers
            ]
            return await asyncio.gather(*coros)

        # create task
        task = asyncio.create_task(_task_runner())

        # done callback to ensure restore state immediately
        def _on_done(fut: asyncio.Future):
            try:
                fut.result()
            finally:
                for worker in self.workers:
                    ray.get(worker.set_state.remote("ready"))
                logger.debug(
                    f"in _run_workers_async executor is {self}, set executor_state to ready"
                )
                self.executor_state = "ready"

        task.add_done_callback(_on_done)

        return task

    async def execute_model_async(
        self,
    ):
        all_outputs = await self._run_workers_async("execute_model")

        # Only the driver worker returns the sampling results.
        output = all_outputs[0]
        return output

    async def check_health_async(self) -> None:
        """Raises an error if engine is unhealthy."""
        self._check_if_any_actor_is_dead()

    async def detect_worker_meta(self) -> List[Dict[str, Any]]:
        """Detect the state and metadata of all workers in parallel."""

        async def _task():
            coros = [worker.get_states.remote() for worker in self.workers]
            return await asyncio.gather(*coros)

        # create task
        task = asyncio.create_task(_task())
        logger.debug("Detecting worker metadata...")
        re = await task
        return re
