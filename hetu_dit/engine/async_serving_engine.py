import asyncio
import copy
import os
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type, Coroutine

import ray
from diffusers import StableDiffusion3Pipeline
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from hetu_dit.config.config import (
    EngineConfig,
    InputConfig,
    ParallelConfig,
    ServingConfig,
)
from hetu_dit.core.request_manager.efficient_ilp import select_tasks
from hetu_dit.core.request_manager.request_manager import RequestManager
from hetu_dit.executor.executor_base import ExecutorBase
from hetu_dit.executor.gpu_executor import RayGPUExecutorAsync
from hetu_dit.logger import init_logger
from hetu_dit.profiler import global_profiler
from hetu_dit.utils import (
    Counter,
    create_new_config,
    estimate_ddl,
    get_distributed_init_method,
    get_ip,
    make_profile_key,
)
from hetu_dit.worker.worker import Worker

from .monitor import WorkerMonitor
from .ray_utils import RayWorkerHetudit, initialize_ray_cluster

logger = init_logger(__name__)


def find_least_busy_machine(
    detect_meta: dict, constrained_worker_ids: list[int] = None
) -> int:
    from collections import defaultdict

    machine_workers = defaultdict(list)

    # allocate workers to their respective machines
    for worker_id, info in detect_meta.items():
        num = int(worker_id.replace("Worker", ""))

        if constrained_worker_ids is not None and num not in constrained_worker_ids:
            continue

        machine_id = num // 8
        machine_workers[machine_id].append(info["state"])

    if not machine_workers:
        raise ValueError("No valid workers found in constrained_worker_ids.")

    # count busy workers in each machine
    busy_counts = {}
    for machine_id, states in machine_workers.items():
        busy_count = sum(1 for state in states if state == "busy")
        busy_counts[machine_id] = busy_count

    # find the machine_id with the least busy workers, if tied choose the one with the largest id
    logger.debug(f"busy_counts: {busy_counts}")
    min_busy = min(busy_counts.values())
    candidate_machines = [
        mid for mid, count in busy_counts.items() if count == min_busy
    ]
    return max(candidate_machines)


def generate_parallel_config_name(
    parallel_config: "ParallelConfig", machine_id: int = 0, stage: str = "diffusion"
) -> str:
    """
    Generate a formatted name string based on a ParallelConfig instance.
    Format: tp_<tp_degree>_ulysses_<ulysses_degree>_ring_<ring_degree>_pp_<pp_degree>_cfg_<cfg_degree>_dp_<dp_degree>
    """
    return (
        f"tp_{parallel_config.tp_degree}_"
        f"ulysses_{parallel_config.ulysses_degree}_"
        f"ring_{parallel_config.ring_degree}_"
        f"pp_{parallel_config.pp_degree}_"
        f"cfg_{parallel_config.cfg_degree}_"
        f"dp_{parallel_config.dp_degree}_"
        f"text_encoder_tp_{parallel_config.text_encoder_tp_config}_"
        f"stage_{stage}_"
        f"machine_{machine_id}"
    )


def cal_needed_workers(new_engine_config: EngineConfig) -> int:
    pconf = new_engine_config.parallel_config
    dp_degree = pconf.dp_config.dp_degree
    cfg_degree = pconf.dp_config.cfg_degree
    ulysses_degree = pconf.sp_config.ulysses_degree
    ring_degree = pconf.sp_config.ring_degree
    tp_degree = pconf.tp_config.tp_degree
    pp_degree = pconf.pp_config.pp_degree

    return dp_degree * cfg_degree * ulysses_degree * ring_degree * tp_degree * pp_degree


def _log_executor_worker_states(
    executor: "RayGPUExecutorAsync", task_id: Optional[str], stage: str
) -> None:
    """Log the current state of each worker inside an executor for debugging."""
    for worker in executor.workers:
        worker_state = ray.get(worker.get_state.remote())
        logger.debug(
            "task_id %s | stage %s | worker %s state: %s",
            task_id,
            stage,
            worker,
            worker_state,
        )


def _set_executor_state(
    executor: "RayGPUExecutorAsync",
    state: str,
    executor_states: Dict["RayGPUExecutorAsync", str],
) -> None:
    """Update executor state locally and remotely."""
    executor_states[executor] = state
    executor.set_state(state)


def _set_remote_workers_state(executor: "RayGPUExecutorAsync", state: str) -> None:
    """Set remote state for all workers within the executor."""
    for worker in executor.workers:
        ray.get(worker.set_state.remote(state))


class AsyncServingEngine:
    """Asynchronous serving engine that manages executor pools and worker scheduling.

    The engine maintains executor state locally, coordinates worker reuse, and
    exposes utility coroutines for different serving flavors (standard,
    disaggregated, downscale VAE, and fixed worker scheduling). The overall
    behavior matches the synchronous serving engine but introduces waiting
    queues and background tasks to avoid blocking on resource acquisition.
    """

    def __init__(
        self,
        engine_config: "EngineConfig",
        model_class: Type,
        executor_class: Type["ExecutorBase"],
        search_mode: str = "random",
        model_class_name: str = "",
        use_disaggregated_encode_decode: bool = False,
        stage_level: bool = False,
        encode_worker_ids: Optional[List[int]] = None,
        decode_worker_ids: Optional[List[int]] = None,
    ) -> None:
        self.engine_config = engine_config
        self.model_config = engine_config.model_config
        self.runtime_config = engine_config.runtime_config
        self.parallel_config = engine_config.parallel_config
        self.model_class = model_class
        self.model_class_name = model_class_name
        self.executor_class = executor_class
        self.reqs_counter = Counter()

        self.driver_dummy_worker = None

        # manage all workers
        self.all_workers: List["RayWorkerHetudit"] = []

        # Background tasks management to avoid block
        self.background_tasks = set()

        # executors management
        self.executors_dict: Dict[str, List["RayGPUExecutorAsync"]] = defaultdict(list)
        self.executor_config_counters: Dict[str, int] = defaultdict(int)

        self.scheduler = RequestManager()

        # Waiting for the existing executors to become ready in the future queue (the second situation)
        self.waiting_tasks: Dict[str, List[asyncio.Future]] = defaultdict(list)

        # Waiting for the future queue of reconfiguration to complete (third situation)
        self.waiting_reconfigure_tasks: Dict[str, List[asyncio.Future]] = defaultdict(
            list
        )

        # Future queue waiting for worker resources (used in search_reconfigure_worker)
        self.waiting_worker_tasks: List[asyncio.Future] = []

        # Local maintenance of executor state
        self.executor_states: Dict[RayGPUExecutorAsync, str] = {}

        # Using asynchronous locks and condition variables to ensure concurrency safety.
        self._executor_lock = asyncio.Lock()
        self._executor_condition = asyncio.Condition(self._executor_lock)

        # Worker task queues for fixed search mode
        self.worker_task_queues: Dict[
            int,
            List[Tuple[str, List[int], EngineConfig, asyncio.Future, "InputConfig"]],
        ] = defaultdict(list)

        # Flag to control the queue scanning task
        self.should_scan_queues = True

        # Search mode ("random" or "fixed")
        self.search_mode = search_mode

        # used for add_task_greedy_ilp
        self.request_queue: List[Dict] = []

        self.use_disaggregated_encode_decode = use_disaggregated_encode_decode
        self.stage_level = stage_level

        self.encode_worker_ids = encode_worker_ids
        self.decode_worker_ids = decode_worker_ids
        if self.use_disaggregated_encode_decode:
            if self.encode_worker_ids is not None and self.decode_worker_ids is None:
                self.decode_worker_ids = self.encode_worker_ids
            elif self.encode_worker_ids is None and self.decode_worker_ids is not None:
                self.encode_worker_ids = self.decode_worker_ids
            elif self.encode_worker_ids is None and self.decode_worker_ids is None:
                self.encode_worker_ids = self.decode_worker_ids = list(range(8))
        # init distributed env
        self._init_workers_ray(self.parallel_config.placement_group)

    async def _assign_machine(
        self,
        engine_config: EngineConfig,
        machine_id: Optional[int],
        constrained_worker_ids: Optional[List[int]] = None,
    ) -> None:
        """Assign a machine to the provided engine configuration."""
        if machine_id is not None:
            engine_config.machine_id = machine_id
            return

        await self.monitor.refresh()
        engine_config.machine_id = find_least_busy_machine(
            self.detect_meta(),
            constrained_worker_ids,
        )

    async def _mark_executor_busy(
        self,
        config_name: str,
        executor: "RayGPUExecutorAsync",
        *,
        update_remote: bool = False,
    ) -> None:
        """Log executor states and mark the selected executor as busy."""
        async with self._executor_condition:
            for candidate in self.executors_dict[config_name]:
                logger.debug(
                    "executor %s state: %s",
                    candidate.name,
                    self.executor_states.get(candidate),
                )
            _set_executor_state(executor, "busy", self.executor_states)
            if update_remote:
                _set_remote_workers_state(executor, "busy")
            logger.debug("running executor %s", executor.name)

    def _schedule_background_task(
        self, coro: Coroutine[Any, Any, Any], *, description: str
    ) -> asyncio.Task:
        """Create and track a background task with standardized logging."""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        logger.debug(
            "%s scheduled; loop tasks=%d, background tasks=%d",
            description,
            len(asyncio.all_tasks()),
            len(self.background_tasks),
        )
        return task

    def post_init(self):
        self.global_executor = RayGPUExecutorAsync(
            workers=self.all_workers,
            engine_config=self.engine_config,
            global_ranks=list(range(len(self.all_workers))),
        )
        self.global_executor._run_workers(
            "init_singleton_cpu_model",
            self.engine_config,
            self.model_class,
            self.engine_config.runtime_config.use_parallel_text_encoder,
        )
        self.global_executor._run_workers(
            "init_worker_distributed_environment",
            self.engine_config,
            len(self.all_workers),
        )
        self.static_placement_init()
        self.categorize_workers = {}
        if self.use_disaggregated_encode_decode:
            self.diffusion_worker_ids = [
                i
                for i in range(len(self.all_workers))
                if i not in self.encode_worker_ids + self.decode_worker_ids
            ]
            self.categorize_workers["encode"] = [
                self.all_workers[i] for i in self.encode_worker_ids
            ]
            self.categorize_workers["decode"] = [
                self.all_workers[i] for i in self.decode_worker_ids
            ]
            self.categorize_workers["diffusion"] = [
                self.all_workers[i] for i in self.diffusion_worker_ids
            ]
            self.categorize_workers.update(
                {
                    f"machine_{i}": self.all_workers[i * 8 : (i + 1) * 8]
                    for i in range(len(self.all_workers) // 8)
                }
            )
            logger.debug(f"categorize_workers is {self.categorize_workers}")
            logger.debug(f"encode_worker_ids is {self.encode_worker_ids}")
            logger.debug(f"decode_worker_ids is {self.decode_worker_ids}")
            logger.debug(f"diffusion_worker_ids is {self.diffusion_worker_ids}")
        else:
            self.encode_worker_ids = list(range(len(self.all_workers)))
            self.decode_worker_ids = list(range(len(self.all_workers)))
            self.diffusion_worker_ids = list(range(len(self.all_workers)))
            self.categorize_workers["diffusion"] = self.all_workers
            self.categorize_workers["encode"] = self.all_workers
            self.categorize_workers["decode"] = self.all_workers

    def detect_meta(self) -> Dict:
        return self.monitor.get_worker_states()

    async def init_monitor(self):
        self.monitor = WorkerMonitor(self.global_executor)
        self._schedule_background_task(
            self.monitor.start_monitoring(),
            description="monitor.start_monitoring",
        )

    async def init_all_executors(self):
        tasks = []
        async with self._executor_lock:
            for config_name, executors in self.executors_dict.items():
                for executor in executors:
                    tasks.append(self.init_single_executor(executor))
        await asyncio.gather(*tasks)

    async def init_single_executor(self, executor):
        task1 = await executor._run_workers_async(
            "init_static_env",
            world_size=len(executor.workers),
            ranks=executor.global_ranks,
            engine_config=executor.engine_config,
        )
        await task1
        task2 = await executor._run_workers_async(
            "init_instance_model",
            engine_config=executor.engine_config,
            model_class=self.model_class,
        )
        await task2

    async def run_single_executor(
        self,
        executor,
        input_config=None,
        model_class=StableDiffusion3Pipeline,
        task_id: str = None,
    ):
        try:
            task1 = await executor._run_workers_async(
                "execute_model",
                engine_config=executor.engine_config,
                input_config=input_config,
                model_class=model_class,
                task_id=task_id,
            )
            results = await task1
            if "Model_Profiler" in input_config.prompt:
                self.model_profiler.save_data(tag=task_id, results=results)
            else:
                async with self._executor_condition:
                    global_profiler.end(
                        results=results, tag=task_id, ranks=executor.global_ranks
                    )
        except Exception as e:
            logger.error(f"Error running executor {executor}: {e}")
            traceback.print_exc()
        finally:
            await self._notify_executor_ready_by_executor(executor, task_id)
            task = asyncio.current_task()
            self.background_tasks.discard(task)

    async def run_single_executor_disaggregated(
        self,
        executor,
        text_encoder_executor,
        vae_decoder_executor,
        input_config=None,
        model_class=StableDiffusion3Pipeline,
        task_id: str = None,
    ):
        try:
            _log_executor_worker_states(text_encoder_executor, task_id, "encode-stage")
            task1 = await text_encoder_executor._run_workers_async(
                "execute_encode_stage",
                engine_config=text_encoder_executor.engine_config,
                input_config=input_config,
                model_class=model_class,
                task_id=task_id,
            )

            results1 = await task1
            logger.debug(f"task id is {task_id}, results1 is {results1}")
            encode_stage_results = results1[0]
            await self._notify_executor_ready_by_executor(
                text_encoder_executor, task_id
            )

            _log_executor_worker_states(executor, task_id, "diffusion-stage")
            logger.debug(
                f"task_id is {task_id}, id executor is {id(executor)}, id executor.engine_config is {id(executor.engine_config)}, executor.engine_config.diffusion_stage_ranks is {executor.engine_config.diffusion_stage_ranks}"
            )
            task2 = await executor._run_workers_async(
                "execute_diffusion_stage",
                encode_stage_results,
                engine_config=executor.engine_config,
                input_config=input_config,
                model_class=model_class,
                task_id=task_id,
            )

            results2 = await task2
            logger.debug(f"task id is {task_id}, results2 is {results2}")
            results = None
            latents = None
            for res, lat in results2:
                if lat is not None:
                    results = res
                    latents = lat
                    break

            await self._notify_executor_ready_by_executor(executor, task_id)
            _log_executor_worker_states(vae_decoder_executor, task_id, "decode-stage")
            task3 = await vae_decoder_executor._run_workers_async(
                "execute_decode_stage",
                latents,
                engine_config=executor.engine_config,
                input_config=input_config,
                model_class=model_class,
                task_id=task_id,
            )

            results = await task3
            logger.debug(f"task id is {task_id}, results3 is {results}")
            logger.debug("in run_single_executor_disaggregated, finished task3")

            if input_config.prompt == "Model_Profiler":
                self.model_profiler.save_data(tag=task_id, results=results)
            else:
                global_profiler.end(
                    results=results, tag=task_id, ranks=executor.global_ranks
                )
            await self._notify_executor_ready_by_executor(vae_decoder_executor, task_id)
        except Exception as e:
            logger.error(f"Error running executor {executor}: {e}")
            traceback.print_exc()
        finally:
            task = asyncio.current_task()
            self.background_tasks.discard(task)

    async def run_single_executor_downscale_vae(
        self,
        executor,
        executor_config_name,
        input_config=None,
        model_class=StableDiffusion3Pipeline,
        task_id: str = None,
    ):
        try:
            encode_diffusion_stage_ranks = copy.deepcopy(executor.global_ranks)
            _log_executor_worker_states(executor, task_id, "encode-diffusion-stage")
            logger.debug(
                f"task_id is {task_id}, id executor is {id(executor)}, id executor.engine_config is {id(executor.engine_config)}, executor.engine_config.diffusion_stage_ranks is {executor.engine_config.diffusion_stage_ranks}"
            )
            task2 = await executor._run_workers_async(
                "execute_encode_diffusion_stage",
                engine_config=executor.engine_config,
                input_config=input_config,
                model_class=model_class,
                task_id=task_id,
            )

            results2 = await task2
            logger.debug(f"task id is {task_id}, results2 is {results2}")
            results = None
            latents = None
            all_results = [result for result, lat in results2]
            for res, lat in results2:
                if lat is not None:
                    results = res
                    latents = lat
                    break

            await self._notify_executor_ready_by_executor(executor, task_id)
            if input_config.prompt == "Model_Profiler":
                logger.debug(
                    f"task_id is {task_id}, begin to save data in run_single_executor_downscale_vae"
                )
                self.model_profiler.save_data(tag=task_id, results=all_results)
            else:
                logger.debug(
                    f"task_id is {task_id}, begin to save data in run_single_executor_downscale_vae"
                )
                global_profiler.end(
                    results=all_results, tag=task_id, ranks=executor.global_ranks
                )

            vae_decoder_config, _ = create_new_config(
                self.engine_config,
                ulysses_degree=1,
                ring_degree=1,
                tensor_parallel_degree=1,
                pipefusion_parallel_degree=1,
                use_parallel_text_encoder=self.engine_config.runtime_config.use_parallel_text_encoder,
                text_encoder_tensor_parallel_degree=1,
                is_serving=True,
                machine_id=executor.engine_config.machine_id,
            )
            vae_decoder_parallel_config_name = generate_parallel_config_name(
                vae_decoder_config.parallel_config,
                executor.engine_config.machine_id,
                stage="decode",
            )
            vae_decoder_executor = await self.get_ready_executor_or_reconfigure(
                vae_decoder_parallel_config_name,
                vae_decoder_config,
                task_id=f"{task_id}_vae_decoder",
            )
            logger.debug(
                f"task_id is {task_id}_vae_decoder, vae_decoder_executor is {vae_decoder_executor}, global_ranks is {vae_decoder_executor.global_ranks}, workers is {vae_decoder_executor.workers}"
            )
            await self._mark_executor_busy(
                vae_decoder_parallel_config_name, vae_decoder_executor
            )

            self._schedule_background_task(
                self.run_vae_executor_downscale_vae(
                    latents,
                    vae_decoder_executor,
                    input_config,
                    self.model_class,
                    task_id=task_id,
                ),
                description=f"run_vae_executor_downscale_vae[{task_id}]",
            )
        except Exception as e:
            logger.error(f"Error running executor {executor}: {e}")
            traceback.print_exc()
        finally:
            task = asyncio.current_task()
            self.background_tasks.discard(task)

    async def run_vae_executor_downscale_vae(
        self,
        latents,
        vae_decoder_executor,
        input_config=None,
        model_class=StableDiffusion3Pipeline,
        task_id: str = None,
    ):
        try:
            _log_executor_worker_states(
                vae_decoder_executor, task_id, "vae-decode-stage"
            )
            task3 = await vae_decoder_executor._run_workers_async(
                "execute_decode_stage",
                latents,
                engine_config=vae_decoder_executor.engine_config,
                input_config=input_config,
                model_class=model_class,
                task_id=task_id,
            )

            results = await task3
            logger.debug(f"task id is {task_id}, results3 is {results}")
            logger.debug("in run_single_executor_disaggregated, finished task3")

            await self._notify_executor_ready_by_executor(vae_decoder_executor, task_id)
        except Exception as e:
            logger.error(f"Error running executor {vae_decoder_executor}: {e}")
            traceback.print_exc()
        finally:
            task = asyncio.current_task()
            self.background_tasks.discard(task)

    async def run_task(
        self,
        input_config: "InputConfig" = None,
        engine_config: "EngineConfig" = None,
        task_id: str = None,
        machine_id=None,
    ):
        """
        According to the requested engine_config, select an idle executor to execute the task
        (the logic for the 3 cases is all handled in get_ready_executor_or_reconfigure).
        If there is no idle executor, wait; if there is no executor corresponding to the configuration,
        reconfigure and create one, and then wait.
        """
        engine_config = engine_config or self.engine_config
        logger.debug(
            "task_id %s entered run_task, ulysses_degree=%s",
            task_id,
            engine_config.parallel_config.ulysses_degree,
        )
        await self._assign_machine(engine_config, machine_id)
        parallel_config = engine_config.parallel_config
        logger.debug(
            "task_id %s before generate_parallel_config_name, ulysses_degree=%s",
            task_id,
            engine_config.parallel_config.ulysses_degree,
        )
        parallel_config_name = generate_parallel_config_name(
            parallel_config, engine_config.machine_id
        )
        logger.debug("parallel_config_name is %s", parallel_config_name)
        executor = await self.get_ready_executor_or_reconfigure(
            parallel_config_name, engine_config, task_id
        )

        await self._mark_executor_busy(parallel_config_name, executor)
        executor.engine_config.diffusion_stage_ranks = executor.global_ranks
        self._schedule_background_task(
            self.run_single_executor(
                executor, input_config, self.model_class, task_id=task_id
            ),
            description=f"run_task[{task_id}]",
        )

    async def run_task_batch(
        self, input_configs=None, engine_configs=None, task_ids=None
    ):
        """
        According to the requested engine_config, select an idle executor to execute the task
        (the logic for the 3 cases is all handled in get_ready_executor_or_reconfigure).
        If there is no idle executor, wait; if there is no executor corresponding to the configuration,
        reconfigure and create one, and then wait.
        """
        for input_config, engine_config, task_id in zip(
            input_configs, engine_configs, task_ids
        ):
            engine_config = engine_config or self.engine_config
            await self._assign_machine(engine_config, None)
            parallel_config = engine_config.parallel_config
            parallel_config_name = generate_parallel_config_name(
                parallel_config, engine_config.machine_id
            )
            logger.debug("parallel_config_name is %s", parallel_config_name)
            executor = await self.get_ready_executor_or_reconfigure(
                parallel_config_name, engine_config, task_id
            )

            await self._mark_executor_busy(
                parallel_config_name, executor, update_remote=True
            )
            executor.engine_config.diffusion_stage_ranks = executor.global_ranks
            self._schedule_background_task(
                self.run_single_executor(
                    executor, input_config, self.model_class, task_id=task_id
                ),
                description=f"run_task_batch[{task_id}]",
            )

    async def run_task_disaggregated(
        self,
        input_config: "InputConfig" = None,
        engine_config: "EngineConfig" = None,
        task_id: str = None,
    ):
        """
        According to the requested engine_config, select an idle executor to execute the task
        (the logic for the 3 cases is all handled in get_ready_executor_or_reconfigure).
        If there is no idle executor, wait; if there is no executor corresponding to the configuration,
        reconfigure and create one, and then wait.
        """
        logger.debug("enter run_task_disaggregated")
        engine_config = engine_config or self.engine_config
        await self._assign_machine(engine_config, None, self.diffusion_worker_ids)
        parallel_config = engine_config.parallel_config
        parallel_config_name = generate_parallel_config_name(
            parallel_config, engine_config.machine_id
        )
        logger.debug("parallel_config_name is %s", parallel_config_name)
        executor = await self.get_ready_executor_or_reconfigure_disaggregated(
            parallel_config_name,
            engine_config,
            constraint_worker_ids=self.diffusion_worker_ids,
            tag="diffusion",
            task_id=task_id,
        )
        logger.debug(
            f"task_id is {task_id}, executor is {executor}, global_ranks is {executor.global_ranks}, workers is {executor.workers}"
        )
        await self._mark_executor_busy(parallel_config_name, executor)

        text_encoder_config, _ = create_new_config(
            self.engine_config,
            ulysses_degree=1,
            ring_degree=1,
            tensor_parallel_degree=1,
            pipefusion_parallel_degree=1,
            use_parallel_text_encoder=self.engine_config.runtime_config.use_parallel_text_encoder,
            text_encoder_tensor_parallel_degree=1,
            is_serving=True,
            machine_id=0,
        )
        text_encoder_parallel_config_name = generate_parallel_config_name(
            text_encoder_config.parallel_config, 0
        )
        text_encoder_executor = (
            await self.get_ready_executor_or_reconfigure_disaggregated(
                text_encoder_parallel_config_name,
                text_encoder_config,
                constraint_worker_ids=self.encode_worker_ids,
                tag="encode",
                task_id=f"{task_id}_text_encoder",
            )
        )
        logger.debug(
            f"task_id is {task_id}, text_encoder_executor is {text_encoder_executor}, global_ranks is {text_encoder_executor.global_ranks}, workers is {text_encoder_executor.workers}"
        )
        await self._mark_executor_busy(
            text_encoder_parallel_config_name, text_encoder_executor
        )

        vae_decoder_config, _ = create_new_config(
            self.engine_config,
            ulysses_degree=1,
            ring_degree=1,
            tensor_parallel_degree=1,
            pipefusion_parallel_degree=1,
            use_parallel_text_encoder=self.engine_config.runtime_config.use_parallel_text_encoder,
            text_encoder_tensor_parallel_degree=1,
            is_serving=True,
            machine_id=0,
        )
        vae_decoder_parallel_config_name = generate_parallel_config_name(
            vae_decoder_config.parallel_config, 0
        )
        vae_decoder_executor = (
            await self.get_ready_executor_or_reconfigure_disaggregated(
                vae_decoder_parallel_config_name,
                vae_decoder_config,
                constraint_worker_ids=self.decode_worker_ids,
                tag="decode",
                task_id=f"{task_id}_vae_decoder",
            )
        )
        logger.debug(
            f"task_id is {task_id}, vae_decoder_executor is {vae_decoder_executor}, global_ranks is {vae_decoder_executor.global_ranks}, workers is {vae_decoder_executor.workers}"
        )
        await self._mark_executor_busy(
            vae_decoder_parallel_config_name, vae_decoder_executor
        )

        executor.engine_config.diffusion_stage_ranks = executor.global_ranks
        executor.engine_config.encode_stage_rank = text_encoder_executor.global_ranks
        executor.engine_config.decode_stage_ranks = vae_decoder_executor.global_ranks
        text_encoder_executor.engine_config.diffusion_stage_ranks = (
            executor.global_ranks
        )
        text_encoder_executor.engine_config.encode_stage_rank = (
            text_encoder_executor.global_ranks
        )
        text_encoder_executor.engine_config.decode_stage_ranks = (
            vae_decoder_executor.global_ranks
        )
        vae_decoder_executor.engine_config.diffusion_stage_ranks = executor.global_ranks
        vae_decoder_executor.engine_config.encode_stage_rank = (
            text_encoder_executor.global_ranks
        )
        vae_decoder_executor.engine_config.decode_stage_ranks = (
            vae_decoder_executor.global_ranks
        )
        executor.engine_config = copy.deepcopy(executor.engine_config)
        text_encoder_executor.engine_config = copy.deepcopy(
            text_encoder_executor.engine_config
        )
        vae_decoder_executor.engine_config = copy.deepcopy(
            vae_decoder_executor.engine_config
        )
        logger.debug(
            f"task_id is {task_id}, in run_task_disaggregated, executor is {executor}, text_encoder_executor is {text_encoder_executor}, vae_decoder_executor is {vae_decoder_executor}"
        )
        logger.debug(
            f"task_id is {task_id}, in run_task_disaggregated, executor.workers is {executor.workers}, text_encoder_executor.workers is {text_encoder_executor.workers}, vae_decoder_executor.workers is {vae_decoder_executor.workers}"
        )
        logger.debug(
            f"task_id is {task_id}, in run_task_disaggregated, executor.global_ranks is {executor.global_ranks}, text_encoder_executor.global_ranks is {text_encoder_executor.global_ranks}, vae_decoder_executor.global_ranks is {vae_decoder_executor.global_ranks}"
        )
        logger.debug(
            f"task_id is {task_id}, in run_task_disaggregated, executor.name is {executor.name}, text_encoder_executor.name is {text_encoder_executor.name}, vae_decoder_executor.name is {vae_decoder_executor.name}"
        )
        logger.debug(
            f"task_id is {task_id}, in run_task_disaggregated, id executor is {id(executor)}, id executor.engine_config is {id(executor.engine_config)}, executor.engine_config.diffusion_stage_ranks is {executor.engine_config.diffusion_stage_ranks}, executor.engine_config.encode_stage_rank is {executor.engine_config.encode_stage_rank}, executor.engine_config.decode_stage_ranks is {executor.engine_config.decode_stage_ranks}"
        )
        self._schedule_background_task(
            self.run_single_executor_disaggregated(
                executor,
                text_encoder_executor,
                vae_decoder_executor,
                input_config,
                self.model_class,
                task_id=task_id,
            ),
            description=f"run_task_disaggregated[{task_id}]",
        )

    async def run_task_downscale_vae(
        self,
        input_config: "InputConfig" = None,
        engine_config: "EngineConfig" = None,
        task_id: str = None,
        machine_id=None,
    ):
        """
        According to the requested engine_config, select an idle executor to execute the task
        (the logic for the 3 cases is all handled in get_ready_executor_or_reconfigure).
        If there is no idle executor, wait; if there is no executor corresponding to the configuration,
        reconfigure and create one, and then wait.
        """
        engine_config = engine_config or self.engine_config
        await self._assign_machine(engine_config, machine_id)
        logger.debug(
            "enter run_task_downscale_vae, found machine_id is %s",
            engine_config.machine_id,
        )
        parallel_config = engine_config.parallel_config
        parallel_config_name = generate_parallel_config_name(
            parallel_config, engine_config.machine_id
        )
        logger.debug("parallel_config_name is %s", parallel_config_name)
        executor = await self.get_ready_executor_or_reconfigure(
            parallel_config_name, engine_config, task_id=task_id
        )
        logger.debug(
            f"task_id is {task_id}, executor is {executor}, global_ranks is {executor.global_ranks}, workers is {executor.workers}"
        )
        await self._mark_executor_busy(parallel_config_name, executor)

        executor.engine_config.diffusion_stage_ranks = executor.global_ranks
        executor.engine_config = copy.deepcopy(executor.engine_config)

        self._schedule_background_task(
            self.run_single_executor_downscale_vae(
                executor,
                parallel_config_name,
                input_config,
                self.model_class,
                task_id=task_id,
            ),
            description=f"run_task_downscale_vae[{task_id}]",
        )

    async def add_task(
        self,
        worker_ids: List[int],
        input_config: "InputConfig" = None,
        engine_config: "EngineConfig" = None,
        task_id: str = None,
    ):
        """
        Add a task to the worker queues for fixed search mode.
        This method places the task in the queues of all the specified workers.

        Args:
            worker_ids: List of worker IDs to use
            input_config: Input configuration for the model
            engine_config: Engine configuration

        Returns:
            Future that will be resolved when the task is executed
        """
        if engine_config is None:
            engine_config = self.engine_config

        if not worker_ids:
            raise ValueError("Worker IDs must be specified for fixed search mode")

        parallel_config = engine_config.parallel_config
        parallel_config_name = generate_parallel_config_name(parallel_config)
        logger.debug(
            f"Adding task for parallel_config_name {parallel_config_name} to workers {worker_ids}"
        )

        # Create a future that will be resolved when the task is executed
        result_future = asyncio.Future()

        # Add the task to each worker's queue
        async with self._executor_condition:
            for worker_id in worker_ids:
                if worker_id >= len(self.all_workers):
                    raise ValueError(
                        f"Worker ID {worker_id} is out of range (max: {len(self.all_workers) - 1})"
                    )

                task_info = (
                    parallel_config_name,
                    worker_ids,
                    engine_config,
                    result_future,
                    input_config,
                    task_id,
                )
                self.worker_task_queues[worker_id].append(task_info)

            # Notify the condition to check if this task can be executed immediately
            self._executor_condition.notify_all()

        return result_future

    async def add_task_greedy_ilp(
        self,
        input_config: "InputConfig",
        engine_config: "EngineConfig",
        task_id: str | None = None,
    ) -> None:
        """
        package a request into the fields required for scheduling and put it into self.request_queue.
        """
        if task_id is None:
            task_id = f"task_{self.reqs_counter.inc()}"

        model_name = self.model_class_name
        profile_key = make_profile_key(input_config)
        t_dict = self.model_profiler.get_performance_data(model_name, profile_key)
        t_dict = {k: v * input_config.num_inference_steps for k, v in t_dict.items()}
        ddl_rel_sec = estimate_ddl(t_dict)

        ddl_abs = int(time.time() + ddl_rel_sec)

        self.request_queue.append(
            {
                "task_id": task_id,
                "input_config": input_config,
                "engine_config": engine_config,
                "ddl": ddl_abs,
                "t": t_dict,
            }
        )

    async def run_task_fix(
        self,
        worker_ids: List[int],
        input_config: "InputConfig",
        engine_config: "EngineConfig",
        parallel_config_name: str,
        result_future: asyncio.Future,
    ):
        """
        Run a task with fixed worker IDs. This is similar to run_task but for the fixed search mode.

        Args:
            worker_ids: List of worker IDs to use
            input_config: Input configuration for the model
            engine_config: Engine configuration
            parallel_config_name: Parallel configuration name
            result_future: Future to set the result
        """
        try:
            # First check if there's an existing executor with these workers
            executor = None
            full_executor_name = None  # Define variable here to avoid scope issues

            async with self._executor_condition:
                executor = self._find_executor_with_workers(
                    worker_ids, parallel_config_name
                )

                if executor is not None and executor.get_state() == "ready":
                    # We found a ready executor with the exact workers we need
                    _set_executor_state(executor, "busy", self.executor_states)
                    logger.debug(
                        f"Running executor {executor.name} for workers {worker_ids}"
                    )
                elif executor is not None and executor.get_state() == "busy":
                    return False
                else:  # executor is None
                    # Need to create a new executor
                    # Check if workers are available
                    all_workers_available = True
                    for worker_id in worker_ids:
                        worker = self.all_workers[worker_id]
                        worker_state = ray.get(worker.get_state.remote())
                        if worker_state not in ["ready", "idle"]:
                            all_workers_available = False
                            break

                    if not all_workers_available:
                        # Workers are not available, set the future as not done and return
                        logger.debug(
                            f"Not all workers {worker_ids} are available, will retry later"
                        )
                        return False

                    # Get the workers
                    workers = [self.all_workers[w_id] for w_id in worker_ids]

                    # Check if any workers are in executors and need to be shut down
                    for worker_id in worker_ids:
                        worker = self.all_workers[worker_id]
                        worker_state = ray.get(worker.get_state.remote())

                        if worker_state == "ready":
                            # Worker is part of an executor, find and shut down that executor
                            for config_name, executors in self.executors_dict.items():
                                for executor in list(executors):
                                    if (
                                        worker_id in executor.global_ranks
                                        and executor.get_state() == "ready"
                                    ):
                                        logger.debug(
                                            f"Shutting down executor {executor.name} to reclaim worker {worker_id}"
                                        )
                                        executor.shutdown(prepare_for_reuse=True)
                                        executors.remove(executor)
                                        if executor in self.executor_states:
                                            del self.executor_states[executor]
                                        break

                    # Create new executor
                    idx = self.executor_config_counters[parallel_config_name]
                    self.executor_config_counters[parallel_config_name] += 1
                    full_executor_name = f"{parallel_config_name}_{idx}"

                    executor = RayGPUExecutorAsync(
                        workers,
                        engine_config,
                        global_ranks=worker_ids,
                        name=full_executor_name,
                    )
                    self.executors_dict[parallel_config_name].append(executor)
                    self.executor_states[executor] = (
                        "busy"  # set busy before switch env
                    )
                    # executor.set_state("busy")

            # If we created a new executor, configure it
            if full_executor_name is not None and executor.name == full_executor_name:
                # Execute switch steps
                logger.debug(
                    f"Configuring new executor {full_executor_name} with workers {worker_ids}"
                )

                executor.set_state("busy")

                task1 = await executor._run_workers_async(
                    "switch_parallel_env",
                    world_size=len(executor.workers),
                    ranks=executor.global_ranks,
                    engine_config=executor.engine_config,
                )
                await task1

                executor.set_state("busy")

                task2 = await executor._run_workers_async(
                    "hotspa_model", engine_config=executor.engine_config
                )
                await task2

            # Execute the task
            logger.debug(f"Running task on executor {executor.name}")

            executor.set_state("busy")

            task3 = await executor._run_workers_async(
                "execute_model",
                engine_config=executor.engine_config,
                input_config=input_config,
                model_class=self.model_class,
            )
            result = await task3

            executor.set_state("ready")

            # Set executor to ready
            await self._notify_executor_ready_by_executor(executor)

            # Resolve the future
            if not result_future.done():
                result_future.set_result(result)

            return True

        except Exception as e:
            logger.error(f"Error in run_task_fix for workers {worker_ids}: {str(e)}")
            if not result_future.done():
                result_future.set_exception(e)

            # Make sure to cleanup and set workers back to idle
            for worker_id in worker_ids:
                worker = self.all_workers[worker_id]
                try:
                    ray.get(worker.set_state.remote("idle"))
                except Exception as cleanup_err:
                    logger.error(
                        f"Error setting worker {worker_id} back to idle: {str(cleanup_err)}"
                    )

            return False

    def _find_executor_with_workers(
        self, worker_ids: List[int], parallel_config_name: str
    ) -> Optional["RayGPUExecutorAsync"]:
        """
        Find an executor that uses exactly the specified worker IDs.

        Args:
            worker_ids: List of worker IDs to look for
            parallel_config_name: The parallel config name to match

        Returns:
            The executor if found, None otherwise
        """
        # Sort worker_ids for comparison
        worker_ids_set = set(worker_ids)

        executors = self.executors_dict.get(parallel_config_name, [])
        for executor in executors:
            executor_worker_ids_set = set(executor.global_ranks)
            if executor_worker_ids_set == worker_ids_set:
                return executor

        return None

    async def _scan_worker_queues(self):
        """
        Continuously scan worker queues and execute tasks when all required workers are available.
        This method runs as a background task.

        Implemented improvements:
        1. Tasks are only removed from queues after confirming they can be executed.
        2. Availability of executors and workers is verified before attempting to execute a task.
        3. Three main cases are handled:
           a) Existing and ready executor: Execute immediately.
           b) Existing but busy executor: Leave the task in queue for the next cycle.
           c) No suitable executor: Verify worker availability, reclaim if necessary,
              and only create a new executor if all workers are available.
        4. Executable tasks are processed separately to avoid blocking the scanning cycle.
        """
        logger.info("Starting queue scanner for fixed search mode")

        while self.should_scan_queues:
            task_groups = {}
            executable_tasks = []

            async with self._executor_condition:
                for worker_id, queue in self.worker_task_queues.items():
                    if not queue:
                        continue

                    # Get the first task in the queue
                    front_task = queue[0]
                    (
                        parallel_config_name,
                        required_worker_ids,
                        engine_config,
                        result_future,
                        input_config,
                        task_id,
                    ) = front_task

                    # Use tuple of (parallel_config_name, required_worker_ids) as key
                    task_key = (
                        parallel_config_name,
                        tuple(sorted(required_worker_ids)),
                    )
                    if task_key not in task_groups:
                        task_groups[task_key] = []

                    task_groups[task_key].append(worker_id)

                # Check which tasks can be executed (all required workers have the same task at the front)
                for (
                    parallel_config_name,
                    required_worker_ids,
                ), available_worker_ids in task_groups.items():
                    required_worker_ids_set = set(required_worker_ids)
                    available_worker_ids_set = set(available_worker_ids)

                    # If all required workers have this task at the front of their queues
                    # We need exact equality here, not just subset
                    if required_worker_ids_set == available_worker_ids_set:
                        task_details = self.worker_task_queues[
                            list(available_worker_ids)[0]
                        ][0]
                        _, _, engine_config, result_future, input_config, task_id = (
                            task_details
                        )

                        # Check if there's an existing executor with the required workers
                        executor = self._find_executor_with_workers(
                            list(required_worker_ids_set), parallel_config_name
                        )

                        if executor is not None and executor.get_state() == "ready":
                            # We found a ready executor with the exact workers we need
                            # Here we can safely remove the task from the queues and execute it immediately
                            for worker_id in required_worker_ids_set:
                                if (
                                    self.worker_task_queues[worker_id]
                                    and self.worker_task_queues[worker_id][0][3]
                                    is result_future
                                ):
                                    self.worker_task_queues[worker_id].pop(0)

                            # Mark the executor as busy
                            self.executor_states[executor] = "busy"
                            executor.set_state("busy")

                            # Create task to execute the model
                            exec_task = asyncio.create_task(
                                self._execute_with_existing_executor(
                                    executor,
                                    input_config,
                                    engine_config,
                                    result_future,
                                    task_id=task_id,
                                )
                            )
                            exec_task.set_name(
                                f"exec_task_{parallel_config_name}_{','.join(map(str, required_worker_ids_set))}"
                            )
                            executable_tasks.append(exec_task)

                            logger.debug(
                                f"Using existing ready executor {executor.name} for workers {required_worker_ids_set}"
                            )

                        elif executor is not None and executor.get_state() == "busy":
                            # The executor exists but is busy, do nothing and leave it in the queue
                            logger.info(
                                f"Executor {executor.name} is busy, leaving task in queue for workers {required_worker_ids_set}"
                            )
                            continue

                        else:
                            # No existing executor, check if we can create a new one
                            # Check if all required workers are available
                            all_workers_available = True
                            workers_to_reclaim = []

                            for worker_id in required_worker_ids_set:
                                worker = self.all_workers[worker_id]
                                worker_state = ray.get(worker.get_state.remote())

                                if worker_state not in ["idle", "ready"]:
                                    # The worker is not available
                                    all_workers_available = False
                                    logger.info(
                                        f"Worker {worker_id} is not available (state: {worker_state})"
                                    )
                                    break
                                elif worker_state == "ready":
                                    # The worker is in an executor and needs to be reclaimed
                                    workers_to_reclaim.append(worker_id)

                            if not all_workers_available:
                                # Not all workers are available, leave the task in the queue
                                logger.info(
                                    f"Not all workers {required_worker_ids_set} are available, will try later"
                                )
                                continue

                            # If we get here, all workers are available
                            # First, shut down executors that contain workers we need to reclaim
                            for worker_id in workers_to_reclaim:
                                for (
                                    config_name,
                                    executors,
                                ) in self.executors_dict.items():
                                    for executor in list(executors):
                                        if (
                                            worker_id in executor.global_ranks
                                            and executor.get_state() == "ready"
                                        ):
                                            logger.debug(
                                                f"Shutting down executor {executor.name} to reclaim worker {worker_id}"
                                            )
                                            executor.shutdown(prepare_for_reuse=True)
                                            executors.remove(executor)
                                            if executor in self.executor_states:
                                                del self.executor_states[executor]
                                            break

                            # Verify again that all workers are available after reclaiming
                            all_workers_available = True
                            for worker_id in required_worker_ids_set:
                                worker_state = ray.get(
                                    self.all_workers[worker_id].get_state.remote()
                                )
                                if worker_state != "idle":
                                    all_workers_available = False
                                    break

                            if not all_workers_available:
                                # Some worker is still not available after reclaiming
                                logger.info(
                                    "Some workers still not available after reclaiming, will try later"
                                )
                                continue

                            # Create a new executor
                            workers = [
                                self.all_workers[w_id]
                                for w_id in required_worker_ids_set
                            ]
                            idx = self.executor_config_counters[parallel_config_name]
                            self.executor_config_counters[parallel_config_name] += 1
                            executor_name = f"{parallel_config_name}_{idx}"

                            new_executor = RayGPUExecutorAsync(
                                workers,
                                engine_config,
                                global_ranks=list(required_worker_ids_set),
                                name=executor_name,
                            )
                            self.executors_dict[parallel_config_name].append(
                                new_executor
                            )
                            self.executor_states[new_executor] = "busy"
                            new_executor.set_state("busy")

                            # Now we can remove the task from the queues
                            for worker_id in required_worker_ids_set:
                                if (
                                    self.worker_task_queues[worker_id]
                                    and self.worker_task_queues[worker_id][0][3]
                                    is result_future
                                ):
                                    self.worker_task_queues[worker_id].pop(0)

                            # Create task to configure and execute the model
                            exec_task = asyncio.create_task(
                                self._setup_and_execute_with_new_executor(
                                    new_executor,
                                    input_config,
                                    engine_config,
                                    result_future,
                                    task_id=task_id,
                                )
                            )
                            exec_task.set_name(
                                f"new_exec_task_{parallel_config_name}_{','.join(map(str, required_worker_ids_set))}"
                            )
                            executable_tasks.append(exec_task)

                            logger.debug(
                                f"Created new executor {executor_name} for workers {required_worker_ids_set}"
                            )

                # Notify all waiting for a change in condition
                if executable_tasks:
                    self._executor_condition.notify_all()

            # Wait for all executable tasks to complete
            if executable_tasks:
                # Don't block the next scanning cycle
                asyncio.create_task(self._wait_for_exec_tasks(executable_tasks))
            await asyncio.sleep(1)

    async def _scan_request_queues(self):
        while self.should_scan_queues:
            async with self._executor_condition:
                await self.monitor.refresh()
                meta = self.detect_meta()  # use monitor to detect meta
                free = sum(1 for info in meta.values() if info["state"] != "busy")

                now = time.time()
                alloc = select_tasks(now=now, m_free=free, tasks=self.request_queue)
                logger.debug(
                    f"self.request_queue len is {len(self.request_queue)}, now is {now}"
                )
                logger.debug(f"select_task len is {len(alloc)}, select_task is {alloc}")
                input_configs = []
                engine_configs = []
                task_ids = []

                for task_id, k in alloc:
                    idx = next(
                        i
                        for i, d in enumerate(self.request_queue)
                        if d["task_id"] == task_id
                    )
                    req = self.request_queue.pop(idx)

                    new_conf, _ = create_new_config(
                        self.engine_config,
                        ulysses_degree=k,
                        ring_degree=1,
                        tensor_parallel_degree=1,
                        pipefusion_parallel_degree=1,
                        use_parallel_text_encoder=self.engine_config.runtime_config.use_parallel_text_encoder,
                        text_encoder_tensor_parallel_degree=1,
                        is_serving=True,
                    )
                    input_configs.append(req["input_config"])
                    engine_configs.append(new_conf)
                    task_ids.append(task_id)

            if len(input_configs) > 0:
                asyncio.create_task(
                    self.run_task_batch(input_configs, engine_configs, task_ids)
                )
            await asyncio.sleep(0.5)

    async def _execute_with_existing_executor(
        self, executor, input_config, engine_config, result_future, task_id=None
    ):
        """
        Execute a model on an existing executor that is already ready.
        """
        try:
            logger.debug(f"Executing model on existing executor {executor.name}")

            # Execute the model
            task1 = await executor._run_workers_async(
                "execute_model",
                engine_config=engine_config,
                input_config=input_config,
                model_class=self.model_class,
                task_id=task_id,
            )
            result = await task1
            executor.set_state("ready")

            # Resolve the future with the result
            if not result_future.done():
                result_future.set_result(result)

            return True

        except Exception as e:
            logger.error(f"Error executing model on executor {executor.name}: {str(e)}")
            if not result_future.done():
                result_future.set_exception(e)
            executor.set_state("ready")

            return False

    async def _setup_and_execute_with_new_executor(
        self, executor, input_config, engine_config, result_future, task_id=None
    ):
        """
        Configure a new executor and execute the model on it.
        """
        try:
            logger.debug(f"Setting up new executor {executor.name}")

            # Step 1: Configure the parallel environment
            task1 = await executor._run_workers_async(
                "switch_parallel_env",
                world_size=len(executor.workers),
                ranks=executor.global_ranks,
                engine_config=executor.engine_config,
            )
            await task1

            # Step 2: Initialize the model
            task2 = await executor._run_workers_async(
                "hotspa_model", engine_config=executor.engine_config
            )
            await task2

            # Step 3: Execute the model
            logger.debug(f"Executing model on new executor {executor.name}")
            task3 = await executor._run_workers_async(
                "execute_model",
                engine_config=engine_config,
                input_config=input_config,
                model_class=self.model_class,
                task_id=task_id,
            )
            result = await task3

            # Mark the executor as ready
            # await self._notify_executor_ready_by_executor(executor)
            executor.set_state("ready")

            # Resolve the future with the result
            if not result_future.done():
                result_future.set_result(result)

            return True

        except Exception as e:
            logger.error(
                f"Error setting up or executing model on new executor {executor.name}: {str(e)}"
            )
            if not result_future.done():
                result_future.set_exception(e)
            executor.set_state("ready")

            return False

    async def _wait_for_exec_tasks(self, exec_tasks):
        """
        Wait for execution tasks to complete and handle the results.
        This is a helper method that allows tasks to run in parallel without blocking the queue scanning.

        Args:
            exec_tasks: List of tasks to wait for
        """
        try:
            # Wait for all tasks to complete in parallel
            results = await asyncio.gather(*exec_tasks, return_exceptions=True)
            # Check results and log failed tasks
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Task {exec_tasks[i].get_name()} failed: {str(result)}"
                    )
                elif result is False:
                    logger.warning(
                        f"Task {exec_tasks[i].get_name()} returned False, will retry later"
                    )
        except Exception as e:
            logger.error(f"Error waiting for execution tasks: {str(e)}")

    async def get_ready_executor_or_reconfigure(
        self, parallel_config_name: str, new_engine_config: EngineConfig, task_id=None
    ):
        logger.debug(
            f"task_id is {task_id}, in get_ready_executor_or_reconfigure, parallel_config_name is {parallel_config_name}, new_engine_config.machine_id is {new_engine_config.machine_id}"
        )
        async with self._executor_condition:
            executor = self._find_ready_executor(parallel_config_name)
            logger.debug(
                f"task_id is {task_id}, after self._find_ready_executor, executor is {executor}"
            )
            if executor is not None:
                logger.debug(
                    f"task_id is {task_id}, after self._find_ready_executor and executor is not None, executor.get_state is {executor.get_state()}, self.executor_states[executor] is {self.executor_states[executor]}, executor.global_ranks is {executor.global_ranks}, executor.workers is {executor.workers}"
                )
                self.executor_states[executor] = "busy"
                executor.set_state("busy")
                return executor

            needed_workers = cal_needed_workers(new_engine_config)
            idle_workers = []
            for i, w in enumerate(self.all_workers):
                if ray.get(w.get_machine_id.remote()) != new_engine_config.machine_id:
                    continue
                state = ray.get(w.get_state.remote())
                logger.debug(
                    f"task_id is {task_id}, in get_ready_executor_or_reconfigure, w.get_state is {state}"
                )
                if state == "idle":
                    idle_workers.append(i)
            logger.debug(
                f"task_id is {task_id}, in get_ready_executor_or_reconfigure, needed_workers is {needed_workers}, idle_workers is {idle_workers}"
            )
            if len(idle_workers) >= needed_workers:
                # have enough idle workers
                idle_workers = idle_workers[:needed_workers]
                workers = [self.all_workers[w_id] for w_id in idle_workers]
                idx = self.executor_config_counters[parallel_config_name]
                self.executor_config_counters[parallel_config_name] += 1
                full_executor_name = f"{parallel_config_name}_{idx}"

                new_executor = RayGPUExecutorAsync(
                    workers,
                    new_engine_config,
                    global_ranks=idle_workers,
                    name=full_executor_name,
                )
                self.executors_dict[parallel_config_name].append(new_executor)
                self.executor_states[new_executor] = (
                    "ready"  # set busy before switch env
                )
                # Execute switch steps
                task1 = await new_executor._run_workers_async(
                    "switch_parallel_env",
                    world_size=len(new_executor.workers),
                    ranks=new_executor.global_ranks,
                    engine_config=new_executor.engine_config,
                )
                await task1
                # immediately set to busy, because the switch is not finished, and _run_worker will auto set it to ready
                new_executor.set_state("busy")

                self.executor_states[new_executor] = "busy"

                task2 = await new_executor._run_workers_async(
                    "hotspa_model", engine_config=new_executor.engine_config
                )
                await task2

                # switch complete, setting executor to ready.
                self.executor_states[new_executor] = "ready"
                new_executor.set_state("ready")
                waiting_list = self.waiting_reconfigure_tasks[parallel_config_name]
                logger.debug(
                    f"task_id is {task_id}, in reconfigure_executors, parallel_config_name is {parallel_config_name}, waiting_list is {waiting_list}"
                )
                for item in waiting_list:
                    logger.debug(
                        f"task_id is {task_id}, parallel_config_name is {parallel_config_name}, poping item is {id(item)}"
                    )
                while waiting_list:
                    fut = waiting_list.pop(0)
                    logger.debug(
                        f"task_id is {task_id}, in reconfigure_executors, parallel_config_name is {parallel_config_name}, poping fut is {id(fut)}"
                    )
                    fut.set_result(
                        new_executor
                    )  # Created the executor with this name, waking up the previously waiting coroutines that obtained the executor.
                self._executor_condition.notify_all()
                return new_executor

            executors = self.executors_dict.get(parallel_config_name, [])
            logger.debug(
                f"task_id is {task_id}, when idle worker is not enough, executors is {executors}"
            )
            if len(executors) > 0:
                fut = asyncio.Future()
                logger.debug(
                    f"task_id is {task_id}, create new fut {id(fut)}, for self.waiting_reconfigure_tasks {parallel_config_name}"
                )
                self.waiting_reconfigure_tasks[parallel_config_name].append(fut)
                logger.debug(
                    f"task_id is {task_id}, when idle worker is not enough and len(executors) > 0, fut is {id(fut)}, self.waiting_reconfigure_tasks[parallel_config_name] is {self.waiting_reconfigure_tasks[parallel_config_name]}"
                )
                for item in self.waiting_reconfigure_tasks[parallel_config_name]:
                    logger.debug(
                        f"task_id is {task_id}, parallel_config_name is {parallel_config_name}, poping item is {id(item)}"
                    )
                asyncio.create_task(
                    self.reconfigure_executors(
                        new_engine_config, parallel_config_name, task_id
                    )
                )
            else:
                # no executor of this executor，need reconfigure
                fut = asyncio.Future()
                logger.debug(
                    f"task_id is {task_id}, create new fut {id(fut)}, for self.waiting_reconfigure_tasks {parallel_config_name}"
                )
                self.waiting_reconfigure_tasks[parallel_config_name].append(fut)
                logger.debug(
                    f"task_id is {task_id}, when idle worker is not enough and len(executors) == 0, fut is {id(fut)}, self.waiting_reconfigure_tasks[parallel_config_name] is {self.waiting_reconfigure_tasks[parallel_config_name]}"
                )
                for item in self.waiting_reconfigure_tasks[parallel_config_name]:
                    logger.debug(
                        f"task_id is {task_id}, parallel_config_name is {parallel_config_name}, poping item is {id(item)}"
                    )
                asyncio.create_task(
                    self.reconfigure_executors(
                        new_engine_config, parallel_config_name, task_id
                    )
                )
        logger.debug(
            f"task_id is {task_id}, in the last of get_ready_executor_or_reconfigure, before await fut, fut is {id(fut)}"
        )
        executor = await fut
        logger.debug(
            f"task_id is {task_id}, in the last of get_ready_executor_or_reconfigure, after await fut, result of fut is {fut.result()}, executor is {executor}, executor.get_state is {executor.get_state()}, executor.global_ranks is {executor.global_ranks}, executor.workers is {executor.workers}"
        )
        return executor

    async def get_ready_executor_or_reconfigure_disaggregated(
        self,
        parallel_config_name: str,
        new_engine_config: EngineConfig,
        constraint_worker_ids: Optional[List[int]] = None,
        tag: str = "diffusion",
        task_id=None,
    ):
        logger.debug(
            f"task_id is {task_id}, in get_ready_executor_or_reconfigure, parallel_config_name is {parallel_config_name}, new_engine_config.machine_id is {new_engine_config.machine_id}"
        )
        async with self._executor_condition:
            executor = self._find_ready_executor(parallel_config_name)
            logger.debug(
                f"task_id is {task_id}, after self._find_ready_executor, executor is {executor}"
            )
            if executor is not None:
                if constraint_worker_ids is None or (
                    constraint_worker_ids is not None
                    and set(executor.global_ranks).issubset(set(constraint_worker_ids))
                ):
                    logger.debug(
                        f"task_id is {task_id}, after self._find_ready_executor and executor is not None, executor.get_state is {executor.get_state()}, self.executor_states[executor] is {self.executor_states[executor]}, executor.global_ranks is {executor.global_ranks}, executor.workers is {executor.workers}"
                    )
                    self.executor_states[executor] = "busy"
                    executor.set_state("busy")
                    return executor

            needed_workers = cal_needed_workers(new_engine_config)
            idle_workers = []
            for i, w in enumerate(self.categorize_workers[tag]):
                if ray.get(w.get_machine_id.remote()) != new_engine_config.machine_id:
                    continue
                state = ray.get(w.get_state.remote())
                logger.debug(
                    f"task_id is {task_id}, in get_ready_executor_or_reconfigure, w.get_state is {state}"
                )
                if state == "idle":
                    idle_workers.append(i)
            logger.debug(
                f"task_id is {task_id}, in get_ready_executor_or_reconfigure, needed_workers is {needed_workers}, idle_workers is {idle_workers}"
            )
            if len(idle_workers) >= needed_workers:
                # have enough idle workers
                idle_workers = idle_workers[:needed_workers]
                workers = [self.categorize_workers[tag][w_id] for w_id in idle_workers]
                idx = self.executor_config_counters[parallel_config_name]
                self.executor_config_counters[parallel_config_name] += 1
                full_executor_name = f"{parallel_config_name}_{idx}"

                new_executor = RayGPUExecutorAsync(
                    workers,
                    new_engine_config,
                    global_ranks=idle_workers,
                    name=full_executor_name,
                )
                self.executors_dict[parallel_config_name].append(new_executor)
                self.executor_states[new_executor] = (
                    "ready"  # set busy before switch env
                )
                # Execute switch steps
                task1 = await new_executor._run_workers_async(
                    "switch_parallel_env",
                    world_size=len(new_executor.workers),
                    ranks=new_executor.global_ranks,
                    engine_config=new_executor.engine_config,
                )
                await task1
                # immediately set to busy, because the switch is not finished, and _run_worker will auto set it to ready
                new_executor.set_state("busy")

                self.executor_states[new_executor] = "busy"

                task2 = await new_executor._run_workers_async(
                    "hotspa_model", engine_config=new_executor.engine_config
                )
                await task2

                # switch complete, setting executor to ready.
                self.executor_states[new_executor] = "ready"
                new_executor.set_state("ready")
                waiting_list = self.waiting_reconfigure_tasks[parallel_config_name]
                logger.debug(
                    f"task_id is {task_id}, in reconfigure_executors, parallel_config_name is {parallel_config_name}, waiting_list is {waiting_list}"
                )
                for item in self.waiting_reconfigure_tasks[parallel_config_name]:
                    logger.debug(
                        f"task_id is {task_id}, parallel_config_name is {parallel_config_name}, poping item is {id(item)}"
                    )
                while waiting_list:
                    fut = waiting_list.pop(0)
                    logger.debug(
                        f"task_id is {task_id}, in reconfigure_executors, parallel_config_name is {parallel_config_name}, poping fut is {id(fut)}"
                    )
                    fut.set_result(
                        new_executor
                    )  # Created the executor with this name, waking up the previously waiting coroutines that obtained the executor.
                self._executor_condition.notify_all()
                return new_executor

            executors = self.executors_dict.get(parallel_config_name, [])
            logger.debug(
                f"task_id is {task_id}, when idle worker is not enough, executors is {executors}"
            )
            if len(executors) > 0:
                fut = asyncio.Future()
                logger.debug(
                    f"task_id is {task_id}, create new fut {id(fut)}, for self.waiting_reconfigure_tasks {parallel_config_name}"
                )
                self.waiting_reconfigure_tasks[parallel_config_name].append(fut)
                logger.debug(
                    f"task_id is {task_id}, when idle worker is not enough and len(executors) > 0, fut is {id(fut)}, self.waiting_reconfigure_tasks[parallel_config_name] is {self.waiting_reconfigure_tasks[parallel_config_name]}"
                )
                for item in self.waiting_reconfigure_tasks[parallel_config_name]:
                    logger.debug(
                        f"task_id is {task_id}, parallel_config_name is {parallel_config_name}, poping item is {id(item)}"
                    )
                asyncio.create_task(
                    self.reconfigure_executors_disaggregated(
                        new_engine_config,
                        parallel_config_name,
                        constraint_worker_ids,
                        tag,
                        task_id,
                    )
                )
            else:
                # no executor of this executor，need reconfigure
                fut = asyncio.Future()
                logger.debug(
                    f"task_id is {task_id}, create new fut {id(fut)}, for self.waiting_reconfigure_tasks {parallel_config_name}"
                )
                self.waiting_reconfigure_tasks[parallel_config_name].append(fut)
                logger.debug(
                    f"task_id is {task_id}, when idle worker is not enough and len(executors) == 0, fut is {id(fut)}, self.waiting_reconfigure_tasks[parallel_config_name] is {self.waiting_reconfigure_tasks[parallel_config_name]}"
                )
                for item in self.waiting_reconfigure_tasks[parallel_config_name]:
                    logger.debug(
                        f"task_id is {task_id}, parallel_config_name is {parallel_config_name}, poping item is {id(item)}"
                    )
                asyncio.create_task(
                    self.reconfigure_executors_disaggregated(
                        new_engine_config,
                        parallel_config_name,
                        constraint_worker_ids,
                        tag,
                        task_id,
                    )
                )
        logger.debug(
            f"task_id is {task_id}, in the last of get_ready_executor_or_reconfigure, before await fut, fut is {id(fut)}"
        )
        executor = await fut
        logger.debug(
            f"task_id is {task_id}, in the last of get_ready_executor_or_reconfigure, after await fut, result of fut is {fut.result()}, executor is {executor}"
        )
        return executor

    async def reconfigure_executors(
        self, new_engine_config: EngineConfig, parallel_config_name: str, task_id=None
    ):
        # based on new_engine_config to reconfigure
        executor_name, worker_ids = await self.search_reconfigure_worker(
            new_engine_config, task_id
        )

        async with self._executor_condition:
            workers = [self.all_workers[w_id] for w_id in worker_ids]
            logger.debug(
                f"task_id is {task_id}, in reconfigure_executors, parallel_config_name is {parallel_config_name}, workers is {workers}"
            )
            idx = self.executor_config_counters[parallel_config_name]
            self.executor_config_counters[parallel_config_name] += 1
            full_executor_name = f"{parallel_config_name}_{idx}"

            new_executor = RayGPUExecutorAsync(
                workers,
                new_engine_config,
                global_ranks=worker_ids,
                name=full_executor_name,
            )
            logger.debug(
                f"task_id is {task_id}, in reconfigure_executors, new_executor is {new_executor}, new_executor's state is {new_executor.get_state()}, new_executor's workers is {new_executor.workers}, global_ranks is {new_executor.global_ranks}"
            )
            self.executors_dict[parallel_config_name].append(new_executor)
            self.executor_states[new_executor] = "busy"  # set busy before switch env

        # Execute switch steps
        logger.debug(
            f"task_id is {task_id}, in reconfigure_executors, before switch_parallel_env, new_executor is {new_executor}, new_executor's state is {new_executor.get_state()}, new_executor's workers is {new_executor.workers}, global_ranks is {new_executor.global_ranks}"
        )
        task1 = await new_executor._run_workers_async(
            "switch_parallel_env",
            world_size=len(new_executor.workers),
            ranks=new_executor.global_ranks,
            engine_config=new_executor.engine_config,
        )
        await task1
        logger.debug(
            f"task_id is {task_id}, in reconfigure_executors, after switch_parallel_env, new_executor is {new_executor}, new_executor's state is {new_executor.get_state()}, new_executor's workers is {new_executor.workers}, global_ranks is {new_executor.global_ranks}"
        )
        # immediately set to busy, because the switch is not finished, and _run_worker will auto set it to ready
        new_executor.set_state("busy")
        async with self._executor_condition:
            self.executor_states[new_executor] = "busy"
        logger.debug(
            f"task_id is {task_id}, in reconfigure_executors, before hotspa_model, new_executor is {new_executor}, new_executor's state is {new_executor.get_state()}, new_executor's workers is {new_executor.workers}, global_ranks is {new_executor.global_ranks}"
        )
        task2 = await new_executor._run_workers_async(
            "hotspa_model", engine_config=new_executor.engine_config
        )
        await task2
        logger.debug(
            f"task_id is {task_id}, in reconfigure_executors, after hotspa_model, new_executor is {new_executor}, new_executor's state is {new_executor.get_state()}, new_executor's workers is {new_executor.workers}, global_ranks is {new_executor.global_ranks}"
        )
        # switch complete, setting executor to ready.
        async with self._executor_condition:
            self.executor_states[new_executor] = "ready"
            new_executor.set_state("ready")
            waiting_list = self.waiting_reconfigure_tasks[parallel_config_name]
            for item in waiting_list:
                logger.debug(
                    f"task_id is {task_id}, parallel_config_name is {parallel_config_name}, poping item is {id(item)}"
                )
            while waiting_list:
                fut = waiting_list.pop(0)
                logger.debug(
                    f"task_id is {task_id}, in reconfigure_executors, parallel_config_name is {parallel_config_name}, poping fut is {id(fut)}, new_executor is {new_executor}, new_executor's state is {new_executor.get_state()}, new_executor's workers is {new_executor.workers}, global_ranks is {new_executor.global_ranks}"
                )
                fut.set_result(
                    new_executor
                )  # Created the executor with this name, waking up the previously waiting coroutines that obtained the executor.
            self._executor_condition.notify_all()

    async def reconfigure_executors_disaggregated(
        self,
        new_engine_config: EngineConfig,
        parallel_config_name: str,
        constraint_worker_ids: Optional[List[int]] = None,
        tag: str = "diffusion",
        task_id=None,
    ):
        # based on new_engine_config to reconfigure
        executor_name, worker_ids = await self.search_reconfigure_worker_disaggregated(
            new_engine_config, constraint_worker_ids, tag, task_id
        )

        async with self._executor_condition:
            workers = [self.categorize_workers[tag][w_id] for w_id in worker_ids]
            logger.debug(
                f"task_id is {task_id}, in reconfigure_executors, parallel_config_name is {parallel_config_name}, workers is {workers}"
            )
            idx = self.executor_config_counters[parallel_config_name]
            self.executor_config_counters[parallel_config_name] += 1
            full_executor_name = f"{parallel_config_name}_{idx}"

            new_executor = RayGPUExecutorAsync(
                workers,
                new_engine_config,
                global_ranks=worker_ids,
                name=full_executor_name,
            )
            logger.debug(
                f"task_id is {task_id}, in reconfigure_executors, new_executor is {new_executor}, new_executor's workers is {new_executor.workers}, global_ranks is {new_executor.global_ranks}"
            )
            self.executors_dict[parallel_config_name].append(new_executor)
            self.executor_states[new_executor] = "busy"  # set busy before switch env

        # Execute switch steps
        task1 = await new_executor._run_workers_async(
            "switch_parallel_env",
            world_size=len(new_executor.workers),
            ranks=new_executor.global_ranks,
            engine_config=new_executor.engine_config,
        )
        await task1
        # immediately set to busy, because the switch is not finished, and _run_worker will auto set it to ready
        new_executor.set_state("busy")
        async with self._executor_condition:
            self.executor_states[new_executor] = "busy"

        task2 = await new_executor._run_workers_async(
            "hotspa_model", engine_config=new_executor.engine_config
        )
        await task2

        # switch complete, setting executor to ready.
        async with self._executor_condition:
            self.executor_states[new_executor] = "ready"
            new_executor.set_state("ready")
            waiting_list = self.waiting_reconfigure_tasks[parallel_config_name]
            for item in waiting_list:
                logger.debug(
                    f"task_id is {task_id}, parallel_config_name is {parallel_config_name}, poping item is {id(item)}"
                )
            while waiting_list:
                fut = waiting_list.pop(0)
                logger.debug(
                    f"task_id is {task_id}, in reconfigure_executors, parallel_config_name is {parallel_config_name}, poping fut is {id(fut)}"
                )
                fut.set_result(
                    new_executor
                )  # Created the executor with this name, waking up the previously waiting coroutines that obtained the executor.
            self._executor_condition.notify_all()

    async def search_reconfigure_worker_v1(
        self, new_engine_config: EngineConfig, task_id=None
    ) -> Tuple[str, List[int]]:
        # Calculate the required number of workers based on the degree of parallelism.
        needed_workers = cal_needed_workers(new_engine_config)

        while True:
            async with self._executor_condition:
                idle_workers = []
                for i, w in enumerate(self.all_workers):
                    if (
                        ray.get(w.get_machine_id.remote())
                        != new_engine_config.machine_id
                    ):
                        continue
                    state = ray.get(w.get_state.remote())
                    if state == "idle":
                        idle_workers.append(i)
                        if len(idle_workers) >= needed_workers:
                            break

                if len(idle_workers) < needed_workers:
                    # Not enough will recycle ready's executor
                    found = False
                    for config_name, executors in self.executors_dict.items():
                        for exe in list(executors):
                            if exe.get_machine_id() != new_engine_config.machine_id:
                                continue
                            # exe_state = self.executor_states.get(exe, "busy")
                            exe_state = exe.get_state()
                            if exe_state == "ready":
                                exe.shutdown(prepare_for_reuse=True)
                                executors.remove(exe)
                                del self.executor_states[exe]
                                # iter all_workers again
                                new_idle_workers = []
                                for j, w2 in enumerate(self.all_workers):
                                    if (
                                        ray.get(w2.get_machine_id.remote())
                                        != new_engine_config.machine_id
                                    ):
                                        continue
                                    s = ray.get(w2.get_state.remote())
                                    if s == "idle" and j not in idle_workers:
                                        new_idle_workers.append(j)
                                idle_workers.extend(new_idle_workers)
                                if len(idle_workers) >= needed_workers:
                                    found = True
                                    break
                        if found:
                            break

                if len(idle_workers) >= needed_workers:
                    # have sufficient idle worker
                    for w_id in idle_workers[:needed_workers]:
                        w = self.all_workers[w_id]
                        ray.get(w.set_state.remote("ready"))
                    executor_name = "dynamic_exec"
                    return executor_name, idle_workers[:needed_workers]

                # Not enough, waiting for resources to appear.
                fut = asyncio.Future()
                logger.debug(
                    f"task_id is {task_id}, create new fut {id(fut)}, for self.waiting_worker_tasks"
                )
                self.waiting_worker_tasks.append(fut)

            await fut
            # Check again in a loop after the fut is completed.

    async def search_reconfigure_worker(
        self, new_engine_config: EngineConfig, task_id=None
    ) -> Tuple[str, List[int]]:
        needed_workers = cal_needed_workers(new_engine_config)

        while True:
            async with self._executor_condition:
                idle_workers = []
                total_found_worker_num = 0
                ready_executors = []
                for i, w in enumerate(self.all_workers):
                    if (
                        ray.get(w.get_machine_id.remote())
                        != new_engine_config.machine_id
                    ):
                        continue
                    state = ray.get(w.get_state.remote())
                    if state == "idle":
                        idle_workers.append(i)
                        if len(idle_workers) >= needed_workers:
                            break
                total_found_worker_num += len(idle_workers)
                logger.debug(
                    f"task_id is {task_id}, in search_reconfigure_worker, after add idle_workers, total_found_worker_num is {total_found_worker_num}, needed_workers is {needed_workers}, idle_workers is {idle_workers}"
                )
                if total_found_worker_num < needed_workers:
                    # Not enough will recycle ready's executor
                    found = False
                    for config_name, executors in self.executors_dict.items():
                        for exe in list(executors):
                            if exe.get_machine_id() != new_engine_config.machine_id:
                                continue
                            exe_state = exe.get_state()
                            logger.debug(
                                f"task_id is {task_id}, in search_reconfigure_worker, first travel, executor is {exe}, executor's workers is {exe.workers}, global_ranks is {exe.global_ranks}, exe_state is {exe_state}"
                            )
                            if exe_state == "ready":
                                total_found_worker_num += len(exe.workers)
                                logger.debug(
                                    f"task_id is {task_id}, in search_reconfigure_worker, after shutdown exe, config_name is {config_name}, total_found_worker_num is {total_found_worker_num}"
                                )
                                ready_executors.append((config_name, exe))
                                logger.debug(
                                    f"task_id is {task_id}, in search_reconfigure_worker, add ready_executor {exe}, config_name is {config_name}, total_found_worker_num is {total_found_worker_num}, needed_workers is {needed_workers}"
                                )
                                if total_found_worker_num >= needed_workers:
                                    found = True
                                    logger.debug(
                                        f"task_id is {task_id}, set found to true, config_name is {config_name}, total_found_worker_num is {total_found_worker_num}, needed_workers is {needed_workers}"
                                    )
                                    break
                        if found:
                            logger.debug(
                                f"task_id is {task_id}, second found break, config_name is {config_name}, total_found_worker_num is {total_found_worker_num}, needed_workers is {needed_workers}"
                            )
                            break
                logger.debug(
                    f"task_id is {task_id}, in search_reconfigure_worker, after found ready_executors, ready_executors is {ready_executors}, total_found_worker_num is {total_found_worker_num}, needed_workers is {needed_workers}"
                )
                if total_found_worker_num >= needed_workers:
                    logger.debug(
                        f"task_id is {task_id}, in search_reconfigure_worker, have enough workers, ready_executors is {ready_executors}"
                    )
                    for config_name, exe in ready_executors:
                        if exe.get_machine_id() != new_engine_config.machine_id:
                            continue
                        logger.debug(
                            f"task_id is {task_id}, search_reconfigure_worker, second travel, executor is {exe}, executor's workers is {exe.workers}, global_ranks is {exe.global_ranks}, exe's state is {exe.get_state()}"
                        )
                        for eworker in exe.workers:
                            logger.debug(
                                f"task_id is {task_id}, search_reconfigure_worker, for the exe to be shutdown, worker is {eworker}, state is {ray.get(eworker.get_state.remote())}"
                            )
                        exe.shutdown(prepare_for_reuse=True)
                        for eworker in exe.workers:
                            logger.debug(
                                f"task_id is {task_id}, search_reconfigure_worker, after shutdown, worker is {eworker}, state is {ray.get(eworker.get_state.remote())}"
                            )
                        for jj, w22 in enumerate(self.all_workers):
                            if (
                                ray.get(w22.get_machine_id.remote())
                                != new_engine_config.machine_id
                            ):
                                continue
                            logger.debug(
                                f"task_id is {task_id}, check after shutdown if the worker is idle, jj is {jj}, w22 is {w22}, w22's state is {ray.get(w22.get_state.remote())}"
                            )
                        self.executors_dict[config_name].remove(
                            exe
                        )  # remove from the config correctly
                        del self.executor_states[exe]

                        # remark idle worker
                        new_idle_workers = []
                        for j, w2 in enumerate(self.all_workers):
                            if (
                                ray.get(w2.get_machine_id.remote())
                                != new_engine_config.machine_id
                            ):
                                continue
                            s = ray.get(w2.get_state.remote())
                            logger.debug(
                                f"task_id is {task_id}, idle workers is {idle_workers}, j is {j}, in remark idle workers, w2 is {w2}, w2's state is {s}"
                            )
                            if s == "idle" and j not in idle_workers:
                                new_idle_workers.append(j)
                        idle_workers.extend(new_idle_workers)
                        logger.debug(
                            f"task_id is {task_id}, search_reconfigure_worker,  after extend, idle_workers is {idle_workers}"
                        )

                    # have sufficient idle worker
                    for w_id in idle_workers[:needed_workers]:
                        w = self.all_workers[w_id]
                        ray.get(w.set_state.remote("ready"))
                    executor_name = "dynamic_exec"
                    return executor_name, idle_workers[:needed_workers]

                # Not enough, waiting for resources to appear.
                fut = asyncio.Future()
                logger.debug(
                    f"task_id is {task_id}, create new fut {id(fut)}, for self.waiting_worker_tasks"
                )
                logger.debug(
                    f"task_id is {task_id}, in search_reconfigure_worker, not have enough idle workers, fut is {id(fut)}"
                )
                self.waiting_worker_tasks.append(fut)
            logger.debug(
                f"task_id is {task_id}, in search_reconfigure_worker, waiting for resources to appear, fut is {id(fut)}"
            )
            await fut
            logger.debug(
                f"task_id is {task_id}, in search_reconfigure_worker, after waiting for resources to appear, result of fut is {fut.result()}, fut is {id(fut)}"
            )
            # Check again in a loop after the fut is completed.

    async def search_reconfigure_worker_disaggregated(
        self,
        new_engine_config: EngineConfig,
        constraint_worker_ids: Optional[List[int]] = None,
        tag: str = "diffusion",
        task_id=None,
    ) -> Tuple[str, List[int]]:
        needed_workers = cal_needed_workers(new_engine_config)

        while True:
            async with self._executor_condition:
                idle_workers = []
                total_found_worker_num = 0
                ready_executors = []
                for i, w in enumerate(self.categorize_workers[tag]):
                    if (
                        ray.get(w.get_machine_id.remote())
                        != new_engine_config.machine_id
                    ):
                        continue
                    state = ray.get(w.get_state.remote())
                    if state == "idle":
                        idle_workers.append(i)
                        if len(idle_workers) >= needed_workers:
                            break
                total_found_worker_num += len(idle_workers)
                if total_found_worker_num < needed_workers:
                    # Not enough will recycle ready's executor
                    found = False
                    for config_name, executors in self.executors_dict.items():
                        for exe in list(executors):
                            if exe.get_machine_id() != new_engine_config.machine_id:
                                continue
                            if constraint_worker_ids is not None and not set(
                                exe.global_ranks
                            ).issubset(set(constraint_worker_ids)):
                                continue
                            exe_state = exe.get_state()
                            logger.debug(
                                f"task_id is {task_id}, in search_reconfigure_worker, first travel, executor is {exe}, executor's workers is {exe.workers}, global_ranks is {exe.global_ranks}, exe_state is {exe_state}"
                            )
                            if exe_state == "ready":
                                total_found_worker_num += len(exe.workers)
                                ready_executors.append((config_name, exe))
                                if total_found_worker_num >= needed_workers:
                                    found = True
                                    break
                        if found:
                            break

                if total_found_worker_num >= needed_workers:
                    for config_name, exe in ready_executors:
                        if exe.get_machine_id() != new_engine_config.machine_id:
                            continue
                        if constraint_worker_ids is not None and not set(
                            exe.global_ranks
                        ).issubset(set(constraint_worker_ids)):
                            continue
                        logger.debug(
                            f"task_id is {task_id}, search_reconfigure_worker, second travel, executor is {exe}, executor's workers is {exe.workers}, global_ranks is {exe.global_ranks}, exe's state is {exe.get_state()}"
                        )
                        for eworker in exe.workers:
                            logger.debug(
                                f"task_id is {task_id}, search_reconfigure_worker, for the exe to be shutdown, worker is {eworker}, state is {ray.get(eworker.get_state.remote())}"
                            )
                        exe.shutdown(prepare_for_reuse=True)
                        self.executors_dict[config_name].remove(
                            exe
                        )  # remove from the config correctly
                        del self.executor_states[exe]

                        # remark idle worker
                        new_idle_workers = []
                        for j, w2 in enumerate(self.categorize_workers[tag]):
                            if (
                                ray.get(w2.get_machine_id.remote())
                                != new_engine_config.machine_id
                            ):
                                continue
                            s = ray.get(w2.get_state.remote())
                            logger.debug(
                                f"task_id is {task_id}, idle workers is {idle_workers}, j is {j}, in remark idle workers, w2 is {w2}, w2's state is {s}"
                            )
                            if s == "idle" and j not in idle_workers:
                                new_idle_workers.append(j)
                        idle_workers.extend(new_idle_workers)
                        logger.debug(
                            f"task_id is {task_id}, search_reconfigure_worker,  after extend, idle_workers is {idle_workers}"
                        )

                    # have sufficient idle worker
                    for w_id in idle_workers[:needed_workers]:
                        w = self.categorize_workers[tag][w_id]
                        ray.get(w.set_state.remote("ready"))
                    executor_name = "dynamic_exec"
                    return executor_name, idle_workers[:needed_workers]

                # Not enough, waiting for resources to appear.
                fut = asyncio.Future()
                logger.debug(
                    f"task_id is {task_id}, create new fut {id(fut)}, for self.waiting_worker_tasks"
                )
                logger.debug(
                    f"task_id is {task_id}, in search_reconfigure_worker, not have enough idle workers, fut is {id(fut)}"
                )
                self.waiting_worker_tasks.append(fut)
            logger.debug(
                f"task_id is {task_id}, in search_reconfigure_worker, waiting for resources to appear, fut is {id(fut)}"
            )
            await fut
            logger.debug(
                f"task_id is {task_id}, in search_reconfigure_worker, after waiting for resources to appear, result of fut is {fut.result()}, fut is {id(fut)}"
            )
            # Check again in a loop after the fut is completed.

    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
        num_gpus = 1
        logger.debug(f"init_workers_ray's pid = {os.getpid()}")
        driver_ip = get_ip()
        work_dir = os.getcwd()
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )
            logger.debug(f"bundle_id = {bundle_id}")

            worker = (
                ray.remote(RayWorkerHetudit)
                .options(
                    max_concurrency=10,
                    num_cpus=10,
                    num_gpus=num_gpus,
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )
                .remote(bundle_id, work_dir, bundle_id // 8)
            )

            # worker_ip = ray.get(worker.get_node_ip.remote())
            # if worker_ip == driver_ip and self.driver_dummy_worker is None:
            #     self.driver_dummy_worker = worker

            self.all_workers.append(worker)

        worker_node_and_gpu_ids = ray.get(
            [worker.get_node_and_gpu_ids.remote() for worker in self.all_workers[:]]
        )

        node_workers = defaultdict(list)
        node_gpus = defaultdict(list)

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids, start=0):
            node_workers[node_id].append(i)
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        # set_cuda_visible_devices(node_gpus[driver_node_id])
        for worker, (node_id, _) in zip(self.all_workers, worker_node_and_gpu_ids):
            worker.set_cuda_visible_devices.remote(node_gpus[node_id])

        rank0_ip = ray.get(self.all_workers[0].get_node_ip.remote())
        rank0_port = ray.get(self.all_workers[0].get_node_open_port.remote())
        distributed_init_method = get_distributed_init_method(rank0_ip, rank0_port)
        engine_config = copy.deepcopy(self.engine_config)

        init_tasks = []
        for rank, (worker, (node_id, _)) in enumerate(
            zip(self.all_workers, worker_node_and_gpu_ids),
            start=0,
        ):
            logger.debug(f"node_id = {node_id}, rank = {rank}")
            local_rank = node_workers[node_id].index(rank)
            task = worker.init_worker.remote(
                lambda rank=rank, local_rank=local_rank: Worker(
                    engine_config,
                    local_rank,
                    rank,
                    distributed_init_method,
                    work_dir=work_dir,
                )
            )
            init_tasks.append(task)

        ray.get(init_tasks)
        worker_handles = {rank: handle for rank, handle in enumerate(self.all_workers)}

        set_handle_tasks = []
        for worker in self.all_workers:
            task = worker.set_worker_handles.remote(worker_handles)
            set_handle_tasks.append(task)

        results = ray.get(set_handle_tasks)
        if not all(results):
            raise RuntimeError("Failed to set worker handles on one or more workers.")
        logger.info("Successfully distributed all worker handles to each worker.")

        # Serialize NIXL initialization and handshake (p2p strategy only)
        if self.engine_config.runtime_config.adjust_strategy == "p2p":
            ranks = sorted(worker_handles.keys())
            # 1) Serialize agent creation to avoid UCX concurrent initialization segfault
            for r in ranks:
                ray.get(worker_handles[r].create_nixl_manager.remote())
            # 2) Serialize metadata collection after all agents are created
            all_meta = []
            for r in ranks:
                all_meta.append(ray.get(worker_handles[r].get_nixl_metadata.remote()))
            # 3) Serialize handshake (add_remote_agent + make_connection)
            for r in ranks:
                ray.get(worker_handles[r].init_nixl_peers.remote(all_meta))

    def add_request(self, request: Dict[str, Any]) -> None:
        pass

    def execute(self) -> Dict[str, Any]:
        pass

    def search_best_static_placement(self, machine_num=1):
        if machine_num == 1:
            new_config, _ = create_new_config(
                self.engine_config,
                ulysses_degree=1,
                ring_degree=1,
                tensor_parallel_degree=1,
                pipefusion_parallel_degree=1,
                use_parallel_text_encoder=self.engine_config.runtime_config.use_parallel_text_encoder,
                text_encoder_tensor_parallel_degree=1,
                is_serving=True,
            )
            return {
                f"instance{i}": (new_config, [i]) for i in range(len(self.all_workers))
            }
        else:
            ret_config = {}
            for iter in range(machine_num):
                new_config, _ = create_new_config(
                    self.engine_config,
                    ulysses_degree=1,
                    ring_degree=1,
                    tensor_parallel_degree=1,
                    pipefusion_parallel_degree=1,
                    use_parallel_text_encoder=self.engine_config.runtime_config.use_parallel_text_encoder,
                    text_encoder_tensor_parallel_degree=1,
                    is_serving=True,
                    machine_id=iter,
                )
                new_items = {
                    f"instance{i}": (new_config, [i])
                    for i in range(iter * 8, (iter + 1) * 8)
                }
                ret_config.update(new_items)
            logger.debug(f"ret_config = {ret_config}")
        return ret_config

    def static_placement_init(self):
        logger.debug(
            f"in static placement_init, machine_num = {self.engine_config.machine_num}"
        )
        instance_dict = self.search_best_static_placement(
            self.engine_config.machine_num
        )
        for name, (engine_config, gpu_ids) in instance_dict.items():
            workers = [self.all_workers[i] for i in gpu_ids]
            parallel_config_name = generate_parallel_config_name(
                engine_config.parallel_config, engine_config.machine_id
            )
            idx = self.executor_config_counters[parallel_config_name]
            self.executor_config_counters[parallel_config_name] += 1

            executor_name = f"{parallel_config_name}_{idx}"
            executor = RayGPUExecutorAsync(
                workers, engine_config, global_ranks=gpu_ids, name=executor_name
            )
            self.executors_dict[parallel_config_name].append(executor)
            self.executor_states[executor] = "ready"

    @classmethod
    def from_engine_args(
        cls,
        serving_config: "ServingConfig",
        search_mode="random",
        use_disaggregated_encode_decode=False,
        stage_level=False,
        encode_worker_ids=None,
        decode_worker_ids=None,
        model_class_name="",
    ):
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = serving_config.engine_config
        parallel_config = engine_config.parallel_config
        # Initialize the cluster and specify the executor class.
        initialize_ray_cluster(parallel_config)

        executor_class = RayGPUExecutorAsync

        # Create the LLM engine.
        engine = cls(
            engine_config=engine_config,
            model_class=serving_config.model_class,
            executor_class=executor_class,
            search_mode=search_mode,
            use_disaggregated_encode_decode=use_disaggregated_encode_decode,
            stage_level=stage_level,
            encode_worker_ids=encode_worker_ids,
            decode_worker_ids=decode_worker_ids,
            model_class_name=model_class_name,
        )
        return engine

    def _find_ready_executor(
        self, parallel_config_name: str
    ) -> Optional["RayGPUExecutorAsync"]:
        # Reliance on local state when invoked within the lock.
        executors = self.executors_dict.get(parallel_config_name, [])
        for exe in executors:
            state = exe.get_state()
            if state == "ready":
                return exe
        return None

    def _notify_executor_ready(self, parallel_config_name: str):
        # When the executor becomes ready, attempts to allocate to waiting requests.
        while self.waiting_tasks[parallel_config_name]:
            exe = self._find_ready_executor(parallel_config_name)
            if exe is None:
                break
            fut = self.waiting_tasks[parallel_config_name].pop(0)
            # Assigned to the request, set as busy.
            self.executor_states[exe] = "busy"
            exe.set_state("busy")
            logger.debug(f"in _notify_executor_ready, exe = {exe}, set exe to busy")
            fut.set_result(exe)

    async def _notify_executor_ready_by_executor(
        self, executor: "RayGPUExecutorAsync", task_id=None
    ):
        # executor finished task, set to ready
        logger.debug(
            f"task_id is {task_id}, come into _notify_executor_ready_by_executor"
        )
        async with self._executor_condition:
            logger.debug(
                f"task_id is {task_id}, in _notify_executor_ready_by_executor, into lock"
            )
            self.executor_states[executor] = "ready"
            executor.set_state("ready")
            logger.debug(
                f"task_id is {task_id}, in _notify_executor_ready_by_executor, executor = {executor}, set state to ready"
            )
            for eworker in executor.workers:
                ray.get(eworker.set_state.remote("ready"))
            logger.debug(
                f"task_id is {task_id}, in _notify_executor_ready_by_executor, executor = {executor}, set state to ready"
            )
            self._notify_executor_ready(
                generate_parallel_config_name(executor.engine_config.parallel_config)
            )
            # There may be waiting workers or futures for reconfiguration, which are also notified here.
            while self.waiting_worker_tasks:
                fut = self.waiting_worker_tasks.pop(0)
                fut.set_result(True)
            self._executor_condition.notify_all()
