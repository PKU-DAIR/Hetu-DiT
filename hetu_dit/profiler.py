import time
import logging
import inspect
import os
from typing import Optional, Dict
import torch
from pathlib import Path
from collections import defaultdict
from hetu_dit.logger import init_logger

logger = init_logger(__name__)


"""
hetu_dit/profiler.log
example log:

09:56:59 [INFO] 
Task 'task-1745632611302' Completed
Task Config: {'prompt': 'A futuristic cityscape', 'negative_prompt': 'low quality, blurry', 'height': 1024, 'width': 128, 'num_inference_steps': 30}
Worker_num: 4
Time Summary:
  - Request total time : 8.2666 s
  - Max execution time : 2.8879 s (max encode time : 0.1779 s, max diffusion time : 2.6054 s, max vae time : 0.0647 s)
  - Avg execution time : 2.8140 s (avg encode time : 0.1352 s, avg diffusion time : 2.5880 s, avg vae time : 0.0647 s)
GPU Memory Usage(among workers):
 - Max Allocated memory : 17159.485 MB
 - Avg Allocated memory : 16973.768 MB
Worker rank list: [1, 2, 3, 4]
"""


class Profiler:
    """A simple profiler to record execution time and GPU memory usage."""

    def __init__(
        self,
        name: str = "",
        log_file=None,
        overwrite: bool = False,
        count_mem: bool = True,
    ):
        frame = inspect.stack()[1]
        self.caller_filename = frame.filename
        self.name = name or os.path.basename(self.caller_filename)
        if log_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            log_file = f"{current_dir}/profiler{os.getpid()}.log"

        self.track_gpu_mem = torch.cuda.is_available() and count_mem
        if self._is_ray_driver():
            self.log_file = Path(log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.log_file.touch(exist_ok=True)
        else:
            self.log_file = Path(os.devnull)

        self._reset_state()
        self.log_setup = False
        self.overwrite = overwrite

        self.dict = {}

    @staticmethod
    def _is_ray_driver() -> bool:
        try:
            import ray

            if not ray.is_initialized():
                return True
            ctx = ray.get_runtime_context()

            is_actor = (
                hasattr(ctx, "actor_id")
                and hasattr(ctx.actor_id, "is_nil")
                and (not ctx.actor_id.is_nil())
            )
            is_task = (
                hasattr(ctx, "task_id")
                and hasattr(ctx.task_id, "is_nil")
                and (not ctx.task_id.is_nil())
            )
            return not (is_actor or is_task)
        except Exception:
            return True

    def timer(self):
        # torch.cuda.synchronize()
        # return time.perf_counter()
        return time.time()

    def count_mem(self):
        return torch.cuda.max_memory_allocated() / (1024**2)

    def reset_peak_memory_stats(self):
        torch.cuda.reset_peak_memory_stats()

    def start(self, tag: str = "default", config: Optional[Dict] = None):
        self.dict[tag] = {}
        self.dict[tag]["start_time"] = self.timer()
        self.dict[tag]["config"] = config

    def end(self, results, tag: str = "default", ranks: Optional[list] = None):
        if not self.log_setup:
            self._setup_logger()
        if tag not in self.dict:
            raise KeyError(f"Tag '{tag}' not found in profiler records")

        times = []
        mems = []

        stages = ["encode", "diffusion", "vae"]
        stage_times = defaultdict(list)
        stage_avg_times = {}
        stage_max_times = {}
        logger.debug(f"task_id is {tag}, results is {results}")
        for r in results:
            logger.debug(f" come into results, is {r}")
            exec_time = r["after"] - r["before"]
            times.append(exec_time)

            if "store" in r:
                self.dict[tag]["end_time"] = r["store"]
            memory = None
            if r["inner_results"]:
                memory = []
                for stage in stages:
                    if stage in r["inner_results"]:
                        stage_times[stage].append(r["inner_results"][stage])
                    else:
                        stage_times[stage].append(-1)
                    if f"{stage}_mem" in r["inner_results"]:
                        memory.append(r["inner_results"][f"{stage}_mem"])
                    else:
                        memory.append(-1)
            mems.append(max(memory) if memory else 0)

        max_time = max(times)
        avg_time = sum(times) / len(times)
        max_mem = max(mems)
        avg_mem = sum(mems) / len(mems)

        for stage in stages:
            if stage_times[stage]:
                stage_max_times[stage] = max(stage_times[stage])
                stage_avg_times[stage] = sum(stage_times[stage]) / len(
                    stage_times[stage]
                )

        request_total_time = self.dict[tag]["end_time"] - self.dict[tag]["start_time"]
        log_parts = [
            "",
            f"Task '{tag}' Completed",
            f"Task Config: {self.dict[tag]['config']}",
            f"Worker_num: {len(results)}",
            "Time Summary:",
            f"  - Request total time : {request_total_time:.4f} s",
            f"  - Max execution time : {max_time:.4f} s"
            f"({', '.join([f'max {stage} time : {stage_max_times[stage]:.4f} s' for stage in stages])})",
            f"  - Avg execution time : {avg_time:.4f} s"
            f"({', '.join([f'avg {stage} time : {stage_avg_times[stage]:.4f} s' for stage in stages])})",
            "GPU Memory Usage(among workers):",
            f" - Max Allocated memory : {max_mem / 1024**2:.3f} MB",
            f" - Avg Allocated memory : {avg_mem / 1024**2:.3f} MB",
        ]
        if ranks is not None:
            log_parts.append(f"Worker rank list: {ranks}")
        self.logger.info("\n".join(log_parts))
        self._cleanup_tag(tag)
        for handler in self.logger.handlers:
            handler.flush()

    def _cleanup_tag(self, tag: str):
        self.dict.pop(tag, None)

    def _reset_state(self):
        self.dict = {}

    def reset(self):
        self._reset_state()

    def _setup_logger(self):
        self.logger = logging.getLogger(f"{id(self)}.{self.name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(
            self.log_file, mode="a" if not self.overwrite else "w"
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            )
        )
        self.logger.addHandler(file_handler)
        self.log_setup = True


class DummyProfiler:
    def timer(self, *args, **kwargs):
        return 0

    def count_mem(self, *args, **kwargs):
        return 0

    def reset_peak_memory_stats(self, *args, **kwargs):
        pass

    def start(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass

    def reset(self):
        pass


def create_profiler(**kwargs):
    enabled = os.getenv("ENABLE_PROFILING", "0") == "1"
    if enabled:
        return Profiler(**kwargs)
    else:
        return DummyProfiler()


LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", None)
global_profiler = create_profiler(overwrite=False, log_file=LOG_FILE_PATH)
