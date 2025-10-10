import asyncio
from typing import Dict, Any
from hetu_dit.executor.gpu_executor import RayGPUExecutorAsync
from hetu_dit.logger import init_logger

logger = init_logger(__name__)


class WorkerMonitor:
    """Monitor the states of workers in the distributed system."""

    def __init__(self, executer: "RayGPUExecutorAsync", interval=2):
        self.executer = executer
        self.interval = interval
        self.meta_data: Dict[str, Dict[str, Any]] = {}
        self.monitor_task = None

    async def refresh(self):
        logger.debug("Refreshing worker states...")
        results = await self.executer.detect_worker_meta()

        organized_results = self.organize_worker_states(results)
        self.meta_data = organized_results

    def organize_worker_states(self, worker_states):
        # sorting by rank
        sorted_states = sorted(worker_states, key=lambda x: x.get("rank", float("inf")))
        organized_states = {
            f"Worker{x.get('rank', '_unknown')}": {
                "running_task": x.get("running_task", None),
                "last_detect_time": x.get("last_detected_time", None),
                "task_start_time": x.get("task_start_time", None),
                "task_running_time": x.get("task_running_time", None),
                "memory": x.get("memory", None),
                "state": x.get("state", None),
                "running_stage": x.get("running_stage", None),
                "stage_running_time": x.get("stage_running_time", None),
                "estimated_running_time": x.get("estimated_running_time", None),
                "estimated_idle_time": x.get("estimated_idle_time", None),
            }
            for x in sorted_states
        }
        return organized_states

    def get_worker_states(self):
        """result example
        {
        Worker1:{
        'running_task': 'task-1745632609855',
        'last_detect_time': 207715.385057355,
        'task_start_time': 207711.191355591,
        'task_running_time': 4.19370176398661,
        'memory': 17665384960,
        'state': 'busy',
        'running_stage': 'Diffusion',
        'stage_running_time': 2.511513876990648,
        'estimated_running_time': 4.19370176398661,
        'estimated_idle_time': 1.6721882360133887,
        }
        """
        return self.meta_data

    async def start_monitoring(self):
        logger.info("Starting monitoring...")
        while True:
            await self.refresh()
            await asyncio.sleep(self.interval)
