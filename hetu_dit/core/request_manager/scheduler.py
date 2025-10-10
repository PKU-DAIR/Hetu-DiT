from abc import ABC, abstractmethod
import asyncio
import heapq
from typing import Any, Tuple, Dict, List, Optional
import time
import random
from hetu_dit.core.request_manager.efficient_ilp import select_tasks
from hetu_dit.core.request_manager.multi_machine_efficient_ilp import (
    select_tasks_multi,
    EFF_TH,
    K_SET,
)
import copy
import re
from hetu_dit.logger import init_logger


logger = init_logger(__name__)


def _sync_parallel_config(parallel_config):
    """Sync aggregated degrees and world size after modifying sub-configs."""
    parallel_config.dp_degree = parallel_config.dp_config.dp_degree
    parallel_config.cfg_degree = parallel_config.dp_config.cfg_degree
    parallel_config.sp_degree = parallel_config.sp_config.sp_degree
    parallel_config.tp_degree = parallel_config.tp_config.tp_degree
    parallel_config.pp_degree = parallel_config.pp_config.pp_degree
    parallel_config.ulysses_degree = parallel_config.sp_config.ulysses_degree
    parallel_config.ring_degree = parallel_config.sp_config.ring_degree
    parallel_config.world_size = (
        parallel_config.dp_degree
        * parallel_config.cfg_degree
        * parallel_config.sp_degree
        * parallel_config.tp_degree
        * parallel_config.pp_degree
    )
    return parallel_config


def _configure_splitk_parallel(parallel_config, k):
    """Configure the parallel config for split-k style strategies."""
    parallel_config.dp_config.dp_degree = 1
    parallel_config.sp_config.ulysses_degree = k
    parallel_config.sp_config.ring_degree = 1
    parallel_config.tp_config.tp_degree = 1
    parallel_config.pp_config.pp_degree = 1
    parallel_config.sp_config.sp_degree = k
    return _sync_parallel_config(parallel_config)


def _configure_random_parallel(
    parallel_config, ulysses_degree, ring_degree, tp_degree, pp_degree
):
    """Configure the parallel config for random/ILP fix search modes."""
    parallel_config.dp_config.dp_degree = 1
    parallel_config.sp_config.ulysses_degree = ulysses_degree
    parallel_config.sp_config.ring_degree = ring_degree
    parallel_config.tp_config.tp_degree = tp_degree
    parallel_config.pp_config.pp_degree = pp_degree
    parallel_config.sp_config.sp_degree = ulysses_degree * ring_degree
    return _sync_parallel_config(parallel_config)


def _select_machine_with_capacity(m_free: List[int], required: int) -> Optional[int]:
    """Return machine index with sufficient free GPUs and maximum remaining capacity."""
    capable = [idx for idx, free in enumerate(m_free) if free >= required]
    if not capable:
        return None
    max_free = max(m_free[idx] for idx in capable)
    best_candidates = [idx for idx in capable if m_free[idx] == max_free]
    return min(best_candidates)


def _numeric_suffix(task_id: str) -> int | None:
    """
    Parse task-<idx> or task-<idx>_<...> → return idx (int)
    Return None if failed
    """
    # Capture the number immediately after "task-", followed by "_" or end of string
    m = re.search(r"task-(\d+)(?:_|$)", task_id)
    return int(m.group(1)) if m else None


# ================ use for scheduler==========================


def make_profile_key(input_cfg) -> str:
    """
    Generate profile key from InputConfig
    Required fields: height, width, num_frames

    Example
    -------
    In : input_cfg.height=768, input_cfg.width=432, input_cfg.num_frames=33
    Out: "768-432-33"
    """
    return f"{input_cfg.height}-{input_cfg.width}-{input_cfg.num_frames}"


def estimate_ddl(t_dict: dict[int, float]) -> int:
    """
    Given {k: tk}, find the maximum parallelism k★ with efficiency ≥0.8,
    then return ddl = t_{k★} * 2 (rounded up to seconds)

    Parameters
    ----------
    t_dict : {1: t1, 2: t4, 4: t4, 8: t8}

    Returns
    -------
    ddl_sec : int   # seconds relative to now
    """
    t1 = t_dict[1]
    best_k = 1
    for k, tk in t_dict.items():
        if (t1 / tk) / k >= 0.8 and k > best_k:
            best_k = k
    ddl = int(round(t_dict[best_k] * 4) * 1.1) + 2
    return ddl


class SchedulingStrategy(ABC):
    """Interface for scheduling strategies that operate on the shared queue."""

    @abstractmethod
    async def put(self, queue: List[Any], priority: int, item: Any, *args, **kwargs):
        """Insert an item into the scheduling queue."""

    @abstractmethod
    async def get(self, queue: List[Any], *args, **kwargs) -> Tuple[int, Any]:
        """Retrieve the next schedulable item from the queue."""


class PriorityStrategy(SchedulingStrategy):
    """Classic priority queue scheduling strategy."""

    def __init__(self) -> None:
        self._counter = 0

    async def put(
        self,
        queue: List[Tuple[int, int, Any]],
        priority: int,
        item: Any,
        *_: Any,
        **__: Any,
    ):
        entry = (priority, self._counter, item)
        heapq.heappush(queue, entry)
        self._counter += 1

    async def get(
        self, queue: List[Tuple[int, int, Any]], *_: Any, **__: Any
    ) -> Tuple[int, Any]:
        if not queue:
            raise asyncio.QueueEmpty
        priority, _, item = heapq.heappop(queue)
        return priority, item


class FIFOStrategy(SchedulingStrategy):
    """First-in-first-out strategy that ignores explicit priorities."""

    async def put(
        self, queue: List[Tuple[int, Any]], priority: int, item: Any, *_: Any, **__: Any
    ):
        queue.append((priority, item))

    async def get(
        self, queue: List[Tuple[int, Any]], *_: Any, **__: Any
    ) -> Tuple[int, Any]:
        if not queue:
            raise asyncio.QueueEmpty
        return queue.pop(0)


class ILP_fix_Strategy(SchedulingStrategy):
    def __init__(
        self, alpha_coef=0.5, beta_coef=0.5, max_diff=10, max_m=10, lambda_val=0.1
    ):
        self.alpha_coef = alpha_coef
        self.beta_coef = beta_coef
        self.max_diff = max_diff
        self.max_m = max_m
        self.lambda_val = lambda_val
        self.last_schedule_time = 0
        self.schedule_interval = 5
        self.scheduled_tasks = []
        self.M = 8
        self.machines = [
            {"id": i, "available_at": 0.0} for i in range(self.M)
        ]  # Initialize machine status
        # Allowed machine count options
        self.allowed_machine_counts = [1, 2, 4, 8]
        # Priority-machine-t_jk-parallel strategy table
        self.priority_k_table = {
            (16384, 1): (8.997584, (1, 1, 1, 1)),
            (16384, 2): (8.221663, (1, 1, 1, 2)),
            (16384, 4): (8.837153, (1, 1, 1, 4)),
            (16384, 8): (9.999092, (1, 1, 1, 8)),
            (32768, 1): (8.634759, (1, 1, 1, 1)),
            (32768, 2): (8.205663, (1, 1, 1, 2)),
            (32768, 4): (8.713233, (1, 1, 1, 4)),
            (32768, 8): (9.763339, (1, 1, 1, 8)),
            (65536, 1): (7.419797, (1, 1, 1, 1)),
            (65536, 2): (8.022464, (1, 1, 1, 2)),
            (65536, 4): (8.634823, (1, 1, 1, 4)),
            (65536, 8): (9.650290, (1, 1, 1, 8)),
            (131072, 1): (7.613352, (1, 1, 1, 1)),
            (131072, 2): (7.963421, (1, 1, 1, 2)),
            (131072, 4): (8.602235, (1, 1, 1, 4)),
            (131072, 8): (9.612588, (1, 1, 1, 8)),
            (262144, 1): (7.629080, (1, 1, 1, 1)),
            (262144, 2): (8.046077, (1, 1, 1, 2)),
            (262144, 4): (8.653901, (1, 1, 1, 4)),
            (262144, 8): (9.629589, (1, 1, 1, 8)),
            (524288, 1): (11.878682, (1, 1, 1, 1)),
            (524288, 2): (9.718105, (1, 1, 1, 2)),
            (524288, 4): (8.826129, (1, 1, 1, 4)),
            (524288, 8): (10.172442, (1, 1, 1, 8)),
            (1048576, 1): (23.551868, (1, 1, 1, 1)),
            (1048576, 2): (15.072551, (1, 1, 1, 2)),
            (1048576, 4): (10.797386, (1, 1, 1, 4)),
            (1048576, 8): (10.829527, (1, 1, 1, 8)),
        }

    async def put(self, queue, priority: int, item: Any):
        task_id, input_config, engine_config, future, worker_ids = item
        q_min = 1
        r_j = time.time()
        d_j = r_j + random.randint(12, 30)
        t_jk = {}
        for k in self.allowed_machine_counts:
            t, _ = self.priority_k_table.get((priority, k), (8, (1, 1, 1, 8)))
            t_jk[k] = int(round(t))
        task = {
            "id": task_id,
            "q_min": q_min,
            "r_j": r_j,
            "d_j": d_j,
            "t_jk": t_jk,
            "priority": priority,
            "input_config": input_config,
            "engine_config": engine_config,
            "future": future,
            "worker_ids": worker_ids,
            "original_item": item,
        }
        queue.append(task)

    async def get(self, queue) -> Tuple[int, Any]:
        if not queue:
            raise asyncio.QueueEmpty

        current_time = time.time()

        # Check if scheduling is needed
        if (
            current_time - self.last_schedule_time >= self.schedule_interval
            or not self.scheduled_tasks
        ):
            self.last_schedule_time = current_time

            # Only process arrived tasks
            available_tasks = [task for task in queue if task["r_j"] <= current_time]

            if available_tasks:
                # Calculate task scores
                scores = self.parameterized_compute_scores(
                    available_tasks, current_time
                )

                # Sort tasks by score
                sorted_tasks_scores = sorted(
                    zip(available_tasks, scores), key=lambda x: x[1], reverse=True
                )

                # Select subset of tasks
                selected_tasks = self.parameterized_select_subset(sorted_tasks_scores)

                if selected_tasks:
                    # Calculate time window
                    time_window = self.compute_time_window(
                        selected_tasks, self.M, current_time
                    )

                    # Call ILP algorithm for scheduling
                    schedule, solve_time = self.parameterized_solve_ilp(
                        selected_tasks, current_time, time_window
                    )
                    logger.info(
                        f"[Scheduler] ILP solved in {solve_time:.2f}s, scheduled {len(schedule)} tasks"
                    )

                    # Key fix: sort by start time ascending
                    schedule.sort(key=lambda x: x["start_time"])

                    # Update scheduled task list
                    self.scheduled_tasks = []
                    for sched_item in schedule:
                        task_id = sched_item["task_id"]
                        num_required_machines = sched_item["k"]
                        task_actual_start_time = sched_item[
                            "start_time"
                        ]  # Absolute time from ILP

                        # Find corresponding task
                        task_in_queue = None
                        for t_obj in available_tasks:
                            if t_obj["id"] == task_id:
                                task_in_queue = t_obj
                                break

                        if task_in_queue is None:
                            # Should not happen if ILP schedules tasks from available_tasks
                            logger.debug(
                                f"[Scheduler] Error: Task {task_id} from schedule not found in available tasks."
                            )
                            continue

                        duration = task_in_queue["t_jk"][num_required_machines]
                        task_completion_time = task_actual_start_time + duration

                        # Assign worker_ids based on scheduling result
                        # Filter machines available before task start
                        eligible_machines = [
                            m
                            for m in self.machines
                            if m["available_at"] <= task_actual_start_time
                        ]
                        # Sort by available time ascending, then by ID
                        eligible_machines.sort(
                            key=lambda m: (m["available_at"], m["id"])
                        )

                        if len(eligible_machines) < num_required_machines:
                            logger.warning(
                                f"[Scheduler] Warning: Task {task_in_queue['id']} requires {num_required_machines} machines, "
                                f"but only {len(eligible_machines)} are eligible by {task_actual_start_time:.2f}. "
                                f"This might indicate an issue with scheduling or resource contention."
                            )
                            # Ensure assigned machine count is in allowed list
                            available_count = len(eligible_machines)
                            allowed_count = max(
                                [
                                    k
                                    for k in self.allowed_machine_counts
                                    if k <= available_count
                                ],
                                default=0,
                            )
                            if allowed_count > 0:
                                assigned_machines = eligible_machines[:allowed_count]
                            else:
                                logger.warning(
                                    f"[Scheduler] Warning: No allowed machine count ({self.allowed_machine_counts}) available for task {task_in_queue['id']}."
                                )
                                assigned_machines = []
                            if not assigned_machines and num_required_machines > 0:
                                logger.error(
                                    f"[Scheduler] Critical Error: No machines available for task {task_in_queue['id']} requiring {num_required_machines}."
                                )
                                # Skip this task or handle error appropriately
                                continue
                        else:
                            # Even if enough machines, ensure using allowed machine count
                            if num_required_machines in self.allowed_machine_counts:
                                assigned_machines = eligible_machines[
                                    :num_required_machines
                                ]
                            else:
                                allowed_count = max(
                                    [
                                        k
                                        for k in self.allowed_machine_counts
                                        if k <= num_required_machines
                                    ],
                                    default=self.allowed_machine_counts[0],
                                )
                                assigned_machines = eligible_machines[:allowed_count]
                                logger.warning(
                                    f"[Scheduler] Adjusting: Task {task_in_queue['id']} requires {num_required_machines} machines, "
                                    f"but using allowed count {allowed_count} instead."
                                )

                        selected_machine_ids = [m["id"] for m in assigned_machines]
                        task_in_queue["worker_ids"] = (
                            selected_machine_ids  # Update worker_ids for the task
                        )

                        # Update available time for selected machines
                        for m_obj in assigned_machines:
                            m_obj["available_at"] = task_completion_time

                        # Add to scheduled task list
                        self.scheduled_tasks.append(task_in_queue)

        if self.scheduled_tasks:
            # Pop the first scheduled task
            task_to_return = self.scheduled_tasks.pop(0)

            if task_to_return in queue:  # Check if it's still in the main queue
                queue.remove(task_to_return)

            # Update worker_ids in original_item
            original_item_tuple = task_to_return["original_item"]
            updated_original_item = (
                original_item_tuple[0],  # task_id
                original_item_tuple[1],  # input_config
                original_item_tuple[2],  # engine_config
                original_item_tuple[3],  # future
                task_to_return["worker_ids"],
            )  # newly assigned worker_ids

            return task_to_return["priority"], updated_original_item
        else:
            # No scheduled task ready to be returned
            raise asyncio.QueueEmpty

    def parameterized_compute_scores(self, tasks, current_time):
        slack_times = []
        wait_times = []
        for task in tasks:
            r_j = task["r_j"]
            d_j = task["d_j"]
            remaining = max(0, d_j - current_time)
            slack = (
                remaining - (current_time - r_j) if current_time >= r_j else remaining
            )
            slack = max(slack, 1e-5)
            wait = max(current_time - r_j, 0)
            slack_times.append(slack)
            wait_times.append(wait)

        avg_slack = sum(slack_times) / len(tasks) if slack_times else 0.1
        avg_wait = sum(wait_times) / len(tasks) if wait_times else 0
        alpha = self.alpha_coef / (avg_slack + 0.1)
        beta = self.beta_coef * avg_wait

        scores = []
        for task, st, wt in zip(tasks, slack_times, wait_times):
            scores.append(alpha * (1 / st) + beta * wt)
        return scores

    def parameterized_select_subset(self, sorted_tasks_scores):
        if not sorted_tasks_scores:
            return []
        m = 1
        for i in range(1, len(sorted_tasks_scores)):
            score_diff = sorted_tasks_scores[i - 1][1] - sorted_tasks_scores[i][1]
            if score_diff > self.max_diff:
                m = i
                break
            m += 1
        m = min(m, self.max_m)
        return [task for task, _ in sorted_tasks_scores[:m]]

    def compute_time_window(self, selected_tasks, M, current_time):
        l = [current_time] * M
        prev_s = current_time
        for task in selected_tasks:
            A = sorted(l)
            best_c = float("inf")
            best_k = None
            # Only consider allowed machine counts
            for k in self.allowed_machine_counts:
                if k < task["q_min"] or k > len(A):
                    continue
                if k not in task["t_jk"]:
                    continue
                s = max(prev_s, A[k - 1])
                c = s + task["t_jk"][k]
                if c < best_c:
                    best_c = c
                    best_k = k
            prev_s = best_c
            for i in range(best_k):
                A[i] = best_c
            l = sorted(A)
        return max(
            30, int(1.5 * (prev_s - current_time))
        )  # Return at least 30 seconds time window

    def parameterized_solve_ilp(self, subset, current_time, T):
        import pulp

        model = pulp.LpProblem("Scheduling", pulp.LpMaximize)
        task_map = {task["id"]: task for task in subset}
        L = 1e5  # A large constant

        # Decision variable definition - create variables only for allowed machine counts
        x_vars = pulp.LpVariable.dicts(
            "x",
            [
                (j["id"], t, k)
                for j in subset
                for k in self.allowed_machine_counts
                if k >= j["q_min"] and k <= self.M and k in j["t_jk"]
                for t in range(
                    max(0, int(j["r_j"] - current_time)), T - j["t_jk"][k] + 1
                )
            ],
            cat="Binary",
        )

        # Auxiliary variables
        z_vars = pulp.LpVariable.dicts("z", [j["id"] for j in subset], cat="Binary")
        s_vars = pulp.LpVariable.dicts("s", [j["id"] for j in subset], lowBound=0)

        # Objective function
        model += pulp.lpSum(
            z_vars[j["id"]] for j in subset
        ) - self.lambda_val * pulp.lpSum(s_vars[j["id"]] for j in subset)

        # Constraints
        for j in subset:
            j_id = j["id"]
            q_min = j["q_min"]
            d_j = j["d_j"]

            # Constraint 1: Must be assigned once
            # Ensure only allowed machine counts are considered
            allowed_k_values = [
                k
                for k in self.allowed_machine_counts
                if k >= q_min and k <= self.M and k in j["t_jk"]
            ]
            model += (
                pulp.lpSum(
                    x_vars.get((j_id, t, k), 0)
                    for k in allowed_k_values
                    for t in range(
                        max(0, int(j["r_j"] - current_time)), T - j["t_jk"][k] + 1
                    )
                )
                == 1
            )

            # Completion time calculation
            c_j = current_time + pulp.lpSum(
                (t + j["t_jk"][k]) * x_vars.get((j_id, t, k), 0)
                for k in allowed_k_values
                for t in range(
                    max(0, int(j["r_j"] - current_time)), T - j["t_jk"][k] + 1
                )
            )

            # Constraint 3: On-time completion determination
            model += c_j <= d_j + L * (1 - z_vars[j_id])
            model += c_j >= d_j - L * z_vars[j_id]

            # Constraint 4: Overtime amount definition
            model += s_vars[j_id] >= c_j - d_j
            model += s_vars[j_id] <= L * (1 - z_vars[j_id])

        # Constraint 5: Machine occupancy
        for t_rel in range(T + 1):
            total = 0
            for j in subset:
                j_id = j["id"]
                # Only consider allowed machine counts
                for k in self.allowed_machine_counts:
                    if k < j["q_min"] or k > self.M or k not in j["t_jk"]:
                        continue
                    for t in range(
                        max(0, int(j["r_j"] - current_time)), T - j["t_jk"][k] + 1
                    ):
                        if t <= t_rel < t + j["t_jk"][k]:
                            if (j_id, t, k) in x_vars:
                                total += k * x_vars[(j_id, t, k)]
            model += total <= self.M

        # Add solver time limit
        solver = pulp.PULP_CBC_CMD(msg=False)
        # Add solving time statistics
        start_time = time.time()
        status = model.solve(solver)
        solve_time = time.time() - start_time  # Record solving time

        if status != pulp.LpStatusOptimal:
            return [], solve_time  # Return empty list and time

        schedule = []
        for j_id, t, k in x_vars:
            if (j_id, t, k) in x_vars and x_vars[(j_id, t, k)].value() == 1:
                schedule.append(
                    {"task_id": j_id, "start_time": current_time + t, "k": k}
                )
        return schedule, solve_time


class ILP_random_Strategy(SchedulingStrategy):
    def __init__(
        self, alpha_coef=0.5, beta_coef=0.5, max_diff=5, max_m=10, lambda_val=0.1
    ):
        self.alpha_coef = alpha_coef
        self.beta_coef = beta_coef
        self.max_diff = max_diff
        self.max_m = max_m
        self.lambda_val = lambda_val
        self.last_schedule_time = 0
        self.schedule_interval = 5
        self.scheduled_tasks = []
        self.M = 8
        self.machines = [{"id": i, "available_at": 0.0} for i in range(self.M)]
        self.allowed_machine_counts = [1, 2, 4, 8]
        self.priority_k_table = {
            (1296, 1): (0.384615385, (1, 1, 1, 1)),
            (1296, 2): (0.262467192, (2, 1, 1, 1)),
            (1296, 4): (0.225733634, (4, 1, 1, 1)),
            (1296, 8): (0.158730159, (8, 1, 1, 1)),
            (81600, 1): (30.5, (1, 1, 1, 1)),
            (81600, 2): (17.75, (2, 1, 1, 1)),
            (81600, 4): (9.1, (4, 1, 1, 1)),
            (81600, 8): (4.75, (8, 1, 1, 1)),
            (32640, 1): (9.37, (1, 1, 1, 1)),
            (32640, 2): (5.63, (2, 1, 1, 1)),
            (32640, 4): (3.0, (4, 1, 1, 1)),
            (32640, 8): (1.57, (8, 1, 1, 1)),
            (65280, 1): (22.14, (1, 1, 1, 1)),
            (65280, 2): (13.07, (2, 1, 1, 1)),
            (65280, 4): (6.77, (4, 1, 1, 1)),
            (65280, 8): (3.56, (8, 1, 1, 1)),
            (25920, 1): (5.72, (1, 1, 1, 1)),
            (25920, 2): (3.51, (2, 1, 1, 1)),
            (25920, 4): (1.89, (4, 1, 1, 1)),
            (25920, 8): (1.01, (8, 1, 1, 1)),
            (20736, 1): (4.42, (1, 1, 1, 1)),
            (20736, 2): (2.76, (2, 1, 1, 1)),
            (20736, 4): (1.49, (4, 1, 1, 1)),
            (20736, 8): (0.826446281, (8, 1, 1, 1)),
            (10368, 1): (2.16, (1, 1, 1, 1)),
            (10368, 2): (0.751879699, (2, 1, 1, 1)),
            (10368, 4): (0.751879699, (4, 1, 1, 1)),
            (10368, 8): (0.452488688, (8, 1, 1, 1)),
            (5184, 1): (1.19, (1, 1, 1, 1)),
            (5184, 2): (0.763358779, (2, 1, 1, 1)),
            (5184, 4): (0.45045045, (4, 1, 1, 1)),
            (5184, 8): (0.3, (8, 1, 1, 1)),
            (4080, 1): (1.26, (1, 1, 1, 1)),
            (4080, 2): (0.78125, (2, 1, 1, 1)),
            (4080, 4): (0.483091787, (4, 1, 1, 1)),
            (4080, 8): (0.297619048, (8, 1, 1, 1)),
            (16320, 1): (4.71, (1, 1, 1, 1)),
            (16320, 2): (2.91, (2, 1, 1, 1)),
            (16320, 4): (1.55, (4, 1, 1, 1)),
            (16320, 8): (0.840336134, (8, 1, 1, 1)),
        }

    async def put(self, queue, priority: int, item: Any):
        task_id, input_config, engine_config = item
        q_min = 1
        r_j = time.time()
        d_j = r_j + 20
        t_jk = {}
        for k in self.allowed_machine_counts:
            t, _ = self.priority_k_table.get((priority, k), (8, (1, 1, 1, 1)))
            t_jk[k] = int(round(t))
        task = {
            "id": task_id,
            "q_min": q_min,
            "r_j": r_j,
            "d_j": d_j,
            "t_jk": t_jk,
            "priority": priority,
            "input_config": input_config,
            "engine_config": engine_config,
            "original_item": item,
        }
        queue.append(task)

    async def get(self, queue) -> Tuple[int, Any]:
        logger.debug(f"[Scheduler] enter ilp random Current queue size: {len(queue)}")
        if not queue:
            raise asyncio.QueueEmpty

        current_time = time.time()

        if (
            current_time - self.last_schedule_time >= self.schedule_interval
            or not self.scheduled_tasks
        ):
            self.last_schedule_time = current_time
            available_tasks = [task for task in queue if task["r_j"] <= current_time]

            if available_tasks:
                scores = self.parameterized_compute_scores(
                    available_tasks, current_time
                )
                sorted_tasks_scores = sorted(
                    zip(available_tasks, scores), key=lambda x: x[1], reverse=True
                )
                selected_tasks = self.parameterized_select_subset(sorted_tasks_scores)

                if selected_tasks:
                    time_window = self.compute_time_window(
                        selected_tasks, self.M, current_time
                    )
                    schedule, solve_time = self.parameterized_solve_ilp(
                        selected_tasks, current_time, time_window
                    )
                    logger.info(
                        f"[Scheduler] ILP solved in {solve_time:.2f}s, scheduled {len(schedule)} tasks"
                    )

                    self.scheduled_tasks = []
                    for sched_item in schedule:
                        task_id = sched_item["task_id"]
                        num_required_machines = sched_item["k"]
                        task_actual_start_time = sched_item[
                            "start_time"
                        ]  # Absolute time from ILP

                        task_in_queue = None
                        for t_obj in available_tasks:
                            if t_obj["id"] == task_id:
                                task_in_queue = t_obj
                                break

                        if task_in_queue is None:
                            # Should not happen if ILP schedules tasks from available_tasks
                            logger.error(
                                f"[Scheduler] Error: Task {task_id} from schedule not found in available tasks."
                            )
                            continue

                        if num_required_machines in self.allowed_machine_counts:
                            assigned_count = num_required_machines
                        else:
                            assigned_count = max(
                                [
                                    k
                                    for k in self.allowed_machine_counts
                                    if k <= num_required_machines
                                ],
                                default=self.allowed_machine_counts[0],
                            )
                            logger.warning(
                                f"[Scheduler] Warning: Task {task_id} required {num_required_machines} machines, "
                                f"but adjusting to allowed count {assigned_count}."
                            )

                        task_in_queue["assigned_machine_count"] = assigned_count
                        self.scheduled_tasks.append(task_in_queue)

                    for task in self.scheduled_tasks:
                        if task in queue:
                            queue.remove(task)

        if self.scheduled_tasks:
            task_to_return = self.scheduled_tasks.pop(0)
            return task_to_return["priority"], (
                task_to_return["original_item"],
                task_to_return["assigned_machine_count"],
            )
        else:
            raise asyncio.QueueEmpty

    def parameterized_compute_scores(self, tasks, current_time):
        slack_times = []
        wait_times = []
        for task in tasks:
            r_j = task["r_j"]
            d_j = task["d_j"]
            remaining = max(0, d_j - current_time)
            slack = (
                remaining - (current_time - r_j) if current_time >= r_j else remaining
            )
            slack = max(slack, 1e-5)
            wait = max(current_time - r_j, 0)
            slack_times.append(slack)
            wait_times.append(wait)

        avg_slack = sum(slack_times) / len(tasks) if slack_times else 0.1
        avg_wait = sum(wait_times) / len(tasks) if wait_times else 0
        alpha = self.alpha_coef / (avg_slack + 0.1)
        beta = self.beta_coef * avg_wait

        scores = []
        for task, st, wt in zip(tasks, slack_times, wait_times):
            scores.append(alpha * (1 / st) + beta * wt)
        return scores

    def parameterized_select_subset(self, sorted_tasks_scores):
        if not sorted_tasks_scores:
            return []
        m = 1
        for i in range(1, len(sorted_tasks_scores)):
            score_diff = sorted_tasks_scores[i - 1][1] - sorted_tasks_scores[i][1]
            if score_diff > self.max_diff:
                m = i
                break
            m += 1
        m = min(m, self.max_m)
        return [task for task, _ in sorted_tasks_scores[:m]]

    def compute_time_window(self, selected_tasks, M, current_time):
        l = [current_time] * M
        prev_s = current_time
        for task in selected_tasks:
            A = sorted(l)
            best_c = float("inf")
            best_k = None
            # Only consider allowed machine counts
            for k in self.allowed_machine_counts:
                if k < task["q_min"] or k > len(A):
                    continue
                if k not in task["t_jk"]:
                    continue
                s = max(prev_s, A[k - 1])
                c = s + task["t_jk"][k]
                if c < best_c:
                    best_c = c
                    best_k = k
            prev_s = best_c
            for i in range(best_k):
                A[i] = best_c
            l = sorted(A)
        return max(
            30, int(1.5 * (prev_s - current_time))
        )  # Return at least 30 seconds time window

    def parameterized_solve_ilp(self, subset, current_time, T):
        import pulp

        model = pulp.LpProblem("Scheduling", pulp.LpMaximize)
        task_map = {task["id"]: task for task in subset}
        L = 1e5  # A large constant

        # Decision variable definition - create variables only for allowed machine counts
        x_vars = pulp.LpVariable.dicts(
            "x",
            [
                (j["id"], t, k)
                for j in subset
                for k in self.allowed_machine_counts
                if k >= j["q_min"] and k <= self.M and k in j["t_jk"]
                for t in range(
                    max(0, int(j["r_j"] - current_time)), T - j["t_jk"][k] + 1
                )
            ],
            cat="Binary",
        )

        # Auxiliary variables
        z_vars = pulp.LpVariable.dicts("z", [j["id"] for j in subset], cat="Binary")
        s_vars = pulp.LpVariable.dicts("s", [j["id"] for j in subset], lowBound=0)

        # Objective function
        model += pulp.lpSum(
            z_vars[j["id"]] for j in subset
        ) - self.lambda_val * pulp.lpSum(s_vars[j["id"]] for j in subset)

        # Constraints
        for j in subset:
            j_id = j["id"]
            q_min = j["q_min"]
            d_j = j["d_j"]

            # Constraint 1: Must be assigned once
            # Ensure only allowed machine counts are considered
            allowed_k_values = [
                k
                for k in self.allowed_machine_counts
                if k >= q_min and k <= self.M and k in j["t_jk"]
            ]
            model += (
                pulp.lpSum(
                    x_vars.get((j_id, t, k), 0)
                    for k in allowed_k_values
                    for t in range(
                        max(0, int(j["r_j"] - current_time)), T - j["t_jk"][k] + 1
                    )
                )
                == 1
            )

            # Completion time calculation
            c_j = current_time + pulp.lpSum(
                (t + j["t_jk"][k]) * x_vars.get((j_id, t, k), 0)
                for k in allowed_k_values
                for t in range(
                    max(0, int(j["r_j"] - current_time)), T - j["t_jk"][k] + 1
                )
            )

            # Constraint 3: On-time completion determination
            model += c_j <= d_j + L * (1 - z_vars[j_id])
            model += c_j >= d_j - L * z_vars[j_id]

            # Constraint 4: Overtime amount definition
            model += s_vars[j_id] >= c_j - d_j
            model += s_vars[j_id] <= L * (1 - z_vars[j_id])

        # Constraint 5: Machine occupancy
        for t_rel in range(T + 1):
            total = 0
            for j in subset:
                j_id = j["id"]
                # Only consider allowed machine counts
                for k in self.allowed_machine_counts:
                    if k < j["q_min"] or k > self.M or k not in j["t_jk"]:
                        continue
                    for t in range(
                        max(0, int(j["r_j"] - current_time)), T - j["t_jk"][k] + 1
                    ):
                        if t <= t_rel < t + j["t_jk"][k]:
                            if (j_id, t, k) in x_vars:
                                total += k * x_vars[(j_id, t, k)]
            model += total <= self.M

        # Add solver time limit
        solver = pulp.PULP_CBC_CMD(msg=False)
        # Add solving time statistics
        start_time = time.time()
        status = model.solve(solver)
        solve_time = time.time() - start_time  # Record solving time

        if status != pulp.LpStatusOptimal:
            return [], solve_time  # Return empty list and time

        schedule = []
        for j_id, t, k in x_vars:
            if (j_id, t, k) in x_vars and x_vars[(j_id, t, k)].value() == 1:
                schedule.append(
                    {"task_id": j_id, "start_time": current_time + t, "k": k}
                )
        return schedule, solve_time


class GreedyRandomStrategy(SchedulingStrategy):
    """Greedy scheduling strategy (random mode): prioritize scheduling tasks with the least remaining time"""

    def __init__(self, schedule_interval=1):
        self.schedule_interval = schedule_interval
        self.last_schedule_time = 0
        self.M = 8
        self.allowed_machine_counts = [1, 2, 4, 8]
        # Priority-machine-t_jk-parallel strategy table
        self.priority_k_table = {
            (16384, 1): (8.997584, (1, 1, 1, 1)),
            (16384, 2): (8.221663, (1, 1, 1, 2)),
            (16384, 4): (8.837153, (1, 1, 1, 4)),
            (16384, 8): (9.999092, (1, 1, 1, 8)),
            (32768, 1): (8.634759, (1, 1, 1, 1)),
            (32768, 2): (8.205663, (1, 1, 1, 2)),
            (32768, 4): (8.713233, (1, 1, 1, 4)),
            (32768, 8): (9.763339, (1, 1, 1, 8)),
            (65536, 1): (7.419797, (1, 1, 1, 1)),
            (65536, 2): (8.022464, (1, 1, 1, 2)),
            (65536, 4): (8.634823, (1, 1, 1, 4)),
            (65536, 8): (9.650290, (1, 1, 1, 8)),
            (131072, 1): (7.613352, (1, 1, 1, 1)),
            (131072, 2): (7.963421, (1, 1, 1, 2)),
            (131072, 4): (8.602235, (1, 1, 1, 4)),
            (131072, 8): (9.612588, (1, 1, 1, 8)),
            (262144, 1): (7.629080, (1, 1, 1, 1)),
            (262144, 2): (8.046077, (1, 1, 1, 2)),
            (262144, 4): (8.653901, (1, 1, 1, 4)),
            (262144, 8): (9.629589, (1, 1, 1, 8)),
            (524288, 1): (11.878682, (1, 1, 1, 1)),
            (524288, 2): (9.718105, (1, 1, 1, 2)),
            (524288, 4): (8.826129, (1, 1, 1, 4)),
            (524288, 8): (10.172442, (1, 1, 1, 8)),
            (1048576, 1): (23.551868, (1, 1, 1, 1)),
            (1048576, 2): (15.072551, (1, 1, 1, 2)),
            (1048576, 4): (10.797386, (1, 1, 1, 4)),
            (1048576, 8): (10.829527, (1, 1, 1, 8)),
        }

    async def put(self, queue, priority: int, item: Any):
        task_id, input_config, engine_config = item
        q_min = 1
        r_j = time.time()
        d_j = r_j + 20
        t_jk = {}
        for k in self.allowed_machine_counts:
            t, _ = self.priority_k_table.get((priority, k), (8, (1, 1, 1, 1)))
            t_jk[k] = int(round(t))
        task = {
            "id": task_id,
            "q_min": q_min,
            "r_j": r_j,
            "d_j": d_j,
            "t_jk": t_jk,
            "priority": priority,
            "input_config": input_config,
            "engine_config": engine_config,
            "original_item": item,
        }
        queue.append(task)

    async def get(self, queue) -> Tuple[int, Any]:
        if not queue:
            raise asyncio.QueueEmpty

        current_time = time.time()
        available_tasks = [task for task in queue if task["r_j"] <= current_time]

        if not available_tasks:
            raise asyncio.QueueEmpty

        tasks_with_remaining_time = []
        for task in available_tasks:
            remaining_time = task["d_j"] - current_time
            tasks_with_remaining_time.append((remaining_time, task))

        tasks_with_remaining_time.sort(key=lambda x: x[0])
        _, selected_task = tasks_with_remaining_time[0]

        best_time = float("inf")
        best_k = 1

        for k in self.allowed_machine_counts:
            if k < selected_task["q_min"] or k > self.M:
                continue
            if k not in selected_task["t_jk"]:
                continue

            execution_time = selected_task["t_jk"][k]
            if execution_time < best_time:
                best_time = execution_time
                best_k = k

        selected_task["assigned_machine_count"] = best_k
        queue.remove(selected_task)

        return selected_task["priority"], (
            selected_task["original_item"],
            selected_task["assigned_machine_count"],
        )


class ILP_makespan_Strategy(SchedulingStrategy):
    def __init__(
        self, alpha_coef=0.5, beta_coef=0.5, max_diff=10, max_m=8, lambda_val=0.1
    ):
        self.alpha_coef = alpha_coef
        self.beta_coef = beta_coef
        self.max_diff = max_diff
        self.max_m = max_m
        self.lambda_val = lambda_val
        self.last_schedule_time = 0
        self.schedule_interval = 7
        self.scheduled_tasks = []
        self.M = 8
        self.machines = [{"id": i, "available_at": 0.0} for i in range(self.M)]
        self.allowed_machine_counts = [1, 2, 4, 8]
        self.priority_k_table = {
            (16384, 1): (8.997584, (1, 1, 1, 1)),
            (16384, 2): (8.221663, (1, 1, 1, 2)),
            (16384, 4): (8.837153, (1, 1, 1, 4)),
            (16384, 8): (9.999092, (1, 1, 1, 8)),
            (32768, 1): (8.634759, (1, 1, 1, 1)),
            (32768, 2): (8.205663, (1, 1, 1, 2)),
            (32768, 4): (8.713233, (1, 1, 1, 4)),
            (32768, 8): (9.763339, (1, 1, 1, 8)),
            (65536, 1): (7.419797, (1, 1, 1, 1)),
            (65536, 2): (8.022464, (1, 1, 1, 2)),
            (65536, 4): (8.634823, (1, 1, 1, 4)),
            (65536, 8): (9.650290, (1, 1, 1, 8)),
            (131072, 1): (7.613352, (1, 1, 1, 1)),
            (131072, 2): (7.963421, (1, 1, 1, 2)),
            (131072, 4): (8.602235, (1, 1, 1, 4)),
            (131072, 8): (9.612588, (1, 1, 1, 8)),
            (262144, 1): (7.629080, (1, 1, 1, 1)),
            (262144, 2): (8.046077, (1, 1, 1, 2)),
            (262144, 4): (8.653901, (1, 1, 1, 4)),
            (262144, 8): (9.629589, (1, 1, 1, 8)),
            (524288, 1): (11.878682, (1, 1, 1, 1)),
            (524288, 2): (9.718105, (1, 1, 1, 2)),
            (524288, 4): (8.826129, (1, 1, 1, 4)),
            (524288, 8): (10.172442, (1, 1, 1, 8)),
            (1048576, 1): (23.551868, (1, 1, 1, 1)),
            (1048576, 2): (15.072551, (1, 1, 1, 2)),
            (1048576, 4): (10.797386, (1, 1, 1, 4)),
            (1048576, 8): (10.829527, (1, 1, 1, 8)),
        }

    async def put(self, queue, priority: int, item: Any):
        task_id, input_config, engine_config, future, worker_ids = item
        q_min = 1
        r_j = time.time()
        d_j = r_j + 20
        t_jk = {}
        for k in self.allowed_machine_counts:
            t, _ = self.priority_k_table.get((priority, k), (8, (1, 1, 1, 1)))
            t_jk[k] = int(round(t))
        task = {
            "id": task_id,
            "q_min": q_min,
            "r_j": r_j,
            "d_j": d_j,
            "t_jk": t_jk,
            "priority": priority,
            "input_config": input_config,
            "engine_config": engine_config,
            "future": future,
            "worker_ids": worker_ids,
            "original_item": item,
        }
        queue.append(task)

    async def get(self, queue) -> Tuple[int, Any]:
        if not queue:
            raise asyncio.QueueEmpty

        current_time = time.time()

        if current_time - self.last_schedule_time >= self.schedule_interval:
            self.last_schedule_time = current_time

            available_tasks = [task for task in queue if task["r_j"] <= current_time]

            if available_tasks:
                scores = self.parameterized_compute_scores(
                    available_tasks, current_time
                )
                sorted_tasks_scores = sorted(
                    zip(available_tasks, scores), key=lambda x: x[1], reverse=True
                )
                selected_tasks = self.parameterized_select_subset(sorted_tasks_scores)

                if selected_tasks:
                    time_window = self.compute_time_window(
                        selected_tasks, self.M, current_time
                    )
                    schedule, solve_time = self.parameterized_solve_ilp_makespan(
                        selected_tasks, current_time, time_window
                    )
                    logger.info(
                        f"[Scheduler] ILP Makespan solved in {solve_time:.2f}s, scheduled {len(schedule)} tasks"
                    )

                    self.scheduled_tasks = []
                    for sched_item in schedule:
                        task_id = sched_item["task_id"]
                        num_required_machines = sched_item["k"]
                        task_actual_start_time = sched_item[
                            "start_time"
                        ]  # Absolute time from ILP

                        task_in_queue = None
                        for t_obj in available_tasks:
                            if t_obj["id"] == task_id:
                                task_in_queue = t_obj
                                break

                        if task_in_queue is None:
                            # Should not happen if ILP schedules tasks from available_tasks
                            logger.debug(
                                f"[Scheduler] Error: Task {task_id} from schedule not found in available tasks."
                            )
                            continue

                        duration = task_in_queue["t_jk"][num_required_machines]
                        task_completion_time = task_actual_start_time + duration

                        eligible_machines = [
                            m
                            for m in self.machines
                            if m["available_at"] <= task_actual_start_time
                        ]
                        eligible_machines.sort(
                            key=lambda m: (m["available_at"], m["id"])
                        )

                        if len(eligible_machines) < num_required_machines:
                            logger.warning(
                                f"[Scheduler] Warning: Task {task_in_queue['id']} requires {num_required_machines} machines, "
                                f"but only {len(eligible_machines)} are eligible by {task_actual_start_time:.2f}. "
                                f"This might indicate an issue with scheduling or resource contention."
                            )
                            available_count = len(eligible_machines)
                            allowed_count = max(
                                [
                                    k
                                    for k in self.allowed_machine_counts
                                    if k <= available_count
                                ],
                                default=0,
                            )
                            if allowed_count > 0:
                                assigned_machines = eligible_machines[:allowed_count]
                            else:
                                logger.warning(
                                    f"[Scheduler] Warning: No allowed machine count ({self.allowed_machine_counts}) available for task {task_in_queue['id']}."
                                )
                                assigned_machines = []
                            if not assigned_machines and num_required_machines > 0:
                                logger.debug(
                                    f"[Scheduler] Critical Error: No machines available for task {task_in_queue['id']} requiring {num_required_machines}."
                                )
                                # Skip this task or handle error appropriately
                                continue
                        else:
                            if num_required_machines in self.allowed_machine_counts:
                                assigned_machines = eligible_machines[
                                    :num_required_machines
                                ]
                            else:
                                allowed_count = max(
                                    [
                                        k
                                        for k in self.allowed_machine_counts
                                        if k <= num_required_machines
                                    ],
                                    default=self.allowed_machine_counts[0],
                                )
                                assigned_machines = eligible_machines[:allowed_count]
                                logger.warning(
                                    f"[Scheduler] Adjusting: Task {task_in_queue['id']} requires {num_required_machines} machines, "
                                    f"but using allowed count {allowed_count} instead."
                                )

                        selected_machine_ids = [m["id"] for m in assigned_machines]
                        task_in_queue["worker_ids"] = (
                            selected_machine_ids  # Update worker_ids for the task
                        )

                        # Update available time for selected machines
                        for m_obj in assigned_machines:
                            m_obj["available_at"] = task_completion_time

                        # Add to scheduled task list
                        self.scheduled_tasks.append(task_in_queue)

        if self.scheduled_tasks:
            task_to_return = self.scheduled_tasks.pop(0)

            if task_to_return in queue:  # Check if it's still in the main queue
                queue.remove(task_to_return)

            original_item_tuple = task_to_return["original_item"]
            updated_original_item = (
                original_item_tuple[0],  # task_id
                original_item_tuple[1],  # input_config
                original_item_tuple[2],  # engine_config
                original_item_tuple[3],  # future
                task_to_return["worker_ids"],
            )  # newly assigned worker_ids

            return task_to_return["priority"], updated_original_item
        else:
            raise asyncio.QueueEmpty  # No scheduled task ready to be returned

    def parameterized_compute_scores(self, tasks, current_time):
        slack_times = []
        wait_times = []
        for task in tasks:
            r_j = task["r_j"]
            d_j = task["d_j"]
            remaining = max(0, d_j - current_time)
            slack = (
                remaining - (current_time - r_j) if current_time >= r_j else remaining
            )
            slack = max(slack, 1e-5)
            wait = max(current_time - r_j, 0)
            slack_times.append(slack)
            wait_times.append(wait)

        avg_slack = sum(slack_times) / len(tasks) if slack_times else 0.1
        avg_wait = sum(wait_times) / len(tasks) if wait_times else 0
        alpha = self.alpha_coef / (avg_slack + 0.1)
        beta = self.beta_coef * avg_wait

        scores = []
        for task, st, wt in zip(tasks, slack_times, wait_times):
            scores.append(alpha * (1 / st) + beta * wt)
        return scores

    def parameterized_select_subset(self, sorted_tasks_scores):
        if not sorted_tasks_scores:
            return []
        m = 1
        for i in range(1, len(sorted_tasks_scores)):
            score_diff = sorted_tasks_scores[i - 1][1] - sorted_tasks_scores[i][1]
            if score_diff > self.max_diff:
                m = i
                break
            m += 1
        m = min(m, self.max_m)
        return [task for task, _ in sorted_tasks_scores[:m]]

    def compute_time_window(self, selected_tasks, M, current_time):
        l = [current_time] * M
        prev_s = current_time
        for task in selected_tasks:
            A = sorted(l)
            best_c = float("inf")
            best_k = None
            # Only consider allowed machine counts
            for k in self.allowed_machine_counts:
                if k < task["q_min"] or k > len(A):
                    continue
                if k not in task["t_jk"]:
                    continue
                s = max(prev_s, A[k - 1])
                c = s + task["t_jk"][k]
                if c < best_c:
                    best_c = c
                    best_k = k
            prev_s = best_c
            for i in range(best_k):
                A[i] = best_c
            l = sorted(A)
        return max(30, int(prev_s - current_time))

    def parameterized_solve_ilp_makespan(self, subset, current_time, T):
        """
        Solve the scheduling problem to minimize makespan using Integer Linear Programming (ILP).
        Args:
            subset: List of tasks to be scheduled.
            current_time: Current timestamp.
            T: Time window for scheduling.
        Returns:
            schedule: List of scheduled tasks with start time and machine count.
            solve_time: Time taken to solve the ILP.
        """
        import pulp

        model = pulp.LpProblem("Scheduling_Makespan", pulp.LpMinimize)
        task_map = {task["id"]: task for task in subset}
        L = 1e5  # Large constant

        # Decision variable definition - only create variables for allowed machine counts
        x_vars = pulp.LpVariable.dicts(
            "x",
            [
                (j["id"], t, k)
                for j in subset
                for k in self.allowed_machine_counts
                if k >= j["q_min"] and k <= self.M and k in j["t_jk"]
                for t in range(
                    max(0, int(j["r_j"] - current_time)), T - j["t_jk"][k] + 1
                )
            ],
            cat="Binary",
        )

        # Define makespan variable (maximum completion time of all tasks)
        makespan = pulp.LpVariable("makespan", lowBound=0)

        # Objective: minimize makespan
        model += makespan

        # Constraints
        for j in subset:
            j_id = j["id"]
            q_min = j["q_min"]
            # Constraint 1: Each task must be assigned exactly once
            allowed_k_values = [
                k
                for k in self.allowed_machine_counts
                if k >= q_min and k <= self.M and k in j["t_jk"]
            ]
            model += (
                pulp.lpSum(
                    x_vars.get((j_id, t, k), 0)
                    for k in allowed_k_values
                    for t in range(
                        max(0, int(j["r_j"] - current_time)), T - j["t_jk"][k] + 1
                    )
                )
                == 1
            )
            # Constraint 2: Makespan must be greater than or equal to each task's completion time
            for k in allowed_k_values:
                for t in range(
                    max(0, int(j["r_j"] - current_time)), T - j["t_jk"][k] + 1
                ):
                    if (j_id, t, k) in x_vars:
                        completion_time = current_time + t + j["t_jk"][k]
                        model += makespan >= completion_time * x_vars[(j_id, t, k)]

        # Constraint 3: Machine occupancy - ensure total machines used at any time does not exceed available
        for t_rel in range(T + 1):
            total = 0
            for j in subset:
                j_id = j["id"]
                for k in self.allowed_machine_counts:
                    if k < j["q_min"] or k > self.M or k not in j["t_jk"]:
                        continue
                    for t in range(
                        max(0, int(j["r_j"] - current_time)), T - j["t_jk"][k] + 1
                    ):
                        if t <= t_rel < t + j["t_jk"][k]:
                            if (j_id, t, k) in x_vars:
                                total += k * x_vars[(j_id, t, k)]
            model += total <= self.M

        # Add solver time limit
        solver = pulp.PULP_CBC_CMD(timeLimit=5, msg=False)

        # Record solving time
        start_time = time.time()
        status = model.solve(solver)
        solve_time = time.time() - start_time

        if status != pulp.LpStatusOptimal:
            return [], solve_time  # Return empty list and time

        # Output schedule
        schedule = []
        for j_id, t, k in x_vars:
            if (j_id, t, k) in x_vars and x_vars[
                (j_id, t, k)
            ].value() > 0.5:  # Use 0.5 as threshold for numerical precision
                schedule.append(
                    {"task_id": j_id, "start_time": current_time + t, "k": k}
                )

        return schedule, solve_time


class Efficient_ILP_Strategy(SchedulingStrategy, ABC):
    """
    ILP-based strategy: periodically select a batch of tasks and cache them in self.scheduled_tasks.
    """

    def __init__(self, schedule_interval: float = 0.5, M: int = 8, model_profiler=None):
        self.interval = schedule_interval
        self.last_sched = 0.0
        self.M = M
        self.machines = [{"id": i, "available_at": 0.0} for i in range(M)]
        self.scheduled_tasks: List[Dict] = []  # Cached scheduled tasks
        self.model_profiler = model_profiler

    async def put(self, queue: List, priority_unused: int, item: Any, t_dict):
        """
        Add a task to the scheduling queue.
        Args:
            queue: The scheduling queue.
            priority_unused: Unused priority parameter.
            item: Tuple (task_id, input_cfg, eng_cfg).
            t_dict: Dictionary of {k: t_k} for parallelism.
        """
        task_id, input_config, engine_config = item
        now = time.time()
        ddl_rel_sec = estimate_ddl(t_dict)
        ddl_abs = int(time.time() + ddl_rel_sec)
        element = {
            "task_id": task_id,
            "input_config": input_config,
            "engine_config": engine_config,
            "ddl": ddl_abs,
            "t": t_dict,
        }

        # Do not batch if num_frames != 1 or if it's a profiling task
        if input_config.num_frames != 1 or "Model_Profiler" in input_config.prompt:
            queue.append(element)
            return

        # print(f"patarallel_config = {parallel_config}")
        max_bs = self.model_profiler.query_best_batchsize(
            input_config.height, input_config.width, input_config.num_frames
        )
        if max_bs == 1:
            queue.append(element)
            return
        # Try to merge with existing requests in the queue
        merged = False
        for idx, q_element in enumerate(queue):
            q_input_config = q_element["input_config"]
            if (
                q_input_config.height == input_config.height
                and q_input_config.width == input_config.width
                and q_input_config.num_frames == input_config.num_frames
            ):
                if isinstance(q_input_config.prompt, str):
                    bs = 1
                else:
                    bs = len(q_input_config.prompt)
                if bs < max_bs:
                    for key in ["prompt", "negative_prompt"]:
                        new_val = getattr(input_config, key, None)
                        old_val = getattr(q_input_config, key, None)
                        if new_val is not None:
                            if isinstance(old_val, str):
                                old_val = [old_val]
                            if isinstance(new_val, str):
                                new_val = [new_val]
                            merged_val = (old_val or []) + (new_val or [])
                            setattr(q_input_config, key, merged_val)
                    merged = True
                    break
        if not merged:
            queue.append(element)

    async def get(
        self, queue: List, free_machine_num=8, busy_machine_idle_time=None
    ) -> Tuple[int, Any]:
        """
        Pop a scheduled task from the queue, or perform scheduling if needed.
        Args:
            queue: The scheduling queue.
            free_machine_num: Number of free machines.
            busy_machine_idle_time: Idle time of busy machines.
        Returns:
            Tuple (task_id, input_config, engine_config)
        """
        now = time.time()
        if self.scheduled_tasks:
            t = self.scheduled_tasks.pop(0)
            queue.remove(t)
            logger.debug(f"in get, t is {t}")
            return (t["task_id"], t["input_config"], t["engine_config"])
        if now - self.last_sched < self.interval or not queue:
            raise asyncio.QueueEmpty
        self.last_sched = now
        alloc = select_tasks(
            now, free_machine_num, busy_eta=busy_machine_idle_time, tasks=queue
        )
        if not alloc:
            raise asyncio.QueueEmpty
        for task_id, k in alloc:
            task = next(t for t in queue if t["task_id"] == task_id)
            logger.debug(f"task_id is {task['task_id']}, k is {k}")
            parallel_config = task["engine_config"].parallel_config
            parallel_config.dp_config.dp_degree = 1
            parallel_config.sp_config.ulysses_degree = k
            parallel_config.sp_config.ring_degree = 1
            parallel_config.tp_config.tp_degree = 1
            parallel_config.pp_config.pp_degree = 1
            parallel_config.sp_config.sp_degree = k
            parallel_config.dp_degree = parallel_config.dp_config.dp_degree
            parallel_config.cfg_degree = parallel_config.dp_config.cfg_degree
            parallel_config.sp_degree = parallel_config.sp_config.sp_degree
            parallel_config.tp_degree = parallel_config.tp_config.tp_degree
            parallel_config.pp_degree = parallel_config.pp_config.pp_degree
            parallel_config.ulysses_degree = parallel_config.sp_config.ulysses_degree
            parallel_config.ring_degree = parallel_config.sp_config.ring_degree
            parallel_config.world_size = (
                parallel_config.dp_degree
                * parallel_config.cfg_degree
                * parallel_config.sp_degree
                * parallel_config.tp_degree
                * parallel_config.pp_degree
            )
            task["engine_config"].parallel_config = copy.deepcopy(parallel_config)
            self.scheduled_tasks.append(task)
        t0 = self.scheduled_tasks.pop(0)
        queue.remove(t0)
        logger.debug(f"in get, t0: {t0}")
        return (t0["task_id"], t0["input_config"], t0["engine_config"])


class Efficient_ILP_Multi_Machine_Strategy(SchedulingStrategy, ABC):
    """
    Multi-machine ILP scheduling strategy (single window version).
    Each call to `get()` requires passing the current free GPU count per machine (m_free).
    """

    def __init__(self, schedule_interval: float = 0.5, M: int = 8, model_profiler=None):
        self.interval = schedule_interval
        self.last_sched = 0.0
        self.M = M  # Kept for compatibility, not used internally
        self.scheduled_tasks: List[Dict] = []  # Cached scheduled tasks
        self.model_profiler = model_profiler

    async def put(self, queue: List, priority_unused: int, item: Any, t_dict):
        """
        Add a task to the scheduling queue.
        Args:
            queue: The scheduling queue.
            priority_unused: Unused priority parameter.
            item: Tuple (task_id, input_cfg, eng_cfg).
            t_dict: Dictionary of {k: t_k} for parallelism.
        """
        task_id, input_config, engine_config = item
        now = time.time()
        ddl_rel_sec = estimate_ddl(t_dict)
        ddl_abs = int(time.time() + ddl_rel_sec)
        element = {
            "task_id": task_id,
            "input_config": input_config,
            "engine_config": engine_config,
            "ddl": ddl_abs,
            "t": t_dict,
        }

        # Do not batch if num_frames != 1 or if it's a profiling task
        if input_config.num_frames != 1 or "Model_Profiler" in input_config.prompt:
            queue.append(element)
            return

        # print(f"patarallel_config = {parallel_config}")
        max_bs = self.model_profiler.query_best_batchsize(
            input_config.height, input_config.width, input_config.num_frames
        )
        if max_bs == 1:
            queue.append(element)
            return
        # Try to merge with existing requests in the queue
        merged = False
        for idx, q_element in enumerate(queue):
            q_input_config = q_element["input_config"]
            if (
                q_input_config.height == input_config.height
                and q_input_config.width == input_config.width
                and q_input_config.num_frames == input_config.num_frames
            ):
                if isinstance(q_input_config.prompt, str):
                    bs = 1
                else:
                    bs = len(q_input_config.prompt)
                if bs < max_bs:
                    for key in ["prompt", "negative_prompt"]:
                        new_val = getattr(input_config, key, None)
                        old_val = getattr(q_input_config, key, None)
                        if new_val is not None:
                            if isinstance(old_val, str):
                                old_val = [old_val]
                            if isinstance(new_val, str):
                                new_val = [new_val]
                            merged_val = (old_val or []) + (new_val or [])
                            setattr(q_input_config, key, merged_val)
                    merged = True
                    break
        if not merged:
            queue.append(element)

    async def get(
        self,
        queue: List[Dict],
        m_free: List[int],  # e.g. [5, 3, 8]
        *_,
        **__,
    ) -> Tuple[str, Any, Any, int]:
        """
        Pop a scheduled task with assigned machine and parallelism.
        Args:
            queue: The scheduling queue.
            m_free: List of free GPU counts per machine.
        Returns:
            Tuple (task_id, input_config, engine_config, machine_id)
        """
        now = time.time()
        if self.scheduled_tasks:
            task = self.scheduled_tasks.pop(0)
            queue.remove(task)
            return (
                task["task_id"],
                task["input_config"],
                task["engine_config"],
                task["machine_id"],
            )
        if now - self.last_sched < self.interval or not queue:
            raise asyncio.QueueEmpty

        self.last_sched = now

        alloc = select_tasks_multi(
            now, m_free, tasks=queue
        )  # [(task_id, k, m_id), ...]
        if not alloc:
            raise asyncio.QueueEmpty

        for task_id, k, m_id in alloc:
            task = next(t for t in queue if t["task_id"] == task_id)

            pcfg = task["engine_config"].parallel_config
            pcfg = _configure_splitk_parallel(pcfg, k)
            task["engine_config"].parallel_config = copy.deepcopy(pcfg)
            task["machine_id"] = m_id
            task["k_assigned"] = k
            self.scheduled_tasks.append(task)

        t0 = self.scheduled_tasks.pop(0)
        queue.remove(t0)
        return (
            t0["task_id"],
            t0["input_config"],
            t0["engine_config"],
            t0["machine_id"],
        )


class Greedy_Splitk_Strategy(SchedulingStrategy, ABC):
    """
    Greedy split-k scheduling:
      • Enumerate all k with qualified efficiency for each task in the queue
      • Calculate urgency u = (ddl-now-t_k)/t_optimal
        - u>0 sorted by u ascending (smaller is more urgent)
        - u≤0 sorted by |u| descending
      • Try to fit into current free GPUs (m_free) in order
        - If possible, configure and return immediately
    """

    def __init__(self, schedule_interval: float = 0.5, M: int = 8):
        self.interval = schedule_interval  # This strategy does not actually use interval, kept for compatibility
        self.last_sched = 0.0
        self.M = M  # Placeholder, can be ignored externally

    async def put(
        self,
        queue: List[Dict],
        _priority_unused: int,
        item: Any,
        t_dict: Dict[int, float],
    ):
        """
        Add a task to the scheduling queue.
        """
        task_id, input_cfg, eng_cfg = item
        ddl_rel_sec = estimate_ddl(t_dict)
        ddl_abs = int(time.time() + ddl_rel_sec)

        queue.append(
            {
                "task_id": task_id,
                "input_config": input_cfg,
                "engine_config": eng_cfg,
                "ddl": ddl_abs,
                "t": t_dict,
            }
        )

    async def get(
        self,
        queue: List[Dict],
        m_free: List[int],  # e.g. [1,2,3]
        *_,
        **__,
    ) -> Tuple[str, Any, Any, int]:
        """
        Pop a scheduled task with assigned machine and parallelism.
        Args:
            queue: The scheduling queue.
            m_free: List of free GPU counts per machine.
        Returns:
            Tuple (task_id, input_config, engine_config, machine_id)
        """
        if not queue:
            raise asyncio.QueueEmpty

        now = time.time()
        candidates = []
        for t in queue:
            t1 = t["t"][1]
            feasible_k = [k for k in K_SET if (t1 / t["t"][k]) / k >= EFF_TH]
            if not feasible_k:
                continue

            k_opt = max(feasible_k)
            t_opt = t["t"][k_opt]

            for k in feasible_k:
                if k != k_opt:
                    continue
                tk = t["t"][k]
                u = (t["ddl"] - now - tk) / t_opt
                candidates.append(
                    {
                        "task": t,
                        "k": k,
                        "u": u,
                        "t_opt": t_opt,
                    }
                )

        if not candidates:
            raise asyncio.QueueEmpty

        pos = sorted(
            [c for c in candidates if c["u"] > 0],
            key=lambda c: c["u"],
        )
        neg = sorted(
            [c for c in candidates if c["u"] <= 0],
            key=lambda c: abs(c["u"]),
            reverse=False,
        )
        ordered = pos + neg

        for cand in ordered:
            k = cand["k"]
            capable = [m for m, free in enumerate(m_free) if free >= k]
            if not capable:
                continue

            best_free = max(m_free[m] for m in capable)
            machines_best = [m for m in capable if m_free[m] == best_free]
            m_sel = min(machines_best)

            task = cand["task"]

            pcfg = task["engine_config"].parallel_config
            pcfg = _configure_splitk_parallel(pcfg, k)
            task["engine_config"].parallel_config = copy.deepcopy(pcfg)

            queue.remove(task)
            return (
                task["task_id"],
                task["input_config"],
                task["engine_config"],
                m_sel,
            )

        raise asyncio.QueueEmpty


class Scheduler:
    def __init__(
        self,
        strategy: str = "priority",
        search_mode: str = "random",
        model_profiler=None,
    ) -> None:
        self._queue = []
        self.task_timestamps = {}
        self.task_durations = []
        self.strategy = strategy
        self.search_mode = search_mode

        if strategy == "priority":
            self.strategy = PriorityStrategy()
        elif strategy == "fifo":
            self.strategy = FIFOStrategy()
        elif strategy == "ilp_fix":
            if search_mode != "fix":
                raise ValueError("ILP_fix strategy only works with 'fix' search_mode")
            self.strategy = ILP_fix_Strategy()
        elif strategy == "ilp_random":
            if search_mode != "random":
                raise ValueError(
                    "ILP_random strategy only works with 'random' search_mode"
                )
            self.strategy = ILP_random_Strategy()
        elif strategy == "ilp_makespan":
            if search_mode != "fix":
                raise ValueError(
                    "ILP_makespan strategy only works with 'fix' search_mode"
                )
            self.strategy = ILP_makespan_Strategy()
        elif strategy == "greedy_random":
            if search_mode != "random":
                raise ValueError(
                    "greedy_random strategy only works with 'random' search_mode"
                )
            self.strategy = GreedyRandomStrategy()
        elif strategy == "efficient_ilp":
            self.strategy = Efficient_ILP_Strategy(model_profiler=model_profiler)
        elif strategy == "multi_machine_efficient_ilp":
            self.strategy = Efficient_ILP_Multi_Machine_Strategy(
                model_profiler=model_profiler
            )
        elif strategy == "greedy_splitk":
            self.strategy = Greedy_Splitk_Strategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _map_parallel_degrees(self, priority: int, k: int) -> Tuple[int, int, int, int]:
        table = {
            (16384, 1): (1, 1, 1, 1),
            (16384, 2): (1, 1, 1, 2),
            (16384, 4): (1, 1, 1, 4),
            (16384, 8): (1, 1, 1, 8),
            (32768, 1): (1, 1, 1, 1),
            (32768, 2): (1, 1, 1, 2),
            (32768, 4): (1, 1, 1, 4),
            (32768, 8): (1, 1, 1, 8),
            (65536, 1): (1, 1, 1, 1),
            (65536, 2): (1, 1, 1, 2),
            (65536, 4): (1, 1, 1, 4),
            (65536, 8): (1, 1, 1, 8),
            (131072, 1): (1, 1, 1, 1),
            (131072, 2): (1, 1, 1, 2),
            (131072, 4): (1, 1, 1, 4),
            (131072, 8): (1, 1, 1, 8),
            (262144, 1): (1, 1, 1, 1),
            (262144, 2): (1, 1, 1, 2),
            (262144, 4): (1, 1, 1, 4),
            (262144, 8): (1, 1, 1, 8),
            (524288, 1): (1, 1, 1, 1),
            (524288, 2): (1, 1, 1, 2),
            (524288, 4): (1, 1, 1, 4),
            (524288, 8): (1, 1, 1, 8),
            (1048576, 1): (1, 1, 1, 1),
            (1048576, 2): (1, 1, 1, 2),
            (1048576, 4): (1, 1, 1, 4),
            (1048576, 8): (1, 1, 1, 8),
        }
        return table.get((priority, k), (k, 1, 1, 1))

    async def put(self, priority: int, item: Any, t_dict=None):
        if t_dict is None:
            await self.strategy.put(self._queue, priority, item)
        else:
            await self.strategy.put(self._queue, priority, item, t_dict)

    async def get(
        self, free_machine_num=None, busy_machine_idle_time=None
    ) -> Tuple[int, Any]:
        if free_machine_num is None:
            priority, item = await self.strategy.get(self._queue)
        else:
            logger.debug("before call get")
            item = await self.strategy.get(
                self._queue, free_machine_num, busy_machine_idle_time
            )
        if self.search_mode == "random":
            if isinstance(self.strategy, ILP_random_Strategy):
                original_item, assigned_machine_count = item
                task_id, input_config, engine_config = original_item
                num_workers = assigned_machine_count
                parallel_config = engine_config.parallel_config
                degrees = self._map_parallel_degrees(priority, num_workers)
                _configure_random_parallel(
                    parallel_config,
                    ulysses_degree=degrees[0],
                    ring_degree=degrees[1],
                    tp_degree=degrees[2],
                    pp_degree=degrees[3],
                )
                return priority, (task_id, input_config, engine_config)
            else:
                return priority, item
        elif (
            self.search_mode == "efficient_ilp"
            or self.search_mode == "multi_machine_efficient_ilp"
            or self.search_mode == "greedy_splitk"
        ):
            logger.debug(f"{self.search_mode} item is : {item}")
            return item
        elif self.search_mode == "fix":
            task_id, input_config, engine_config, future, worker_ids = item
            num_workers = len(worker_ids)
            parallel_config = engine_config.parallel_config
            degrees = self._map_parallel_degrees(priority, num_workers)
            _configure_random_parallel(
                parallel_config,
                ulysses_degree=degrees[0],
                ring_degree=degrees[1],
                tp_degree=degrees[2],
                pp_degree=degrees[3],
            )
            return priority, (task_id, input_config, engine_config, future, worker_ids)
        else:
            raise ValueError(f"Unknown search mode: {self.search_mode}")

    def empty(self) -> bool:
        return len(self._queue) == 0

    def record_start_time(self, task_id: str):
        self.task_timestamps[task_id] = time.time()

    def record_end_time(self, task_id: str):
        if task_id in self.task_timestamps:
            start_time = self.task_timestamps[task_id]
            end_time = time.time()
            duration = end_time - start_time
            start_time_local = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(start_time)
            )
            end_time_local = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(end_time)
            )
            logger.info(
                f"[API Server] task:{task_id} start_time:{start_time_local} end_time:{end_time_local} duration:{duration}"
            )
            self.task_durations.append(duration)
            del self.task_timestamps[task_id]

    def get_average_latency(self) -> float:
        if not self.task_durations:
            return 0.0
        avg_durations = sum(self.task_durations) / len(self.task_durations)
        return avg_durations
