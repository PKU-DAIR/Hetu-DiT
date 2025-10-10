from typing import Any, Dict, List, Tuple

import pulp

from hetu_dit.logger import init_logger

logger = init_logger(__name__)

# Constants reused elsewhere; keep names intact.
EFF_TH = 0.8
K_SET = (1, 2, 4, 8)
BIG_ON = 1000
LATE_BASE = 100
BETA = 1.0
PER_NODE_GPU = 8
K_BONUS = 1

Task = Dict[str, Any]


def select_tasks_multi(
    now: float,
    m_free: List[int],
    tasks: List[Task],
) -> List[Tuple[str, int, int]]:
    """Solve the multi-machine ILP and return tasks to launch immediately."""

    machine_count = len(m_free)
    model = pulp.LpProblem("multi_node", pulp.LpMaximize)
    decision_vars: Dict[Tuple[int, int, int], pulp.LpVariable] = {}
    coefficients: Dict[pulp.LpVariable, float] = {}

    for task_index, task in enumerate(tasks):
        t1 = task["t"][1]
        deadline = task["ddl"]
        input_cfg = task.get("input_config")

        for degree in K_SET:
            runtime = task["t"][degree]
            if not _passes_efficiency_filter(t1, runtime, degree, input_cfg):
                continue

            for machine_index in range(machine_count):
                if degree > m_free[machine_index]:
                    continue

                finish_time = now + runtime
                reward = _compute_reward(
                    task,
                    t1,
                    deadline,
                    degree,
                    runtime,
                    now,
                    m_free[machine_index],
                )
                if reward <= 0:
                    continue

                variable = pulp.LpVariable(
                    f"x_{task_index}_{degree}_{machine_index}", 0, 1, cat="Binary"
                )
                decision_vars[(task_index, degree, machine_index)] = variable
                coefficients[variable] = reward * BETA

    slack_variables = [
        pulp.LpVariable(f"dummy_{machine_index}", 0, None, cat="Integer")
        for machine_index in range(machine_count)
    ]

    for task_index in range(len(tasks)):
        model += (
            pulp.lpSum(
                decision_vars.get((task_index, degree, machine_index), 0)
                for degree in K_SET
                for machine_index in range(machine_count)
            )
            <= 1
        )

    for machine_index in range(machine_count):
        model += (
            pulp.lpSum(
                degree * decision_vars.get((task_index, degree, machine_index), 0)
                for task_index in range(len(tasks))
                for degree in K_SET
            )
            + slack_variables[machine_index]
            == m_free[machine_index]
        )

        if m_free[machine_index] > PER_NODE_GPU:
            raise ValueError(
                f"Machine {machine_index} reports {m_free[machine_index]} free GPUs, beyond PER_NODE_GPU"
            )

    model += pulp.lpSum(coefficients[var] * var for var in coefficients) - pulp.lpSum(
        slack_variables
    )
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    results = [
        (tasks[task_index]["task_id"], degree, machine_index)
        for (task_index, degree, machine_index), var in decision_vars.items()
        if var.value() > 0.5
    ]
    results.sort(key=lambda item: (item[2], item[0]))
    return results


def _passes_efficiency_filter(
    t1: float,
    tk: float,
    degree: int,
    input_cfg: Any,
) -> bool:
    if (t1 / tk) / degree < EFF_TH:
        return False
    if not input_cfg:
        return True

    height = getattr(input_cfg, "height", None)
    width = getattr(input_cfg, "width", None)
    frames = getattr(input_cfg, "num_frames", None)
    resolution_volume = None
    if height is not None and width is not None and frames is not None:
        resolution_volume = height * width * frames

    if height == 4096 and width == 4096 and degree < 2:
        return False
    if height == 3072 and width == 3072 and degree < 2:
        return False
    if height == 2048 and width == 2048 and degree < 2:
        return False

    if resolution_volume is not None:
        if 720 * 1280 * 17 < resolution_volume <= 720 * 1280 * 33 and degree < 2:
            return False
        if 720 * 1280 * 33 < resolution_volume <= 720 * 1280 * 65 and degree < 4:
            return False
        if resolution_volume > 720 * 1280 * 65 and degree < 8:
            return False

    return True


def _compute_reward(
    task: Task,
    t1: float,
    deadline: float,
    degree: int,
    runtime: float,
    now: float,
    machine_capacity: int,
) -> float:
    finish_time = now + runtime

    if finish_time <= deadline:
        return BIG_ON

    feasible = [
        k for k in K_SET if (t1 / task["t"][k]) / k >= EFF_TH and k <= machine_capacity
    ]
    if not feasible:
        return 0

    best_runtime = min(task["t"][k] for k in feasible)
    lateness_ratio = (finish_time - deadline) / best_runtime
    late_reward = int(LATE_BASE * (1 + lateness_ratio))
    return min(BIG_ON, late_reward + K_BONUS * (degree - 1))
