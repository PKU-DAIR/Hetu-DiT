from typing import Any, Dict, List, Optional, Tuple

import pulp

EFF_TH = 0.8
K_SET = (1, 2, 4, 8)
BIG_ON = 1000
LATE_BASE = 500
TOTAL_GPU = 8
BETA = 0.1

Task = Dict[str, Any]


def select_tasks(
    now: float,
    m_free: int,
    busy_eta: Optional[List[float]],
    tasks: List[Task],
) -> List[Tuple[str, int]]:
    """Solve the multi-window ILP and return tasks that should start now."""

    starts, ends, capacities = _build_time_windows(now, m_free, busy_eta)
    num_windows = len(starts)

    model = pulp.LpProblem("unified_multiwindow", pulp.LpMaximize)
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

            for window_index in range(num_windows):
                if degree > capacities[window_index]:
                    continue
                if (
                    ends[window_index] is not None
                    and starts[window_index] + runtime > ends[window_index]
                ):
                    continue

                finish_time = starts[window_index] + runtime
                base_reward = (
                    BIG_ON
                    if finish_time <= deadline
                    else max(1, LATE_BASE - (finish_time - deadline))
                )
                discounted_reward = base_reward * (BETA**window_index)

                variable = pulp.LpVariable(
                    f"x_{task_index}_{degree}_{window_index}", 0, 1, cat="Binary"
                )
                decision_vars[(task_index, degree, window_index)] = variable
                coefficients[variable] = discounted_reward

    slack_variables = [
        pulp.LpVariable(f"dummy_{w}", 0, None, cat="Integer")
        for w in range(num_windows)
    ]

    for task_index in range(len(tasks)):
        model += (
            pulp.lpSum(
                decision_vars.get((task_index, degree, window_index), 0)
                for degree in K_SET
                for window_index in range(num_windows)
            )
            <= 1
        )

    for window_index in range(num_windows):
        model += (
            pulp.lpSum(
                degree * decision_vars.get((task_index, degree, window_index), 0)
                for task_index in range(len(tasks))
                for degree in K_SET
            )
            + slack_variables[window_index]
            == capacities[window_index]
        )

    model += pulp.lpSum(coefficients[var] * var for var in coefficients) - pulp.lpSum(
        slack_variables
    )
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    return [
        (tasks[task_index]["task_id"], degree)
        for (task_index, degree, window_index), var in decision_vars.items()
        if window_index == 0 and var.value() > 0.5
    ]


def _build_time_windows(
    now: float,
    free_gpus: int,
    busy_eta: Optional[List[float]],
) -> Tuple[List[float], List[Optional[float]], List[int]]:
    release_times = sorted(max(0.0, float(t)) for t in (busy_eta or []))
    starts = [now] + [now + eta for eta in release_times]
    capacities = [free_gpus]
    for _ in release_times:
        capacities.append(capacities[-1] + 1)
    if capacities[-1] > TOTAL_GPU:
        raise ValueError("Total GPU capacity exceeds the configured limit")
    ends: List[Optional[float]] = starts[1:] + [None]
    return starts, ends, capacities


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
    if height == 4096 and width == 4096 and degree < 2:
        return False
    if height == 3072 and width == 3072 and degree < 2:
        return False
    return True
