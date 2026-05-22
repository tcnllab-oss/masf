from copy import deepcopy
import itertools


def expand_plan_grid(plan: dict):
    grid = plan.get("grid", {}) or {}

    if not grid:
        return [deepcopy(plan)]

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    expanded = []
    for combo in itertools.product(*values):
        plan_i = deepcopy(plan)
        plan_i.setdefault("fixed", {})

        for key, value in zip(keys, combo):
            plan_i["fixed"][key] = value
            # "dynamics.num_samples": 100
            # -> plan_i["fixed"]["dynamics.num_samples"] = 100

        expanded.append(plan_i)

    return expanded


def describe_grid_values(plan: dict) -> str:
    fixed = plan.get("fixed", {}) or {}
    parts = []

    if "dynamics.num_samples" in fixed:
        parts.append(f"num_samples={fixed['dynamics.num_samples']}")

    return ", ".join(parts) if parts else "no grid overrides"