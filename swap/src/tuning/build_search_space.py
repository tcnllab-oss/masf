from copy import deepcopy

from swap.src.utils.plan_utils import deep_get_by_dotted_key
from swap.src.utils.dependency_utils import load_previous_best_configs


def frange_inclusive(low: float, high: float, step: float):
    values = []
    x = low
    eps = abs(step) * 1e-9 if step != 0 else 1e-12
    while x <= high + eps:
        values.append(round(x, 10))
        x += step
    return values


def build_centered_values_from_prev(prev_cfg: dict, param_name: str, spec: dict):
    center = deep_get_by_dotted_key(prev_cfg, param_name, default=None)
    if center is None:
        return None

    ptype = spec.get("type")
    radius = spec.get("radius")
    step = spec.get("step")
    clamp_low = spec.get("clamp_low")
    clamp_high = spec.get("clamp_high")

    if ptype == "categorical":
        values = spec.get("values")
        if not values:
            return None
        return [center] if center in values else None

    if radius is None or step is None:
        return None

    if ptype == "int":
        center = int(center)
        radius = int(radius)
        step = int(step)

        low = center - radius
        high = center + radius

        if clamp_low is not None:
            low = max(low, int(clamp_low))
        if clamp_high is not None:
            high = min(high, int(clamp_high))

        values = list(range(low, high + 1, step))
        return sorted(set(values))

    if ptype == "float":
        center = float(center)
        radius = float(radius)
        step = float(step)

        low = center - radius
        high = center + radius

        if clamp_low is not None:
            low = max(low, float(clamp_low))
        if clamp_high is not None:
            high = min(high, float(clamp_high))

        values = frange_inclusive(low, high, step)
        return sorted(set(values))

    return None


def maybe_recenter_param_specs_from_prev_cfg(plan: dict, project_dir, phase_name: str, param_specs: dict):
    if not bool(plan.get("warm_start_from_prev_cfg", False)):
        return deepcopy(param_specs)

    prev_best_cfg, _ = load_previous_best_configs(plan, project_dir, phase_name)
    if not prev_best_cfg:
        return deepcopy(param_specs)

    warm_start_cfg = plan.get("warm_start", {})
    if not warm_start_cfg:
        return deepcopy(param_specs)

    new_specs = deepcopy(param_specs)

    for param_name, ws_spec in warm_start_cfg.items():
        if param_name not in new_specs:
            continue

        centered_values = build_centered_values_from_prev(
            prev_cfg=prev_best_cfg,
            param_name=param_name,
            spec=ws_spec,
        )

        if not centered_values:
            continue

        new_specs[param_name] = {
            "type": "categorical",
            "values": centered_values,
        }

        print(f"[WARM_START] {param_name} centered around previous cfg -> {centered_values}")

    return new_specs


def make_effective_param_specs(plan: dict, project_dir, phase_name: str):
    phase_cfg = plan["search"][phase_name]
    base_param_specs = deepcopy(phase_cfg["params"])
    return maybe_recenter_param_specs_from_prev_cfg(
        plan=plan,
        project_dir=project_dir,
        phase_name=phase_name,
        param_specs=base_param_specs,
    )