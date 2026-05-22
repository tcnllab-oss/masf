import csv
import json
import math
from pathlib import Path

from swap.src.utils.yaml_utils import load_yaml
from swap.src.utils.plan_utils import resolve_swap_dir
from swap.src.utils.io_utils import append_text, init_text_file
from itertools import product
from swap.src.utils.log_utils import (
    flatten_dict,
    shorten_param_name,
    write_txt_table,
)


def expand_plan_grid(grid: dict):

    """

    Example

    -------

    input:

        {

            "dynamics.num_samples": [100, 300],

            "steps.end": [50, 100],

        }

    output:

        [

            {"dynamics.num_samples": 100, "steps.end": 50},

            {"dynamics.num_samples": 100, "steps.end": 100},

            {"dynamics.num_samples": 300, "steps.end": 50},

            {"dynamics.num_samples": 300, "steps.end": 100},

        ]

    """

    if not grid:

        return [{}]

    keys = list(grid.keys())

    value_lists = []

    for key in keys:

        values = grid[key]

        if not isinstance(values, list):

            raise ValueError(

                f"grid['{key}'] must be a list, but got {type(values).__name__}"

            )

        if len(values) == 0:

            raise ValueError(f"grid['{key}'] must not be empty")

        value_lists.append(values)

    cases = []

    for combo in product(*value_lists):

        case = {}

        for key, value in zip(keys, combo):

            case[key] = value

        cases.append(case)

    return cases

def resolve_post_configs_dir(swap_dir: Path) -> Path:
    return swap_dir / "configs" / "post"


def resolve_plan_path(configs_dir: Path, phase: str) -> Path:
    return configs_dir / f"{phase}.yaml"


def safe_float(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def print_base_yaml_info(base_yaml: Path):
    cfg = load_yaml(base_yaml)

    dynamics = cfg.get("dynamics", {})
    steps = cfg.get("steps", {})
    measurement = cfg.get("measurement", {})

    print("=" * 80)
    print("[POST_EVAL] Loaded base yaml info")
    print("=" * 80)
    print(f"[POST_EVAL] base_yaml              : {base_yaml}")
    print(f"[POST_EVAL] dynamics.num_samples   : {dynamics.get('num_samples')}")
    print(f"[POST_EVAL] dynamics.dim           : {dynamics.get('dim')}")
    print(f"[POST_EVAL] steps.initial          : {steps.get('initial')}")
    print(f"[POST_EVAL] steps.end              : {steps.get('end')}")
    print(f"[POST_EVAL] steps.gap              : {steps.get('gap')}")
    print(f"[POST_EVAL] measurement.type       : {measurement.get('type')}")
    print(f"[POST_EVAL] measurement.T          : {measurement.get('T')}")
    print(f"[POST_EVAL] measurement.noise_std  : {measurement.get('noise_std')}")

    mtype = measurement.get("type")
    if mtype == "grid_mask":
        print(f"[POST_EVAL] measurement.stride      : {measurement.get('stride')}")
    elif mtype == "center_mask":
        print(f"[POST_EVAL] measurement.hole_ratio  : {measurement.get('hole_ratio')}")
    elif mtype == "low_resolution":
        print(f"[POST_EVAL] measurement.scale_factor: {measurement.get('scale_factor')}")
    elif mtype == "nonlinear":
        print(f"[POST_EVAL] measurement.nonlinear_type: {measurement.get('nonlinear_type')}")
        print(f"[POST_EVAL] measurement.alpha        : {measurement.get('alpha')}")

    print("=" * 80)


def inject_measurement_grid(plan: dict, base_yaml: Path):
    if plan["phase"] != "measurement_sensitivity":
        return plan

    cfg = load_yaml(base_yaml)
    mtype = cfg.get("measurement", {}).get("type")

    measurement_grid = plan.get("measurement_grid")
    if measurement_grid is None:
        raise ValueError("measurement_sensitivity config must contain 'measurement_grid'")

    if mtype not in measurement_grid:
        raise ValueError(
            f"measurement.type={mtype} not found in measurement_grid keys={list(measurement_grid.keys())}"
        )

    plan["grid"] = measurement_grid[mtype]
    return plan


def load_phase_plan(configs_dir: Path, phase: str, base_yaml_override=None):
    plan_path = resolve_plan_path(configs_dir, phase)
    if not plan_path.exists():
        raise FileNotFoundError(f"Config not found: {plan_path}")

    plan = load_yaml(plan_path)

    if base_yaml_override is not None:
        plan["base_yaml"] = base_yaml_override

    base_yaml = Path(plan["base_yaml"]).resolve()
    if not base_yaml.exists():
        raise FileNotFoundError(f"base_yaml not found: {base_yaml}")

    plan = inject_measurement_grid(plan, base_yaml)
    return plan_path, plan, base_yaml


def infer_report_root_from_base_yaml(base_yaml: Path) -> Path:
    parts = base_yaml.parts
    if "masf" not in parts:
        raise ValueError(f"Could not infer report root from base_yaml: {base_yaml}")

    masf_idx = parts.index("masf")
    if masf_idx + 1 >= len(parts):
        raise ValueError(f"Could not infer report root from base_yaml: {base_yaml}")

    return Path(*parts[:masf_idx + 2])


def resolve_measurement_root_from_base_yaml(base_yaml: Path) -> Path:
    base_yaml = Path(base_yaml).resolve()
    phase_dir = base_yaml.parent
    measurement_root = phase_dir.parent
    return measurement_root


def resolve_post_phase_root_from_base_yaml(base_yaml: Path, phase: str) -> Path:
    measurement_root = resolve_measurement_root_from_base_yaml(base_yaml)
    return measurement_root / "post" / phase


METRIC_COLUMNS = {
    "rmse_mean",
    "rmse_std",
    "csi_mean",
    "csi_std",
    "wallclock_mean",
    "wallclock_std",
}


def preferred_param_order():
    return [
        "dynamics.num_samples",
        "steps.initial",
        "steps.end",
        "steps.gap",
        "measurement.stride",
        "measurement.hole_ratio",
        "measurement.scale_factor",
        "measurement.alpha",
        "measurement.same_normalization",
        "measurement.normalization_form",
        "measurement.stats_mode",
        "measurement.stats_update_mode",
        "measurement.momentum",
    ]


def format_metric_value(v):
    if v is None:
        return ""
    try:
        x = float(v)
    except Exception:
        return str(v)

    if math.isnan(x):
        return "nan"

    text = f"{x:.4f}"
    return text.rstrip("0").rstrip(".")


def format_param_value(v):
    if v is None:
        return ""
    if isinstance(v, list):
        return "[" + ", ".join(format_param_value(x) for x in v) + "]"
    return str(v)


def write_csv_table(path: Path, headers, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for row in rows:
            out = {}
            for h in headers:
                v = row.get(h, "")
                if h in METRIC_COLUMNS:
                    out[h] = format_metric_value(v)
                elif isinstance(v, list):
                    out[h] = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    out[h] = ""
                else:
                    out[h] = str(v)
            writer.writerow(out)


def build_ranking_rows(case_summaries):
    param_cols = set()
    for item in case_summaries:
        flat = flatten_dict(item.get("overrides", {}))
        param_cols.update(flat.keys())

    pref = preferred_param_order()
    ordered = [k for k in pref if k in param_cols]
    remaining = sorted(k for k in param_cols if k not in ordered)
    param_cols = ordered + remaining

    headers = (
        [shorten_param_name(c) for c in param_cols]
        + [
            "num_seeds",
            "rmse_mean",
            "rmse_std",
            "csi_mean",
            "csi_std",
            "wallclock_mean",
            "wallclock_std",
        ]
    )

    rows = []
    for item in case_summaries:
        flat = flatten_dict(item.get("overrides", {}))
        row = {
            "num_seeds": item.get("num_runs"),
            "rmse_mean": item.get("rmse_mean"),
            "rmse_std": item.get("rmse_std"),
            "csi_mean": item.get("csi_mean"),
            "csi_std": item.get("csi_std"),
            "wallclock_mean": item.get("wallclock_mean"),
            "wallclock_std": item.get("wallclock_std"),
        }
        for c in param_cols:
            row[shorten_param_name(c)] = flat.get(c, "")
        rows.append(row)

    return headers, rows