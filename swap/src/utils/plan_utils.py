# plan helpers
# - apply CLI values to plan
# - build dynamic/config names
# - resolve default plan paths
# - build resolved cfg
# - build output/measurement/phase/report paths

import re
from copy import deepcopy
from pathlib import Path

from swap.src.utils.yaml_utils import load_yaml, merge_cfg
from swap.src.utils.phase_utils import phase_to_config_dirname


# ---------------------------------------------------------------------
# basic dict helpers
# ---------------------------------------------------------------------

def deep_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
    # deep_get(cfg, "model", "dim") -> 128


def deep_set(d: dict, dotted_key: str, value):
    keys = dotted_key.split(".")   # "measurement.stride" -> ["measurement", "stride"]
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value
    # deep_set(cfg, "measurement.stride", 4)


def deep_get_by_dotted_key(d: dict, dotted_key: str, default=None):
    cur = d
    for k in dotted_key.split("."):   # "model.hidden_dim" -> ["model", "hidden_dim"]
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
    # deep_get_by_dotted_key(cfg, "model.hidden_dim") -> 256


# ---------------------------------------------------------------------
# make_* : names / cfg labels
# ---------------------------------------------------------------------

def make_dynamic_name(dynamic_type: str, dim=None, measurement_type=None) -> str:
    if dim is None:
        name = dynamic_type                   # "kolmogorov"
    else:
        name = f"{dynamic_type}_{dim}"        # "kolmogorov_128"

    if measurement_type == "nonlinear":
        name = f"{name}_nonlinear"            # "kolmogorov_128_nonlinear"

    return name


def make_default_plan_name(phase: str, dynamic_type: str, dim: int, method_type: str = None) -> str:
    if phase in {
        "normalization_selection",
        "pretraining",
        "finetuning",
        "sample_sensitivity",
    }:
        return phase                         # "finetuning"

    if phase == "model_selection":
        return make_dynamic_name(dynamic_type, dim)   # "kolmogorov_128"

    if phase == "baseline_tuning":
        if method_type is None:
            raise ValueError("baseline_tuning requires --method_type")
        return f"{method_type}_tuning"       # "enkf_tuning"

    raise ValueError(f"Unknown phase: {phase}")


def make_measurement_setting_name(cfg: dict) -> str:
    measurement = cfg.get("measurement", {}) or {}
    measurement_type = measurement.get("type")             # "grid_mask"
    nonlinear_type = measurement.get("nonlinear_type")     # "sigmoid"

    if measurement_type == "grid_mask":
        stride = measurement.get("stride")
        return f"stride_{stride}"                          # "stride_4"

    if measurement_type == "center_mask":
        hole_ratio = measurement.get("hole_ratio")
        return f"hole_{hole_ratio}"                        # "hole_0.5"

    if measurement_type == "low_resolution":
        scale_factor = measurement.get("scale_factor")
        return f"scale_{scale_factor}"                     # "scale_8"

    if measurement_type == "nonlinear":
        alpha = measurement.get("alpha")
        if alpha is None:
            return str(nonlinear_type)                     # "sigmoid"
        return f"alpha_{alpha}"                            # "alpha_0.1"

    return "default"


# ---------------------------------------------------------------------
# plan -> cfg
# ---------------------------------------------------------------------

def apply_cli_to_plan(plan: dict, args):
    if "dynamic_type" not in plan:
        plan["dynamic_type"] = args.dynamic_type
    if "measurement_type" not in plan:
        plan["measurement_type"] = args.measurement_type
    if "method_type" not in plan:
        plan["method_type"] = args.method_type
    if "nonlinear_type" not in plan:
        plan["nonlinear_type"] = args.nonlinear_type
    if "dim" not in plan:
        plan["dim"] = args.dim
    if "output_dir" not in plan:
        plan["output_dir"] = "results"        # default relative output dir

    if args.dynamic_type is not None:
        plan["dynamic_type"] = args.dynamic_type
    if args.measurement_type is not None:
        plan["measurement_type"] = args.measurement_type
    if args.method_type is not None:
        plan["method_type"] = args.method_type
    if args.nonlinear_type is not None:
        plan["nonlinear_type"] = args.nonlinear_type
    if args.dim is not None:
        plan["dim"] = args.dim
    if args.out_dir is not None:
        plan["output_dir"] = args.out_dir     # "0412_test"

    plan["dynamic_name"] = make_dynamic_name(
        plan["dynamic_type"],                 # "kolmogorov"
        plan["dim"],                          # 128
        plan["measurement_type"],             # "grid_mask"
    )   # "kolmogorov_128"

    return plan


def build_resolved_base_cfg(plan: dict, project_dir) -> dict:
    cfg_root = Path(project_dir) / "configs"   # /.../masf/configs

    base_yaml = load_yaml(cfg_root / "base.yaml")
    dyn_cfg = load_yaml(cfg_root / "dynamics" / f"{plan['dynamic_name']}.yaml")
    meas_cfg = load_yaml(cfg_root / "measurements" / f"{plan['measurement_type']}.yaml")
    method_cfg = load_yaml(cfg_root / "methods" / f"{plan['method_type']}.yaml")

    cfg = deepcopy(base_yaml)
    cfg = merge_cfg(cfg, meas_cfg)
    cfg = merge_cfg(cfg, dyn_cfg)
    cfg = merge_cfg(cfg, method_cfg)

    if plan.get("nonlinear_type") is not None:
        deep_set(cfg, "measurement.nonlinear_type", plan["nonlinear_type"])
        # cfg["measurement"]["nonlinear_type"] = "sigmoid"

    for key, value in plan.get("fixed", {}).items():
        deep_set(cfg, key, value)
        # "train.lr", 1e-4

    return cfg


# ---------------------------------------------------------------------
# get_* : plan / path resolution
# ---------------------------------------------------------------------

def get_default_plan_path(
    configs_dir: Path,
    phase: str,
    dynamic_type: str,
    dim: int,
    method_type: str = None,
) -> Path:
    phase_dir = phase_to_config_dirname(phase)
    plan_name = make_default_plan_name(
        phase=phase,
        dynamic_type=dynamic_type,
        dim=dim,
        method_type=method_type)

    # 1) method-specific override first
    if method_type is not None:
        method_path = configs_dir / phase_dir / str(method_type).lower() / f"{plan_name}.yaml"
        if method_path.exists():
            return method_path

    # 2) fallback to original shared config
    return configs_dir / phase_dir / f"{plan_name}.yaml"

def get_plan_axes(plan):
    measurement_type = (
        deep_get(plan, "measurement_type")
        or deep_get(plan, "measurement", "type")
        or "grid_mask"
    )   # "grid_mask"

    method_type = (
        deep_get(plan, "method_type")
        or deep_get(plan, "method", "type")
        or deep_get(plan, "method", "name")
        or "ours"
    )   # "ours"

    nonlinear_type = (
        deep_get(plan, "nonlinear_type")
        or deep_get(plan, "measurement", "nonlinear_type")
        or None
    )   # "sigmoid" or None

    dynamic_name = (
        deep_get(plan, "dynamic_name")
        or deep_get(plan, "dynamic_type")
        or "unknown_dynamic"
    )   # "kolmogorov_128"

    return {
        "measurement_type": slugify(measurement_type),
        "method_type": slugify(method_type),
        "nonlinear_type": None if nonlinear_type is None else slugify(nonlinear_type),
        "dynamic_name": slugify(dynamic_name),
    }


def get_output_root(plan, project_dir) -> Path:
    output_dir = Path(plan["output_dir"])                # "0412_test" or "/data3/dwkim/results"
    if output_dir.is_absolute():
        return output_dir
    return Path(project_dir).resolve() / output_dir
    # /.../masf + "0412_test" -> /.../masf/0412_test


def get_measurement_root(plan, project_dir) -> Path:
    root = get_output_root(plan, project_dir)                               # /.../masf/0412_test
    axes = get_plan_axes(plan)                                              # {"method_type": "ours", ...}
    resolved_cfg = build_resolved_base_cfg(plan, project_dir)               # merged cfg dict
    measurement_setting_name = make_measurement_setting_name(resolved_cfg)  # "stride_4"

    measurement_root = (
        root
        / axes["method_type"]                                               # /ours
        / axes["dynamic_name"]                                              # /kolmogorov_128
        / axes["measurement_type"]                                          # /grid_mask
    )

    if axes["measurement_type"] == "nonlinear" and axes["nonlinear_type"] is not None:
        measurement_root = measurement_root / axes["nonlinear_type"]
        # /.../nonlinear/sigmoid

    measurement_root = measurement_root / slugify(measurement_setting_name)
    # /.../grid_mask/stride_4

    return measurement_root
    # /.../masf/0412_test/ours/kolmogorov_128/grid_mask/stride_4

# dir 
def get_phase_root(plan, project_dir, phase_name) -> Path:
    phase_root = get_measurement_root(plan, project_dir) / phase_name
    # normal:
    # /.../stride_10/normalization_selection
    # baseline:
    # /.../stride_10/baseline_tuning

    if phase_name == "baseline_tuning":
        suffix = build_suffix_from_plan(plan, "dynamics.num_samples", "num_samples")
        if suffix is not None:
            phase_root = phase_root / suffix
            # /.../stride_10/baseline_tuning/num_samples_10

    return phase_root

def get_report_root(plan, project_dir) -> Path:
    return get_output_root(plan, project_dir)
    # /.../masf/0412_test


def force_all_n_trials_to_one(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "n_trials":
                obj[k] = 1
            else:
                force_all_n_trials_to_one(v)
    elif isinstance(obj, list):
        for item in obj:
            force_all_n_trials_to_one(item)



# ---------------------------------------------------------------------
# make_* : path suffix / dirs
# ---------------------------------------------------------------------

def make_result_dirs(plan, project_dir, phase_name):
    result_dir = get_phase_root(plan, project_dir, phase_name)
    experiments_dir = result_dir / "experiments"
    configs_dir = result_dir / "configs"

    result_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    axes = get_plan_axes(plan)
    resolved_cfg = build_resolved_base_cfg(plan, project_dir)
    measurement_setting_name = make_measurement_setting_name(resolved_cfg)

    return {
        "result_dir": result_dir,
        "experiments_dir": experiments_dir,
        "configs_dir": configs_dir,
        "method_type": axes["method_type"],                       # "ours"
        "dynamic_name": axes["dynamic_name"],                     # "kolmogorov_128"
        "measurement_type": axes["measurement_type"],             # "grid_mask"
        "nonlinear_type": axes["nonlinear_type"],                 # None or "sigmoid"
        "measurement_setting_name": slugify(measurement_setting_name),  # "stride_4"
    }


# ---------------------------------------------------------------------
# misc path helpers
# ---------------------------------------------------------------------

def resolve_swap_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent
    # /.../masf/swap/src/utils/plan_utils.py -> /.../masf/swap


def resolve_configs_dir(swap_dir: Path) -> Path:
    return swap_dir / "configs"
    # /.../masf/swap/configs


def slugify(value: str) -> str:
    value = str(value).strip().lower()                    # " Grid Mask " -> "grid mask"
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)       # "grid mask" -> "grid_mask"
    value = re.sub(r"_+", "_", value).strip("_")          # "__a__b__" -> "a_b"
    return value or "unknown"
    # "Sigmoid / Test" -> "sigmoid_test"


def build_suffix_from_plan(plan: dict, key: str, keyword: str) -> str | None:
    fixed = plan.get("fixed", {}) or {}

    if key not in fixed:
        return None

    value = fixed[key]   # 10 or [10, 20]
    if isinstance(value, (list, tuple)):
        value = "-".join(str(x) for x in value)

    return f"{keyword}_{slugify(value)}"
    # 10 -> "num_samples_10"
