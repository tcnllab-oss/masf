# yaml/config helpers
# - load/save yaml
# - nested dict access
# - config merge/validation
# - trial/run name formatting

import yaml
from copy import deepcopy
from pathlib import Path
import optuna


def load_yaml(path):
    path = Path(path)   # "configs/base.yaml" -> Path("configs/base.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)
    # load_yaml("configs/base.yaml") -> dict


def save_yaml(obj, path):
    path = Path(path)   # "reports/tmp/config.yaml" -> Path(...)
    path.parent.mkdir(parents=True, exist_ok=True)   # create parent dirs if needed
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    # save_yaml(cfg, "reports/tmp/config.yaml")


def set_nested(d, key, value):
    parts = key.split(".")   # "train.lr" -> ["train", "lr"]
    cur = d
    for p in parts[:-1]:
        if p not in cur or cur[p] is None:
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    # set_nested(cfg, "train.lr", 1e-4)
    # cfg["train"]["lr"] = 1e-4


def get_nested(d, key, default=None):
    parts = key.split(".")   # "model.hidden_dim" -> ["model", "hidden_dim"]
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur
    # get_nested(cfg, "model.hidden_dim") -> 256


def merge_cfg(base, updates):
    out = deepcopy(base)   # keep original base unchanged

    def rec(dst, src):
        for k, v in src.items():
            if isinstance(v, dict):
                if k not in dst or dst[k] is None:
                    dst[k] = {}
                rec(dst[k], v)   # recursive merge for nested dict
            else:
                dst[k] = v       # overwrite scalar/list values

    rec(out, updates)
    return out


def validate_cfg(cfg):
    smin = get_nested(cfg, "sample.s_scale_min")   # e.g. 0.1
    smax = get_nested(cfg, "sample.s_scale_max")   # e.g. 1.0
    gmin = get_nested(cfg, "sample.g_scale_min")   # e.g. 0.2
    gmax = get_nested(cfg, "sample.g_scale_max")   # e.g. 2.0

    if smin is not None and smax is not None and smax <= smin:
        return False
    if gmin is not None and gmax is not None and gmax <= gmin:
        return False
    return True
    # True if min/max ranges are valid


def short_key(key):
    return (
        key.replace("model.", "")            # "model.hidden_dim" -> "hidden_dim"
           .replace("pretrain.", "pt_")      # "pretrain.lr" -> "pt_lr"
           .replace("train.online.", "on_")  # "train.online.steps" -> "on_steps"
           .replace("train.", "tr_")         # "train.lr" -> "tr_lr"
           .replace("dynamics.", "dyn_")     # "dynamics.dt" -> "dyn_dt"
           .replace("measurement.", "meas_") # "measurement.stride" -> "meas_stride"
           .replace("steps.", "st_")         # "steps.n_inner" -> "st_n_inner"
    )


def tagify(v):
    if isinstance(v, list):
        s = "-".join(str(x) for x in v)   # [1, 2, 4] -> "1-2-4"
    else:
        s = str(v)                        # 0.1 -> "0.1"

    s = s.replace("/", "_").replace(" ", "").replace(".", "p")
    s = s.replace("[", "").replace("]", "").replace(",", "-")
    return s
    # 0.1 -> "0p1"
    # "a/b" -> "a_b"
    # [1, 2] -> "1-2"


def make_run_name(base_name, updates, trial_number):
    flat = []

    def walk(prefix, obj):
        for k, v in obj.items():
            kk = f"{prefix}.{k}" if prefix else k   # "train" + "lr" -> "train.lr"
            if isinstance(v, dict):
                walk(kk, v)
            else:
                flat.append((kk, v))   # [("train.lr", 1e-4), ("model.hidden_dim", 256)]

    walk("", updates)

    parts = [f"{base_name}_trial{trial_number:04d}"]   # "finetuning_trial0003"
    for k, v in sorted(flat):
        parts.append(f"{short_key(k)}{tagify(v)}")
        # "train.lr" + 1e-4 -> "tr_lr1e-04" or formatted tag string

    return "_".join(parts)
    # make_run_name("finetuning", {"train": {"lr": 1e-4}, "model": {"hidden_dim": 256}}, 3)
    # -> "finetuning_trial0003_hidden_dim256_tr_lr0p0001"