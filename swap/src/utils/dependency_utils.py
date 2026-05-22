from swap.src.utils.yaml_utils import load_yaml, merge_cfg
from swap.src.utils.phase_utils import (
    PHASE_DEPENDENCIES,
    get_best_filename_for_phase,
    get_prev_phase_name,
)
from swap.src.utils.plan_utils import get_phase_root

def summarize_dependency_cfg_by_phase(phase_name: str, cfg: dict) -> dict:
    summary = {}

    measurement = cfg.get("measurement", {})
    model = cfg.get("model", {})
    pretrain = cfg.get("pretrain", {})
    sample = cfg.get("sample", {})
    train = cfg.get("train", {})
    online = train.get("online", {})

    if phase_name == "normalization_selection":
        keys = [
            "same_normalization",
            "normalization_form",
            "stats_mode",
            "stats_update_mode",
            "momentum",
            "nonlinear_type",
        ]
        for key in keys:
            if key in measurement:
                summary[f"measurement.{key}"] = measurement[key]

    elif phase_name == "model_selection":
        keys = [
            "model_channels",
            "num_res_blocks",
            "attention_resolutions",
            "channel_mult",
        ]
        for key in keys:
            if key in model:
                summary[f"model.{key}"] = model[key]

    elif phase_name == "pretraining":
        keys = [
            "batch_size",
            "epoch",
            "lr",
            "weight_decay",
            "betas",
        ]
        for key in keys:
            if key in pretrain:
                summary[f"pretrain.{key}"] = pretrain[key]

    elif phase_name == "sample_sensitivity":
        keys = [
            "nfe",
            "s_scale_min",
            "s_scale_max",
            "s_scale_power",
            "g_scale_min",
            "g_scale_max",
            "g_scale_power",
            "terminal_time",
        ]
        for key in keys:
            if key in sample:
                summary[f"sample.{key}"] = sample[key]

    elif phase_name == "finetuning":
        train_keys = [
            "pretrained",
            "override",
            "epoch",
            "batch_size",
            "lr",
            "weight_decay",
            "betas",
        ]
        online_keys = [
            "warmup_steps",
            "full_epoch",
            "ft_epoch",
            "ft_lr_scale",
            "skip_if_small_drift",
            "drift_threshold",
        ]

        for key in train_keys:
            if key in train:
                summary[f"train.{key}"] = train[key]

        for key in online_keys:
            if key in online:
                summary[f"train.online.{key}"] = online[key]

    return summary


def load_previous_best_configs(plan, project_dir, phase_name):
    merged = {}
    loaded = []

    for filename in PHASE_DEPENDENCIES.get(phase_name, []):
        if filename == "best_normalization.yaml":
            prev_phase = "normalization_selection"
        elif filename == "best_model.yaml":
            prev_phase = "model_selection"
        elif filename == "best_pretrain.yaml":
            prev_phase = "pretraining"
        elif filename == "best_sample.yaml":
            prev_phase = "sample_sensitivity"
        else:
            continue

        prev_path = get_phase_root(plan, project_dir, prev_phase) / filename
        if prev_path.exists():
            prev_cfg = load_yaml(prev_path)
            merged = merge_cfg(merged, prev_cfg)
            loaded.append({
                "phase": prev_phase,
                "path": prev_path,
                "summary": summarize_dependency_cfg_by_phase(prev_phase, prev_cfg),
            })

    return merged, loaded


def get_prev_phase_best_cfg_path(plan, project_dir, phase_name):
    prev_phase = get_prev_phase_name(phase_name)
    if prev_phase is None:
        return None, None

    prev_phase_root = get_phase_root(plan, project_dir, prev_phase)
    prev_best_filename = get_best_filename_for_phase(prev_phase, plan.get("method_type"))
    prev_best_cfg_path = prev_phase_root / prev_best_filename

    if not prev_best_cfg_path.exists():
        return prev_phase, None

    return prev_phase, prev_best_cfg_path