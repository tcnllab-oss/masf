from pathlib import Path
import shutil
from swap.src.utils.yaml_utils import load_yaml

PHASE_ORDER = [
    "normalization_selection",
    "model_selection",
    "pretraining",
    "sample_sensitivity",
    "finetuning",
]

ALL_PHASES = PHASE_ORDER + [
    "baseline_tuning",
]

PHASE_DEPENDENCIES = {
    "normalization_selection": [],
    "model_selection": ["best_normalization.yaml"],
    "pretraining": ["best_normalization.yaml", "best_model.yaml"],
    "sample_sensitivity": ["best_normalization.yaml", "best_model.yaml", "best_pretrain.yaml"],
    "finetuning": ["best_normalization.yaml", "best_model.yaml", "best_pretrain.yaml", "best_sample.yaml"],
    "baseline_tuning": [],
}

PHASE_TO_CONFIG_DIRNAME = {
    "normalization_selection": "normalization",
    "model_selection": "model",
    "pretraining": "pretraining",
    "sample_sensitivity": "sample_sensitivity",
    "finetuning": "finetuning",
    "baseline_tuning": "baseline",
}


def phase_to_config_dirname(phase: str) -> str:
    if phase not in PHASE_TO_CONFIG_DIRNAME:
        raise ValueError(f"Unknown phase: {phase}")
    return PHASE_TO_CONFIG_DIRNAME[phase]


def get_prev_phase_name(phase_name: str):
    if phase_name not in PHASE_ORDER:
        return None
    idx = PHASE_ORDER.index(phase_name)
    if idx == 0:
        return None
    return PHASE_ORDER[idx - 1]


def get_best_filename_for_phase(phase_name: str, method_type: str = None) -> str:
    if phase_name == "normalization_selection":
        return "best_normalization.yaml"
    if phase_name == "model_selection":
        return "best_model.yaml"
    if phase_name == "pretraining":
        return "best_pretrain.yaml"
    if phase_name == "sample_sensitivity":
        return "best_sample.yaml"
    if phase_name == "finetuning":
        return "best_finetune.yaml"
    if phase_name == "baseline_tuning":
        if method_type is None:
            return "best_baseline.yaml"
        return f"best_{method_type}.yaml"
    raise ValueError(f"Unknown phase: {phase_name}")


def parse_txt_table(path: Path):
    if not path.exists():
        raise FileNotFoundError(str(path))

    text = path.read_text(encoding="utf-8").strip()
    lines = [line.rstrip("\n") for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"Invalid table format: {path}")

    headers = [x.strip() for x in lines[0].split("|")]
    rows = []
    for line in lines[2:]:
        parts = [x.strip() for x in line.split("|")]
        if len(parts) != len(headers):
            continue
        rows.append(dict(zip(headers, parts)))
    return headers, rows


def best_yaml_name_for_phase(phase: str, method_type: str) -> str:
    if phase == "normalization_selection":
        return "best_normalization.yaml"
    if phase == "model_selection":
        return "best_model.yaml"
    if phase == "pretraining":
        return "best_pretrain.yaml"
    if phase == "sample_sensitivity":
        return "best_sample.yaml"
    if phase == "finetuning":
        return "best_finetune.yaml"
    if phase == "baseline_tuning":
        return f"best_{method_type}.yaml"
    raise ValueError(f"Unknown phase: {phase}")


def promote_seed_eval_best_to_phase_best(measurement_root: Path, phase: str, method_type: str):
    seed_eval_root = measurement_root / phase / "seed_eval_topk"
    ranking_yaml = seed_eval_root / "ranking_summary.yaml"

    if not ranking_yaml.exists():
        print(f"[SEED_EVAL_PROMOTE] skip: missing {ranking_yaml}")
        return

    summary = load_yaml(ranking_yaml)
    candidates = summary.get("candidates", [])
    if not candidates:
        print(f"[SEED_EVAL_PROMOTE] skip: no candidates in {ranking_yaml}")
        return

    valid_candidates = []
    for cand in candidates:
        rmse_mean = cand.get("rmse_mean")
        try:
            rmse_mean = float(rmse_mean)
            if math.isnan(rmse_mean):
                continue
        except Exception:
            continue
        valid_candidates.append(cand)

    if not valid_candidates:
        print(f"[SEED_EVAL_PROMOTE] skip: no valid rmse_mean in {ranking_yaml}")
        return

    best_cand = min(valid_candidates, key=lambda x: float(x["rmse_mean"]))
    best_cfg_path = Path(best_cand["base_candidate_cfg"])

    if not best_cfg_path.exists():
        raise FileNotFoundError(
            f"[SEED_EVAL_PROMOTE] best candidate cfg not found: {best_cfg_path}"
        )

    phase_dir = measurement_root / phase
    out_best_yaml = phase_dir / best_yaml_name_for_phase(phase, method_type)

    shutil.copy2(best_cfg_path, out_best_yaml)

    print("=" * 80)
    print(f"[SEED_EVAL_PROMOTE] phase        : {phase}")
    print(f"[SEED_EVAL_PROMOTE] selected     : {best_cand.get('candidate_name')}")
    print(f"[SEED_EVAL_PROMOTE] rmse_mean    : {best_cand.get('rmse_mean')}")
    print(f"[SEED_EVAL_PROMOTE] source_cfg   : {best_cfg_path}")
    print(f"[SEED_EVAL_PROMOTE] overwrite    : {out_best_yaml}")
    print("=" * 80)