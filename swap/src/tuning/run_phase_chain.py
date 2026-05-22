from pathlib import Path

from swap.src.post.post_common import load_yaml, merge_cfg


PHASE_ORDER = [
    "normalization_selection",
    "model_selection",
    "pretraining",
    "sample_sensitivity",
    "finetuning",
]

PHASE_BEST_FILE = {
    "normalization_selection": "best_normalization.yaml",
    "model_selection": "best_model.yaml",
    "pretraining": "best_pretrain.yaml",
    "sample_sensitivity": "best_sample.yaml",
    "finetuning": "best_finetune.yaml",
}


def get_previous_phases(phase_name: str):
    idx = PHASE_ORDER.index(phase_name)
    return PHASE_ORDER[:idx]


def build_phase_result_dir(plan: dict, project_dir, phase_name: str) -> Path:
    """
    Must match engine.py directory rule.

    results/
      <method_type>/
        <phase_name>/<measurement_type>/[nonlinear_type]
    """
    project_dir = Path(project_dir).resolve()
    root = project_dir / plan["output_dir"]

    method_type = plan["method_type"]
    measurement_type = plan["measurement_type"]
    nonlinear_type = plan.get("nonlinear_type", None)

    path = root / method_type / phase_name / measurement_type
    if measurement_type == "nonlinear" and nonlinear_type is not None:
        path = path / nonlinear_type

    return path


def load_previous_best_cfg(plan: dict, project_dir, phase_name: str):
    """
    Load and merge all previous best configs.

    Returns:
      merged_cfg: dict
      loaded_files: list[str]
    """
    merged_cfg = {}
    loaded_files = []

    for prev_phase in get_previous_phases(phase_name):
        best_name = PHASE_BEST_FILE[prev_phase]
        prev_dir = build_phase_result_dir(plan, project_dir, prev_phase)
        prev_path = prev_dir / best_name

        if prev_path.exists():
            prev_cfg = load_yaml(prev_path)
            merged_cfg = merge_cfg(merged_cfg, prev_cfg)
            loaded_files.append(str(prev_path))

    return merged_cfg, loaded_files