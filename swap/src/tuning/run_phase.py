# tuning phase runner
# - build optuna study
# - save best config
# - compare with previous phase best
# - optionally run extra trials if current phase is worse

import json
import hashlib
from copy import deepcopy
from pathlib import Path
from math import prod

import optuna
from optuna.trial import TrialState

from swap.src.utils.yaml_utils import load_yaml, save_yaml, set_nested, merge_cfg
from swap.src.utils.dependency_utils import (
    get_prev_phase_best_cfg_path,
    load_previous_best_configs,
)
from swap.src.utils.io_utils import init_text_file, append_text
from swap.src.tuning.evaluate_objective import make_objective
from swap.src.utils.plan_utils import get_phase_root, make_result_dirs
from swap.src.utils.phase_utils import get_best_filename_for_phase, get_prev_phase_name
from swap.src.tuning.build_search_space import make_effective_param_specs


def make_search_space_signature(plan, project_dir, phase_name):
    effective_params = make_effective_param_specs(plan, project_dir, phase_name)

    payload = {
        "phase": phase_name,
        "dynamic_name": plan.get("dynamic_name"),
        "dynamic_type": plan.get("dynamic_type"),
        "dim": plan.get("dim"),
        "method_type": plan.get("method_type"),
        "measurement_type": plan.get("measurement_type"),
        "nonlinear_type": plan.get("nonlinear_type"),
        "fixed": plan.get("fixed", {}),
        "warm_start_from_prev_cfg": bool(plan.get("warm_start_from_prev_cfg", False)),
        "warm_start": plan.get("warm_start", {}),
        "grid_search": bool(plan.get("search", {}).get(phase_name, {}).get("grid_search", False)),
        "params": effective_params,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def make_dependency_signature(plan, project_dir, phase_name):
    deps_payload = {}
    prev_best_cfg, loaded_infos = load_previous_best_configs(plan, project_dir, phase_name)

    if loaded_infos:
        for info in loaded_infos:
            deps_payload[str(info["path"])] = load_yaml(info["path"])
    else:
        prev_phase = get_prev_phase_name(phase_name)
        if prev_phase is not None:
            prev_phase_root = get_phase_root(plan, project_dir, prev_phase)
            prev_best_filename = get_best_filename_for_phase(prev_phase, plan.get("method_type"))
            deps_payload[str(prev_phase_root / prev_best_filename)] = None

    text = json.dumps(deps_payload, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def build_grid_search_space(param_specs: dict):
    search_space = {}

    for key, spec in param_specs.items():
        ptype = spec["type"]

        if ptype == "categorical":
            search_space[key] = list(spec["values"])

        elif ptype == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            search_space[key] = list(range(low, high + 1, step))

        elif ptype == "float":
            if "step" not in spec:
                raise ValueError(
                    f"grid_search=True requires 'step' for float param: {key}"
                )
            low = float(spec["low"])
            high = float(spec["high"])
            step = float(spec["step"])

            vals = []
            cur = low
            eps = step * 1e-9
            while cur <= high + eps:
                vals.append(round(cur, 10))
                cur += step
            search_space[key] = vals

        else:
            raise ValueError(f"Unsupported param type for grid search: {key} ({ptype})")

    return search_space


def create_or_load_study(result_dir: Path, study_name: str, sampler):
    storage = f"sqlite:///{result_dir / 'optuna_search.db'}"

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )
    return study, storage


def save_best_cfg(plan, project_dir, phase_name, study, out_name, output_dir=None):
    cfg_root = Path(project_dir) / "configs"

    base_yaml = load_yaml(cfg_root / "base.yaml")
    dyn_cfg = load_yaml(cfg_root / "dynamics" / f"{plan['dynamic_name']}.yaml")
    meas_cfg = load_yaml(cfg_root / "measurements" / f"{plan['measurement_type']}.yaml")
    method_cfg = load_yaml(cfg_root / "methods" / f"{plan['method_type']}.yaml")

    base_cfg = deepcopy(base_yaml)
    base_cfg = merge_cfg(base_cfg, meas_cfg)
    base_cfg = merge_cfg(base_cfg, dyn_cfg)
    base_cfg = merge_cfg(base_cfg, method_cfg)

    if plan.get("nonlinear_type") is not None:
        set_nested(base_cfg, "measurement.nonlinear_type", plan["nonlinear_type"])

    prev_best_cfg, _ = load_previous_best_configs(plan, project_dir, phase_name)
    if prev_best_cfg:
        base_cfg = merge_cfg(base_cfg, prev_best_cfg)

    fixed_cfg = deepcopy(base_cfg)
    for key, value in plan.get("fixed", {}).items():
        set_nested(fixed_cfg, key, value)

    best_updates = study.best_trial.user_attrs["updates"]
    best_cfg = merge_cfg(fixed_cfg, best_updates)

    if output_dir is None:
        dirs = make_result_dirs(plan, project_dir, phase_name)
        output_dir = dirs["result_dir"]

    save_yaml(best_cfg, output_dir / out_name)
    return best_cfg


def get_phase_trials_jsonl(plan, project_dir, phase_name) -> Path:
    phase_dir = get_phase_root(plan, project_dir, phase_name)
    return phase_dir / f"{phase_name}_trials.jsonl"


def get_best_rmse_from_trials_jsonl(trials_jsonl: Path):
    if not trials_jsonl.exists():
        return None

    best_rmse = None
    with open(trials_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception:
                continue

            if rec.get("returncode", 1) != 0:
                continue

            rmse = rec.get("rmse")
            if rmse is None:
                continue

            try:
                rmse = float(rmse)
            except Exception:
                continue

            if rmse >= 1e8:
                continue

            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse

    return best_rmse


def save_phase_best_with_fallback(plan, project_dir, phase_name, study, output_dir=None, progress_txt=None):
    current_best_filename = get_best_filename_for_phase(phase_name, plan.get("method_type"))

    if output_dir is None:
        dirs = make_result_dirs(plan, project_dir, phase_name)
        output_dir = dirs["result_dir"]

    current_cfg = save_best_cfg(
        plan=plan,
        project_dir=project_dir,
        phase_name=phase_name,
        study=study,
        out_name="__tmp_current_best__.yaml",
        output_dir=output_dir,
    )

    current_best_path = output_dir / current_best_filename
    tmp_current_best_path = output_dir / "__tmp_current_best__.yaml"

    current_trials_jsonl = output_dir / f"{phase_name}_trials.jsonl"
    current_best_rmse = get_best_rmse_from_trials_jsonl(current_trials_jsonl)

    prev_phase, prev_best_cfg_path = get_prev_phase_best_cfg_path(plan, project_dir, phase_name)

    if prev_phase is None or prev_best_cfg_path is None:
        if tmp_current_best_path.exists():
            tmp_current_best_path.replace(current_best_path)

        msg = f"[BEST_SELECT] phase={phase_name} no previous phase fallback. keep current best."
        print(msg)
        if progress_txt is not None:
            append_text(progress_txt, msg)
        return current_cfg, current_best_path

    prev_phase_root = get_phase_root(plan, project_dir, prev_phase)
    prev_trials_jsonl = prev_phase_root / f"{prev_phase}_trials.jsonl"
    prev_best_rmse = get_best_rmse_from_trials_jsonl(prev_trials_jsonl)

    use_previous = False
    if current_best_rmse is not None and prev_best_rmse is not None and current_best_rmse > prev_best_rmse:
        use_previous = True

    if use_previous:
        prev_cfg = load_yaml(prev_best_cfg_path)
        save_yaml(prev_cfg, current_best_path)

        if tmp_current_best_path.exists():
            tmp_current_best_path.unlink()

        msg_lines = [
            "=" * 80,
            f"[BEST_SELECT] phase={phase_name}",
            "[BEST_SELECT] current best is worse than previous phase best",
            f"[BEST_SELECT] current_best_rmse = {current_best_rmse}",
            f"[BEST_SELECT] prev_phase        = {prev_phase}",
            f"[BEST_SELECT] prev_best_rmse    = {prev_best_rmse}",
            f"[BEST_SELECT] use previous cfg  = {prev_best_cfg_path}",
            f"[BEST_SELECT] saved to          = {current_best_path}",
            "=" * 80,
        ]
        for line in msg_lines:
            print(line)
            if progress_txt is not None:
                append_text(progress_txt, line)

        return prev_cfg, current_best_path

    if tmp_current_best_path.exists():
        tmp_current_best_path.replace(current_best_path)

    msg_lines = [
        "=" * 80,
        f"[BEST_SELECT] phase={phase_name}",
        "[BEST_SELECT] keep current phase best",
        f"[BEST_SELECT] current_best_rmse = {current_best_rmse}",
        f"[BEST_SELECT] saved to          = {current_best_path}",
        "=" * 80,
    ]
    for line in msg_lines:
        print(line)
        if progress_txt is not None:
            append_text(progress_txt, line)

    return current_cfg, current_best_path


def run_phase(plan, project_dir, phase_name):
    project_dir = Path(project_dir).resolve()

    dirs = make_result_dirs(plan, project_dir, phase_name)
    result_dir = dirs["result_dir"]
    experiments_dir = dirs["experiments_dir"]
    configs_dir = dirs["configs_dir"]
    measurement_type = dirs["measurement_type"]
    method_type = dirs["method_type"]
    nonlinear_type = dirs["nonlinear_type"]
    dynamic_name = dirs["dynamic_name"]
    measurement_setting_name = dirs["measurement_setting_name"]

    plan["result_dir"] = str(result_dir)
    plan["experiments_dir"] = str(experiments_dir)
    plan["configs_dir"] = str(configs_dir)

    save_yaml(plan, result_dir / "resolved_plan.yaml")

    progress_txt = result_dir / "progress.txt"
    summary_txt = result_dir / "summary.txt"

    if not progress_txt.exists():
        init_text_file(progress_txt, "# progress log")

    if not summary_txt.exists():
        init_text_file(summary_txt, "# local_idx trial_number run_name rmse csi wallclock user sys returncode updates_json")

    phase_cfg = plan["search"][phase_name]
    effective_param_specs = make_effective_param_specs(plan, project_dir, phase_name)
    use_grid_search = bool(phase_cfg.get("grid_search", False))

    search_sig = make_search_space_signature(plan, project_dir, phase_name)
    dependency_sig = make_dependency_signature(plan, project_dir, phase_name)

    if nonlinear_type is not None:
        study_name = f"{dynamic_name}_{method_type}_{measurement_type}_{nonlinear_type}_{measurement_setting_name}_{phase_name}_{search_sig}_{dependency_sig}"
    else:
        study_name = f"{dynamic_name}_{method_type}_{measurement_type}_{measurement_setting_name}_{phase_name}_{search_sig}_{dependency_sig}"

    if use_grid_search:
        search_space = build_grid_search_space(effective_param_specs)
        sampler = optuna.samplers.GridSampler(search_space)
        target_trials = prod(len(v) for v in search_space.values())

        print("=" * 80)
        print(f"[SEARCH] phase        : {phase_name}")
        print(f"[SEARCH] mode         : grid")
        print(f"[SEARCH] total_trials : {target_trials}")
        print(f"[SEARCH] search_space : {search_space}")
        print("=" * 80)

        append_text(progress_txt, f"[SEARCH] phase={phase_name} mode=grid total_trials={target_trials}")
    else:
        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
        target_trials = int(phase_cfg.get("n_trials", 20))

        print("=" * 80)
        print(f"[SEARCH] phase        : {phase_name}")
        print(f"[SEARCH] mode         : optuna")
        print(f"[SEARCH] total_trials : {target_trials}")
        print("=" * 80)

        append_text(progress_txt, f"[SEARCH] phase={phase_name} mode=optuna total_trials={target_trials}")

    study, storage = create_or_load_study(
        result_dir=result_dir,
        study_name=study_name,
        sampler=sampler,
    )

    completed_trials = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    remaining_trials = max(0, target_trials - completed_trials)

    start_msg = (
        f"trials={completed_trials}/{target_trials} "
        f"remaining={remaining_trials} "
    )
    append_text(progress_txt, start_msg)

    if remaining_trials > 0:
        objective = make_objective(
            plan,
            project_dir,
            phase_name,
            progress_txt,
            summary_txt,
            param_specs_override=effective_param_specs,
            enqueue_prev_cfg_first=bool(plan.get("warm_start_from_prev_cfg", False)),
        )
        study.optimize(objective, n_trials=remaining_trials)
    else:
        msg = f"END already reached target completed trials: {target_trials}"
        append_text(progress_txt, msg)
        print(msg)

    completed = [
        t for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None and t.value < 1e8
    ]
    if not completed:
        append_text(progress_txt, "END no valid run found")
        print("No valid run found.")
        return

    selected_cfg, selected_cfg_path = save_phase_best_with_fallback(
        plan=plan,
        project_dir=project_dir,
        phase_name=phase_name,
        study=study,
        output_dir=result_dir,
        progress_txt=progress_txt,
    )

    append_text(progress_txt, f"END best_value={study.best_value} best_trial={study.best_trial.number}")
    append_text(progress_txt, f"[BEST_SELECT] phase={phase_name} selected_cfg={selected_cfg_path}")

    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_params)
    print(f"Saved selected best cfg to: {selected_cfg_path}")