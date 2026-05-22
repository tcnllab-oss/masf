# optuna objective builder
# - sample trial parameters
# - reuse duplicate / existing trials when possible
# - launch main.py for new trials
# - save config/result/log files for each trial

import json
import math
import shutil
import subprocess
from pathlib import Path
from copy import deepcopy

from swap.src.utils.yaml_utils import load_yaml, save_yaml, set_nested, merge_cfg, validate_cfg
from swap.src.utils.plan_utils import build_resolved_base_cfg, deep_get_by_dotted_key
from swap.src.tuning.decode_trial_config import decode_special_value
from swap.src.utils.dependency_utils import load_previous_best_configs
from swap.src.utils.io_utils import parse_log_metrics, write_jsonl, append_text


def suggest_value(trial, name, spec):
    # Sample one value from Optuna according to the parameter spec.
    ptype = spec["type"]  # "categorical" / "int" / "float"

    if ptype == "categorical":
        if "values" not in spec:
            raise ValueError(f"{name}: categorical spec must contain 'values'")
        return trial.suggest_categorical(name, spec["values"])
        # ["relu", "gelu"] -> "relu"

    if ptype == "int":
        if "low" not in spec or "high" not in spec:
            raise ValueError(f"{name}: int spec must contain 'low' and 'high'. Got keys={list(spec.keys())}")
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        log = bool(spec.get("log", False))
        if log:
            return trial.suggest_int(name, low, high, log=True)
        return trial.suggest_int(name, low, high, step=step)
        # low=16, high=128, step=16 -> 32

    if ptype == "float":
        if "low" not in spec or "high" not in spec:
            raise ValueError(f"{name}: float spec must contain 'low' and 'high'. Got keys={list(spec.keys())}")
        low = float(spec["low"])
        high = float(spec["high"])
        step = spec.get("step", None)
        log = bool(spec.get("log", False))
        if step is not None:
            return trial.suggest_float(name, low, high, step=float(step))
        return trial.suggest_float(name, low, high, log=log)
        # low=1e-5, high=1e-3, log=True -> 0.00012

    raise ValueError(f"Unknown parameter type for {name}: {ptype}")


def is_active_param(spec, updates_flat):
    active_if = spec.get("active_if")
    if not active_if:
        return True

    for cond_key, expected_value in active_if.items():
        actual_value = updates_flat.get(cond_key, None)
        if actual_value != expected_value:
            return False
    return True


def build_trial_updates(trial, param_specs):
    # Convert Optuna sampled values into nested update dict.
    updates = {}
    updates_flat = {}

    for key, spec in param_specs.items():
        if not is_active_param(spec, updates_flat):
            continue

        value = suggest_value(trial, key, spec)   # "train.lr" -> 1e-4
        value = decode_special_value(key, value)

        set_nested(updates, key, value)           # updates["train"]["lr"] = 1e-4
        updates_flat[key] = value

    return updates


def build_trial_updates_from_cfg(cfg: dict, param_specs: dict):
    # Build update dict by reading searched keys from an existing cfg.
    # Used for warm-starting with previous best cfg.
    updates = {}
    updates_flat = {}

    for key, spec in param_specs.items():
        if not is_active_param(spec, updates_flat):
            continue

        value = deep_get_by_dotted_key(cfg, key, default=None)
        if value is None:
            continue

        value = decode_special_value(key, value)
        set_nested(updates, key, value)
        updates_flat[key] = value

    return updates


def make_trial_stem(trial_number: int) -> str:
    # Convert slot index into stable trial folder name.
    return f"trial_{trial_number:04d}"
    # 3 -> "trial_0003"


def canonicalize_for_signature(obj):
    # Normalize nested structure before hashing/signature generation.
    if isinstance(obj, dict):
        return {k: canonicalize_for_signature(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [canonicalize_for_signature(x) for x in obj]
    return obj


def make_updates_signature(updates: dict) -> str:
    # Make stable signature string from trial update dict.
    normalized = canonicalize_for_signature(updates)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=True)
    # {"train": {"lr": 1e-4}} -> deterministic json string


def is_valid_trial_result_dict(result: dict) -> bool:
    # Validate result.yaml content.
    # A valid trial must have returncode == 0 and a finite rmse.
    if not isinstance(result, dict):
        return False

    rmse = result.get("rmse")
    returncode = result.get("returncode", 1)

    if rmse is None or returncode != 0:
        return False

    try:
        rmse = float(rmse)
    except Exception:
        return False

    return not math.isnan(rmse)


def load_valid_result_yaml(result_path: Path):
    # Load configs/trial_xxxx/result.yaml only if it exists and is valid.
    if not result_path.exists():
        return None

    try:
        result = load_yaml(result_path)
    except Exception:
        return None

    if not is_valid_trial_result_dict(result):
        return None

    return result


def load_existing_trial_cache(trials_jsonl: Path):
    # Build cache from phase_trials.jsonl.
    # Key: updates signature
    # Value: best successful record for that exact setting
    cache = {}

    if not trials_jsonl.exists():
        return cache

    with open(trials_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception:
                continue

            updates = record.get("updates")
            rmse = record.get("rmse")
            returncode = record.get("returncode", 1)

            cfg_file = record.get("cfg_file")
            log_file = record.get("log_file")
            config_dir = record.get("config_dir")

            if updates is None or rmse is None or returncode != 0:
                continue

            try:
                rmse = float(rmse)
            except Exception:
                continue

            if math.isnan(rmse):
                continue

            # Skip broken records whose files are missing.
            if cfg_file is not None and not Path(cfg_file).exists():
                continue
            if log_file is not None and not Path(log_file).exists():
                continue
            if config_dir is not None:
                cfg_dir = Path(config_dir)
                if not cfg_dir.exists() or not (cfg_dir / "result.yaml").exists():
                    continue

            sig = make_updates_signature(updates)

            # Keep only the best rmse for the same update signature.
            if sig not in cache or rmse < float(cache[sig]["rmse"]):
                cache[sig] = record

    return cache


def summarize_loaded_infos(progress_txt, loaded_infos):
    # Print/log previously loaded dependency cfgs.
    if not loaded_infos:
        return

    print("=" * 63)
    print("[CHAIN] Loaded previous best configs:")
    print("=" * 63)
    append_text(progress_txt, "[CHAIN] Loaded previous best configs:")

    for info in loaded_infos:
        phase_label = info["phase"]      # "pretraining"
        path_label = info["path"]        # /.../pretraining_best.yaml
        summary = info["summary"]

        print(f"  - phase={phase_label}")
        print(f"    path={path_label}")
        append_text(progress_txt, f"[CHAIN] phase={phase_label} path={path_label}")

        if summary:
            print("    settings:")
            append_text(progress_txt, f"[CHAIN] phase={phase_label} settings:")
            for k, v in summary.items():
                print(f"      {k} = {v}")
                append_text(progress_txt, f"[CHAIN]   {k} = {v}")


def find_reusable_trial_slot(exp_root: Path, cfg_trials_root: Path) -> int:
    # Find the first reusable trial slot.
    # A slot is reusable when:
    # - both experiment/config dirs do not exist, or
    # - configs/trial_xxxx/result.yaml is missing/invalid
    idx = 0
    while True:
        trial_stem = make_trial_stem(idx)          # "trial_0000"
        exp_dir = exp_root / trial_stem
        cfg_dir = cfg_trials_root / trial_stem
        result_path = cfg_dir / "result.yaml"

        if not exp_dir.exists() and not cfg_dir.exists():
            return idx

        result = load_valid_result_yaml(result_path)
        if result is None:
            return idx

        idx += 1


def make_objective(
    plan,
    project_dir,
    phase_name,
    progress_txt,
    summary_txt,
    param_specs_override=None,
    enqueue_prev_cfg_first=False,
):
    # Build one Optuna objective function for this phase.
    # The returned `objective(trial)` will:
    # - generate/update cfg
    # - reuse cached results if possible
    # - run main.py if needed
    # - write result/log/summary records

    resolved_base_cfg = build_resolved_base_cfg(plan, project_dir)
    # base.yaml + dynamics yaml + measurements yaml + methods yaml + fixed values

    prev_best_cfg, loaded_infos = load_previous_best_configs(plan, project_dir, phase_name)

    if loaded_infos:
        summarize_loaded_infos(progress_txt, loaded_infos)
        base_cfg = merge_cfg(deepcopy(resolved_base_cfg), prev_best_cfg)
        # start from resolved base cfg, then overlay previous best cfg
    else:
        print(f"[CHAIN] No previous best config found for phase={phase_name}")
        append_text(progress_txt, f"[CHAIN] No previous best config found for phase={phase_name}")
        base_cfg = deepcopy(resolved_base_cfg)
        # start from resolved base cfg

    result_dir = Path(plan["result_dir"])                        # /.../<phase>
    exp_root = Path(plan["experiments_dir"])                     # /.../<phase>/experiments
    cfg_trials_root = Path(plan["configs_dir"])                  # /.../<phase>/configs
    trials_jsonl = result_dir / f"{phase_name}_trials.jsonl"     # /.../<phase>/<phase>_trials.jsonl

    exp_root.mkdir(parents=True, exist_ok=True)
    cfg_trials_root.mkdir(parents=True, exist_ok=True)

    phase_cfg = plan["search"][phase_name]
    param_specs = phase_cfg["params"].copy() if param_specs_override is None else param_specs_override.copy()
    total_trials = int(phase_cfg.get("n_trials", 20))
    force_rerun = bool(plan.get("force_rerun", False))

    # Optional warm-start: first trial uses previous best cfg projected onto current search keys.
    prev_cfg_trial_updates = None
    if enqueue_prev_cfg_first and prev_best_cfg:
        prev_cfg_trial_updates = build_trial_updates_from_cfg(prev_best_cfg, param_specs)
        if prev_cfg_trial_updates:
            print("[WARM_START] previous cfg will be used as the first trial point")
            append_text(progress_txt, "[WARM_START] previous cfg will be used as the first trial point")

    existing_trial_cache = {} if force_rerun else load_existing_trial_cache(trials_jsonl)

    fixed_cfg = merge_cfg({}, base_cfg)
    for key, value in plan.get("fixed", {}).items():
        set_nested(fixed_cfg, key, value)        # apply fixed plan values to base cfg

    state = {
        "count": 0,                  # number of Optuna objective calls
        "consecutive_reuse": 0,      # how many duplicate reuses happened in a row
        "last_reuse_sig": None,      # last reused update signature
        "last_reuse_run": None,      # last reused run name
        "used_prev_cfg_first": False,
    }

    max_consecutive_reuse = int(plan.get("max_consecutive_reuse", 5))

    def objective(trial):
        # One Optuna trial step.

        state["count"] += 1
        local_idx = state["count"]

        use_prev_cfg_first = (
            enqueue_prev_cfg_first
            and prev_cfg_trial_updates is not None
            and not state["used_prev_cfg_first"]
        )

        if use_prev_cfg_first:
            # First trial can reuse previous best cfg values directly.
            trial_updates = dict(prev_cfg_trial_updates)
            state["used_prev_cfg_first"] = True
            append_text(progress_txt, f"[TRIAL warm-start] use previous cfg as first trial: {trial_updates}")
            print(f"[TRIAL warm-start] use previous cfg as first trial: {trial_updates}")
        else:
            trial_updates = build_trial_updates(trial, param_specs)

        updates_sig = make_updates_signature(trial_updates)   # unique signature for this update setting
        cfg = merge_cfg(fixed_cfg, trial_updates)             # final cfg for this trial

        if not validate_cfg(cfg):
            slot_idx = find_reusable_trial_slot(exp_root, cfg_trials_root)
            display_idx = slot_idx + 1
            append_text(progress_txt, f"[TRIAL {display_idx}/{total_trials}] INVALID optuna_trial={trial.number}")
            return 1e9

        cfg.setdefault("exp", {})
        cfg["exp"]["workdir_root"] = str(exp_root)            # experiments root

        slot_idx = find_reusable_trial_slot(exp_root, cfg_trials_root)
        trial_stem = make_trial_stem(slot_idx)                # "trial_0003"
        display_idx = slot_idx + 1

        exp_dir = exp_root / trial_stem                       # .../experiments/trial_0003
        cfg_dir = cfg_trials_root / trial_stem               # .../configs/trial_0003
        log_file = exp_dir / "run.log"                       # .../experiments/trial_0003/run.log
        cfg_path = cfg_dir / "config.yaml"                   # .../configs/trial_0003/config.yaml
        result_path = cfg_dir / "result.yaml"                # .../configs/trial_0003/result.yaml

        # Reuse same updates if already cached in trials_jsonl.
        if (not force_rerun) and (updates_sig in existing_trial_cache):
            cached = existing_trial_cache[updates_sig]
            cached_rmse = float(cached["rmse"])

            cached_slot_idx = int(cached.get("trial_slot_index", slot_idx))
            cached_display_idx = cached_slot_idx + 1
            cached_run_name = cached.get("run_name", make_trial_stem(cached_slot_idx))

            append_text(
                progress_txt,
                f"[TRIAL {cached_display_idx}/{total_trials}] SKIP duplicate optuna_trial={trial.number} "
                f"reuse_run={cached_run_name} rmse={cached_rmse}"
            )
            print(f"[TRIAL {cached_display_idx}/{total_trials}] duplicate setting found -> reuse {cached_run_name} rmse={cached_rmse}")

            trial.set_user_attr("run_name", cached.get("run_name"))
            trial.set_user_attr("cfg_file", cached.get("cfg_file"))
            trial.set_user_attr("log_file", cached.get("log_file"))
            trial.set_user_attr("experiment_dir", cached.get("experiment_dir"))
            trial.set_user_attr("config_dir", cached.get("config_dir"))
            trial.set_user_attr("updates", trial_updates)
            trial.set_user_attr("reused_from_cache", True)

            append_text(
                summary_txt,
                f"{cached_display_idx} {trial.number} REUSED {cached_run_name} "
                f"{cached_rmse} {cached.get('csi', 'NA')} "
                f"{cached.get('wallclock', cached.get('real', 'NA'))} "
                f"{cached.get('user', 'NA')} {cached.get('sys', 'NA')} "
                f"{cached.get('returncode', 0)} {trial_updates}"
            )

            same_as_last = (
                state["last_reuse_sig"] == updates_sig
                or state["last_reuse_run"] == cached_run_name
            )
            state["consecutive_reuse"] = state["consecutive_reuse"] + 1 if same_as_last else 1
            state["last_reuse_sig"] = updates_sig
            state["last_reuse_run"] = cached_run_name

            append_text(
                progress_txt,
                f"[REUSE_COUNT] run={cached_run_name} consecutive_reuse={state['consecutive_reuse']}/{max_consecutive_reuse}"
            )
            print(f"[REUSE_COUNT] run={cached_run_name} consecutive_reuse={state['consecutive_reuse']}/{max_consecutive_reuse}")

            if state["consecutive_reuse"] >= max_consecutive_reuse:
                append_text(progress_txt, f"[STOP] too many consecutive reuses on run={cached_run_name} count={state['consecutive_reuse']}")
                print(f"[STOP] too many consecutive reuses on run={cached_run_name} count={state['consecutive_reuse']}")
                trial.study.stop()

            return cached_rmse

        # Force rerun: remove stale dirs before rerunning.
        if force_rerun:
            shutil.rmtree(exp_dir, ignore_errors=True)
            shutil.rmtree(cfg_dir, ignore_errors=True)

        # Reuse existing configs/trial_xxxx/result.yaml if it is already valid.
        existing_result = None if force_rerun else load_valid_result_yaml(result_path)
        if existing_result is not None:
            existing_rmse = float(existing_result["rmse"])

            append_text(
                progress_txt,
                f"[TRIAL {display_idx}/{total_trials}] REUSE existing result "
                f"optuna_trial={trial.number} slot={slot_idx} run={trial_stem} rmse={existing_rmse}"
            )
            print(f"[TRIAL {display_idx}/{total_trials}] reuse existing result -> optuna_trial={trial.number} slot={slot_idx} run={trial_stem} rmse={existing_rmse}")

            trial.set_user_attr("run_name", existing_result.get("run_name", trial_stem))
            trial.set_user_attr("cfg_file", existing_result.get("cfg_file", str(cfg_path)))
            trial.set_user_attr("log_file", existing_result.get("log_file", str(log_file)))
            trial.set_user_attr("experiment_dir", existing_result.get("experiment_dir", str(exp_dir)))
            trial.set_user_attr("config_dir", existing_result.get("config_dir", str(cfg_dir)))
            trial.set_user_attr("updates", trial_updates)
            trial.set_user_attr("reused_from_existing_result", True)

            if updates_sig not in existing_trial_cache:
                existing_trial_cache[updates_sig] = existing_result
                write_jsonl(trials_jsonl, existing_result)

            append_text(
                summary_txt,
                f"{display_idx} {trial.number} EXISTING {existing_result.get('run_name', trial_stem)} "
                f"{existing_result.get('rmse', 'NA')} {existing_result.get('csi', 'NA')} "
                f"{existing_result.get('wallclock', existing_result.get('real', 'NA'))} "
                f"{existing_result.get('user', 'NA')} {existing_result.get('sys', 'NA')} "
                f"{existing_result.get('returncode', 0)} {trial_updates}"
            )
            return existing_rmse

        # Clean broken/stale dirs before writing new outputs.
        if exp_dir.exists() or cfg_dir.exists():
            append_text(progress_txt, f"[TRIAL {display_idx}/{total_trials}] CLEAN stale dirs optuna_trial={trial.number} slot={slot_idx} run={trial_stem}")
            print(f"[TRIAL {display_idx}/{total_trials}] clean stale dirs -> slot={slot_idx} run={trial_stem}")
            shutil.rmtree(exp_dir, ignore_errors=True)
            shutil.rmtree(cfg_dir, ignore_errors=True)

        state["consecutive_reuse"] = 0
        state["last_reuse_sig"] = None
        state["last_reuse_run"] = None

        exp_dir.mkdir(parents=True, exist_ok=True)
        cfg_dir.mkdir(parents=True, exist_ok=True)

        save_yaml(cfg, cfg_path)                              # save configs/trial_xxxx/config.yaml

        append_text(progress_txt, f"[TRIAL {display_idx}/{total_trials}] START optuna_trial={trial.number} slot={slot_idx} run={trial_stem}")
        print(f"[TRIAL {display_idx}/{total_trials}] optuna_trial={trial.number} slot={slot_idx} run={trial_stem}")

        cmd = [
            "python",
            str(Path(project_dir) / "main.py"),
            "--config", str(cfg_path),
            "--exp", trial_stem,
        ]

        # Run main.py and redirect both stdout/stderr to experiments/trial_xxxx/run.log
        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.run(
                ["/usr/bin/time", "-p"] + cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
                cwd=project_dir,
            )

        rmse, csi, real, user, sys_t = parse_log_metrics(log_file)

        is_valid = (proc.returncode == 0)
        try:
            is_valid = is_valid and (not math.isnan(rmse))
        except Exception:
            is_valid = False

        # Save one trial summary yaml under configs/trial_xxxx/result.yaml
        trial_result = {
            "trial_number": trial.number,
            "trial_slot_index": slot_idx,
            "local_trial_index": local_idx,
            "display_trial_index": display_idx,
            "run_name": trial_stem,
            "rmse": rmse,
            "csi": csi,
            "wallclock": real,
            "real": real,
            "user": user,
            "sys": sys_t,
            "returncode": proc.returncode,
            "is_valid": is_valid,
            "updates": trial_updates,
            "log_file": str(log_file),
            "cfg_file": str(cfg_path),
            "config_dir": str(cfg_dir),
            "experiment_dir": str(exp_dir),
        }

        save_yaml(trial_result, result_path)                  # save configs/trial_xxxx/result.yaml

        if not force_rerun:
            existing_trial_cache[updates_sig] = trial_result

        write_jsonl(trials_jsonl, trial_result)               # append one record to phase_trials.jsonl

        append_text(
            summary_txt,
            f"{display_idx} {trial.number} {trial_stem} {rmse} {csi} {real} {user} {sys_t} {proc.returncode} {trial_updates}"
        )

        trial.set_user_attr("run_name", trial_stem)
        trial.set_user_attr("cfg_file", str(cfg_path))
        trial.set_user_attr("log_file", str(log_file))
        trial.set_user_attr("experiment_dir", str(exp_dir))
        trial.set_user_attr("config_dir", str(cfg_dir))
        trial.set_user_attr("updates", trial_updates)

        if (proc.returncode != 0) or (not is_valid):
            append_text(progress_txt, f"[TRIAL {display_idx}/{total_trials}] FAIL optuna_trial={trial.number} slot={slot_idx} run={trial_stem}")
            return 1e9

        append_text(
            progress_txt,
            f"[TRIAL {display_idx}/{total_trials}] DONE optuna_trial={trial.number} "
            f"slot={slot_idx} run={trial_stem} rmse={rmse} csi={csi} wallclock={real}"
        )
        return rmse

    return objective