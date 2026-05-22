#!/usr/bin/env python
import json
import math
import shutil
from pathlib import Path


def load_best_run_name_from_trials_jsonl(phase_root: Path, phase: str):
    trials_jsonl = phase_root / f"{phase}_trials.jsonl"
    if not trials_jsonl.exists():
        return None

    best_run_name = None
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
            run_name = rec.get("run_name")

            if rmse is None or run_name is None:
                continue

            try:
                rmse = float(rmse)
            except Exception:
                continue

            if math.isnan(rmse):
                continue

            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_run_name = run_name

    return best_run_name


import shutil
from pathlib import Path


def _cleanup_one_phase_leaf(phase_root: Path, phase: str):
    phase_root = Path(phase_root)
    experiments_dir = phase_root / "experiments"
    trials_jsonl = phase_root / f"{phase}_trials.jsonl"

    if not phase_root.exists():
        print(f"[CLEANUP] skip: phase_root not found -> {phase_root}")
        return

    if not experiments_dir.exists():
        print(f"[CLEANUP] skip: experiments dir not found -> {experiments_dir}")
        return

    if not trials_jsonl.exists():
        print(f"[CLEANUP] skip: trials jsonl not found -> {trials_jsonl}")
        return

    best_run_name = load_best_run_name_from_trials_jsonl(phase_root, phase)
    if best_run_name is None:
        print(f"[CLEANUP] skip: could not determine best run from {trials_jsonl}")
        return

    print("=" * 80)
    print(f"[CLEANUP] phase           : {phase}")
    print(f"[CLEANUP] phase_root      : {phase_root}")
    print(f"[CLEANUP] experiments_dir : {experiments_dir}")
    print(f"[CLEANUP] trials_jsonl    : {trials_jsonl}")
    print(f"[CLEANUP] keep_best_run   : {best_run_name}")
    print("=" * 80)

    kept = 0
    deleted = 0
    skipped = 0

    for trial_dir in sorted(experiments_dir.iterdir()):
        if not trial_dir.is_dir():
            continue

        if trial_dir.name == best_run_name:
            kept += 1
            print(f"[CLEANUP] keep best dir   : {trial_dir}")
            continue

        try:
            shutil.rmtree(trial_dir)
            deleted += 1
            print(f"[CLEANUP] deleted         : {trial_dir}")
        except Exception as e:
            skipped += 1
            print(f"[CLEANUP] failed to delete: {trial_dir} ({e})")

    print("-" * 80)
    print(f"[CLEANUP] kept={kept} deleted={deleted} skipped={skipped}")
    print("=" * 80)


def cleanup_trial_dirs_keep_best(
    phase_root: Path,
    phase: str,
    method_dir_name: str = None,
):
    phase_root = Path(phase_root)

    if phase != "baseline_tuning":
        _cleanup_one_phase_leaf(phase_root, phase)
        return

    if not phase_root.exists():
        print(f"[CLEANUP] skip: phase_root not found -> {phase_root}")
        return

    # case 1: leaf root already passed
    # .../baseline_tuning/num_samples_10
    if phase_root.name.startswith("num_samples_"):
        _cleanup_one_phase_leaf(phase_root, phase)
        return

    # case 2: baseline parent root passed
    # .../baseline_tuning
    if phase_root.name == "baseline_tuning":
        child_roots = [
            p for p in sorted(phase_root.iterdir())
            if p.is_dir() and p.name.startswith("num_samples_")
        ]

        if not child_roots:
            print(f"[CLEANUP] skip: no num_samples_* dirs under -> {phase_root}")
            return

        print("=" * 80)
        print(f"[CLEANUP] baseline parent root : {phase_root}")
        print(f"[CLEANUP] child roots          : {[p.name for p in child_roots]}")
        print("=" * 80)

        for child_root in child_roots:
            _cleanup_one_phase_leaf(child_root, phase)
        return

    # case 3: measurement root passed by mistake
    # .../stride_10
    baseline_root = phase_root / "baseline_tuning"
    if baseline_root.exists() and baseline_root.is_dir():
        child_roots = [
            p for p in sorted(baseline_root.iterdir())
            if p.is_dir() and p.name.startswith("num_samples_")
        ]

        if not child_roots:
            print(f"[CLEANUP] skip: no num_samples_* dirs under -> {baseline_root}")
            return

        print("=" * 80)
        print(f"[CLEANUP] auto-resolved baseline root : {baseline_root}")
        print(f"[CLEANUP] child roots                : {[p.name for p in child_roots]}")
        print("=" * 80)

        for child_root in child_roots:
            _cleanup_one_phase_leaf(child_root, phase)
        return

    print(f"[CLEANUP] skip: could not resolve baseline_tuning root from -> {phase_root}")

def cleanup_seed_dir_keep_config_and_result(seed_dir: Path):
    removed = []

    outputs_dir = seed_dir / "outputs"
    log_path = seed_dir / "run.log"

    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)
        removed.append(str(outputs_dir))

    if log_path.exists():
        log_path.unlink()
        removed.append(str(log_path))

    return removed


def cleanup_candidate_seed_outputs(candidate_root: Path):
    total_removed = 0
    seed_count = 0
    skipped = 0

    for seed_dir in sorted(candidate_root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue

        seed_count += 1
        removed = cleanup_seed_dir_keep_config_and_result(seed_dir)
        total_removed += len(removed)

        if removed:
            print(f"[SEED_CLEANUP] {seed_dir.name} removed:")
            for p in removed:
                print(f"  - {p}")
        else:
            skipped += 1
            print(f"[SEED_CLEANUP] {seed_dir.name} nothing to remove")

    return {
        "seed_count": seed_count,
        "removed_items": total_removed,
        "skipped": skipped,
    }