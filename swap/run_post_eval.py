import argparse
import shutil
from copy import deepcopy
from pathlib import Path

import swap.src.post.num_sample as num_sample
import swap.src.post.temporal_sensitivity as temporal_sensitivity
import swap.src.post.measurement_sensitivity as measurement_sensitivity

from swap.src.utils.yaml_utils import load_yaml
from swap.src.utils.plan_utils import resolve_swap_dir



PHASES = {
    "num_sample": num_sample,
    "temporal_sensitivity": temporal_sensitivity,
    "measurement_sensitivity": measurement_sensitivity,
}

GENERAL_BEST_CANDIDATES = [
    ("finetuning", "best_finetune.yaml"),
    ("sample_sensitivity", "best_sample.yaml"),
    ("pretraining", "best_pretrain.yaml"),
    ("model_selection", "best_model.yaml"),
    ("normalization_selection", "best_normalization.yaml"),
]


def resolve_post_config_path(
    phase: str,
    method_name: str | None = None,
    use_pretrain: bool = False,
    use_scratch: bool = False,
) -> Path:
    swap_dir = resolve_swap_dir()
    post_root = swap_dir / "configs" / "post"

    if use_pretrain and use_scratch:
        raise ValueError("Use only one of --pretrain or --scratch")

    config_stem = phase
    if phase == "num_sample":
        if use_pretrain:
            config_stem = "num_sample_pretrain"
        elif use_scratch:
            config_stem = "num_sample_scratch"

    if method_name:
        method_name = str(method_name).strip().lower()
        method_specific = post_root / method_name / f"{config_stem}.yaml"
        if method_specific.exists():
            return method_specific

    shared_config = post_root / f"{config_stem}.yaml"
    if shared_config.exists():
        return shared_config

    tried = []
    if method_name:
        tried.append(str(post_root / method_name / f"{config_stem}.yaml"))
    tried.append(str(shared_config))

    raise FileNotFoundError(
        "post config not found. tried: " + " and ".join(tried)
    )


def parse_summary_txt(summary_path: Path) -> dict:
    data = {}
    if not summary_path.exists():
        return data

    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    return data


def load_base_cfg(base_yaml: Path) -> dict:
    return load_yaml(base_yaml)


def infer_method_name_from_root_dir(root_dir: Path) -> str:
    root_dir = root_dir.resolve()
    parts = root_dir.parts

    if "masf" not in parts:
        raise ValueError(f"'masf' not found in root_dir: {root_dir}")

    masf_idx = parts.index("masf")

    if masf_idx + 2 >= len(parts):
        raise ValueError(f"Could not infer method_name from root_dir: {root_dir}")

    method_name = parts[masf_idx + 2]
    if not method_name:
        raise ValueError(f"Empty method_name inferred from root_dir: {root_dir}")

    return method_name


def infer_report_root_from_measurement_root(root_dir: Path) -> Path:
    root_dir = root_dir.resolve()
    if len(root_dir.parts) < 5:
        raise ValueError(f"Unexpected measurement root depth: {root_dir}")
    return root_dir.parents[3]


def is_baseline_root(root_dir: Path) -> bool:
    baseline_root = root_dir / "baseline_tuning"
    if not baseline_root.exists():
        return False

    for phase_dir in baseline_root.glob("num_samples_*"):
        if not phase_dir.is_dir():
            continue
        for filename in ["best_enkf.yaml", "best_letkf.yaml"]:
            if (phase_dir / filename).exists():
                return True

    return False


def find_general_best_yaml(root_dir: Path) -> Path | None:
    for phase_name, filename in GENERAL_BEST_CANDIDATES:
        cand = root_dir / phase_name / filename
        if cand.exists():
            return cand
    return None


def find_baseline_best_yaml_for_num(root_dir: Path, num_samples: int) -> Path | None:
    phase_dir = root_dir / "baseline_tuning" / f"num_samples_{num_samples}"
    if not phase_dir.exists():
        return None

    for filename in ["best_enkf.yaml", "best_letkf.yaml"]:
        cand = phase_dir / filename
        if cand.exists():
            return cand

    return None


def find_any_baseline_best_yaml(root_dir: Path) -> Path | None:
    baseline_root = root_dir / "baseline_tuning"
    if not baseline_root.exists():
        return None

    for phase_dir in sorted(baseline_root.glob("num_samples_*")):
        if not phase_dir.is_dir():
            continue
        for filename in ["best_enkf.yaml", "best_letkf.yaml"]:
            cand = phase_dir / filename
            if cand.exists():
                return cand

    return None


def resolve_default_base_yaml(root_dir: Path) -> Path:
    best_yaml = find_general_best_yaml(root_dir)
    if best_yaml is not None:
        return best_yaml

    best_yaml = find_any_baseline_best_yaml(root_dir)
    if best_yaml is not None:
        return best_yaml

    raise FileNotFoundError(f"Could not find tuned best yaml under root_dir: {root_dir}")


def print_base_yaml_info(base_yaml: Path):
    cfg = load_yaml(base_yaml)

    dynamics = cfg.get("dynamics", {})
    steps = cfg.get("steps", {})
    measurement = cfg.get("measurement", {})
    method = cfg.get("method", {})

    print("=" * 80)
    print("[POST_EVAL] Loaded base yaml info")
    print("=" * 80)
    print(f"[POST_EVAL] base_yaml                 : {base_yaml}")
    print(f"[POST_EVAL] method.name              : {method.get('name')}")
    print(f"[POST_EVAL] dynamics.num_samples     : {dynamics.get('num_samples')}")
    print(f"[POST_EVAL] dynamics.dim             : {dynamics.get('dim')}")
    print(f"[POST_EVAL] steps.initial            : {steps.get('initial')}")
    print(f"[POST_EVAL] steps.end                : {steps.get('end')}")
    print(f"[POST_EVAL] steps.gap                : {steps.get('gap')}")
    print(f"[POST_EVAL] measurement.type         : {measurement.get('type')}")
    print(f"[POST_EVAL] measurement.T            : {measurement.get('T')}")
    print(f"[POST_EVAL] measurement.noise_std    : {measurement.get('noise_std')}")

    mtype = measurement.get("type")
    if mtype == "grid_mask":
        print(f"[POST_EVAL] measurement.stride        : {measurement.get('stride')}")
    elif mtype == "center_mask":
        print(f"[POST_EVAL] measurement.hole_ratio    : {measurement.get('hole_ratio')}")
    elif mtype == "low_resolution":
        print(f"[POST_EVAL] measurement.scale_factor  : {measurement.get('scale_factor')}")
    elif mtype == "nonlinear":
        print(f"[POST_EVAL] measurement.nonlinear_type: {measurement.get('nonlinear_type')}")
        print(f"[POST_EVAL] measurement.alpha         : {measurement.get('alpha')}")

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

    plan = deepcopy(plan)
    plan["grid"] = measurement_grid[mtype]
    return plan


def get_num_sample_grid_values(plan: dict) -> list[int]:
    grid_block = plan.get("grid", {})
    values = grid_block.get("dynamics.num_samples", [])
    return [int(v) for v in values]


def is_valid_results_yaml(results_yaml: Path) -> bool:
    if not results_yaml.exists() or not results_yaml.is_file():
        return False

    try:
        data = load_yaml(results_yaml)
    except Exception as e:
        print(f"[POST_CACHE] invalid yaml parse failed: {results_yaml} ({e})")
        return False

    if not isinstance(data, dict):
        print(f"[POST_CACHE] invalid yaml content not dict: {results_yaml}")
        return False

    required_any_keys = [
        "status",
        "best_seed",
        "best_metric",
        "metrics",
        "result",
        "results",
    ]
    if not any(k in data for k in required_any_keys):
        print(f"[POST_CACHE] invalid yaml missing expected keys: {results_yaml}")
        return False

    status = data.get("status")
    if status is not None and str(status).strip().lower() in {"failed", "error", "running"}:
        print(f"[POST_CACHE] invalid yaml status={status}: {results_yaml}")
        return False

    return True


def find_valid_results_yaml_under(path: Path) -> Path | None:
    if not path.exists():
        return None

    for cand in sorted(path.rglob("results.yaml")):
        if is_valid_results_yaml(cand):
            return cand

    return None


def resolve_post_phase_root(root_dir: Path, phase: str) -> Path:
    return root_dir / "post" / phase


def should_skip_post_phase(root_dir: Path, phase: str, force_rerun: bool) -> bool:
    if force_rerun:
        return False

    phase_root = resolve_post_phase_root(root_dir, phase)
    cached_yaml = find_valid_results_yaml_under(phase_root)

    if cached_yaml is not None:
        print("=" * 80)
        print(f"[POST_CACHE] skip phase={phase}")
        print(f"[POST_CACHE] valid results.yaml found: {cached_yaml}")
        print("=" * 80)
        return True

    return False


def should_skip_num_sample_case(root_dir: Path, num: int, force_rerun: bool) -> bool:
    if force_rerun:
        return False

    case_root = root_dir / "post" / "num_sample" / f"num_samples_{num}"
    cached_yaml = find_valid_results_yaml_under(case_root)

    if cached_yaml is not None:
        print("=" * 80)
        print(f"[POST_CACHE] skip num_sample={num}")
        print(f"[POST_CACHE] valid results.yaml found: {cached_yaml}")
        print("=" * 80)
        return True

    return False


def cleanup_post_phase_keep_best_seed(root_dir: Path, phase: str):
    phase_root = root_dir / "post" / phase

    if not phase_root.exists():
        print(f"[POST_CLEANUP] skip: missing phase root -> {phase_root}")
        return

    print("=" * 80)
    print(f"[POST_CLEANUP] phase      : {phase}")
    print(f"[POST_CLEANUP] phase_root : {phase_root}")
    print("=" * 80)

    total_cases = 0
    cleaned_cases = 0
    removed_dirs = 0
    skipped_cases = 0

    for case_dir in sorted(phase_root.iterdir()):
        if not case_dir.is_dir():
            continue

        total_cases += 1
        summary_path = case_dir / "summary.txt"
        experiments_dir = case_dir / "experiments"

        if not summary_path.exists():
            print(f"[POST_CLEANUP] skip case={case_dir.name}: missing summary.txt")
            skipped_cases += 1
            continue

        if not experiments_dir.exists():
            print(f"[POST_CLEANUP] skip case={case_dir.name}: missing experiments/")
            skipped_cases += 1
            continue

        summary = parse_summary_txt(summary_path)
        best_seed_raw = summary.get("best_seed", "").strip()

        if not best_seed_raw:
            print(f"[POST_CLEANUP] skip case={case_dir.name}: no best_seed in summary.txt")
            skipped_cases += 1
            continue

        try:
            best_seed = int(best_seed_raw)
        except Exception:
            print(f"[POST_CLEANUP] skip case={case_dir.name}: invalid best_seed={best_seed_raw}")
            skipped_cases += 1
            continue

        keep_dir_name = f"seed_{best_seed:04d}"
        case_removed = 0

        for child in sorted(experiments_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name == keep_dir_name:
                continue
            shutil.rmtree(child, ignore_errors=True)
            case_removed += 1
            removed_dirs += 1

        print(f"[POST_CLEANUP] case={case_dir.name} keep={keep_dir_name} removed={case_removed}")
        cleaned_cases += 1

    print("=" * 80)
    print("[POST_CLEANUP] done")
    print(f"[POST_CLEANUP] total_cases   : {total_cases}")
    print(f"[POST_CLEANUP] cleaned_cases : {cleaned_cases}")
    print(f"[POST_CLEANUP] skipped_cases : {skipped_cases}")
    print(f"[POST_CLEANUP] removed_dirs  : {removed_dirs}")
    print("=" * 80)


def run_single_phase(
    plan: dict,
    phase: str,
    root_dir: Path,
    base_yaml: Path,
    force_rerun: bool,
    cleanup: bool,
):
    local_plan = deepcopy(plan)
    local_plan["phase"] = phase
    local_plan["base_yaml"] = str(base_yaml)

    report_root = infer_report_root_from_measurement_root(root_dir)
    docs_dir = report_root / "docs"


    print("=" * 80)
    print(f"[RUN_POST_EVAL] phase         : {phase}")
    print(f"[RUN_POST_EVAL] root_dir      : {root_dir}")
    print(f"[RUN_POST_EVAL] report_root   : {report_root}")
    print(f"[RUN_POST_EVAL] base_yaml     : {base_yaml}")
    print(f"[RUN_POST_EVAL] docs_dir      : {docs_dir}")
    print(f"[RUN_POST_EVAL] force_rerun   : {force_rerun}")
    print(f"[RUN_POST_EVAL] cleanup       : {cleanup}")
    print("=" * 80)

    PHASES[phase].main(local_plan, force_rerun=force_rerun)

    if cleanup:
        cleanup_post_phase_keep_best_seed(root_dir, phase)

    docs_dir.mkdir(parents=True, exist_ok=True)


    print(f"[RUN_POST_EVAL] done: phase={phase} report={docs_dir / 'index.html'}")


def run_num_sample_for_baseline(plan: dict, root_dir: Path, force_rerun: bool, cleanup: bool):
    nums = get_num_sample_grid_values(plan)
    if not nums:
        raise ValueError("num_sample baseline mode requires plan['grid']['dynamics.num_samples']")

    print("=" * 80)
    print(f"[RUN_POST_EVAL] baseline num_sample grid: {nums}")
    print("=" * 80)

    for num in nums:
        if should_skip_num_sample_case(root_dir, num, force_rerun):
            continue

        base_yaml = find_baseline_best_yaml_for_num(root_dir, num)
        if base_yaml is None:
            print(f"[RUN_POST_EVAL] skip num_sample={num}: no best baseline yaml found")
            continue

        local_plan = deepcopy(plan)
        local_plan["phase"] = "num_sample"
        local_plan["base_yaml"] = str(base_yaml)
        local_plan.setdefault("grid", {})
        local_plan["grid"]["dynamics.num_samples"] = [num]

        print_base_yaml_info(base_yaml)

        run_single_phase(
            plan=local_plan,
            phase="num_sample",
            root_dir=root_dir,
            base_yaml=base_yaml,
            force_rerun=force_rerun,
            cleanup=cleanup,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=list(PHASES.keys()))
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--root_dir", required=True, help="measurement root")
    parser.add_argument("--base_yaml", default=None, help="optional explicit best yaml override")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument(
        "--cleanup_keep_best_seed",
        "--cleanup",
        action="store_true",
        help="Keep only the best seed dir in each case and delete the others.",
    )
    args = parser.parse_args()

    if args.pretrain and args.scratch:
        raise ValueError("Use only one of --pretrain or --scratch")

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    method_name = infer_method_name_from_root_dir(root_dir)

    config_path = resolve_post_config_path(
        phase=args.phase,
        method_name=method_name,
        use_pretrain=args.pretrain,
        use_scratch=args.scratch,
    )
    plan = load_yaml(config_path)

    if args.seeds is not None:
        plan["seeds"] = args.seeds
    if args.num_samples is not None:
        plan.setdefault("fixed", {})
        plan["fixed"]["dynamics.num_samples"] = args.num_samples

    baseline_mode = is_baseline_root(root_dir)

    if args.base_yaml is not None:
        selected_base_yaml = Path(args.base_yaml).resolve()
        if not selected_base_yaml.exists():
            raise FileNotFoundError(f"base_yaml not found: {selected_base_yaml}")
    else:
        if baseline_mode and args.num_samples is not None:
            selected_base_yaml = find_baseline_best_yaml_for_num(root_dir, args.num_samples)
            if selected_base_yaml is None:
                raise FileNotFoundError(
                    f"Could not find baseline best yaml for num_samples={args.num_samples} under {root_dir}"
                )
        else:
            selected_base_yaml = resolve_default_base_yaml(root_dir)

    plan["phase"] = args.phase
    plan["base_yaml"] = str(selected_base_yaml)
    plan = inject_measurement_grid(plan, selected_base_yaml)

    print(f"[DEBUG] inferred method_name = {method_name}")
    print(f"[DEBUG] loaded config_path = {config_path}")
    print(f"[DEBUG] raw loaded plan = {plan}")
    print(f"[DEBUG] raw grid from plan = {plan.get('grid')}")

    report_root = infer_report_root_from_measurement_root(root_dir)

    print("=" * 80)
    print(f"[RUN_POST_EVAL] method_name    : {method_name}")
    print(f"[RUN_POST_EVAL] config         : {config_path}")
    print(f"[RUN_POST_EVAL] phase          : {args.phase}")
    print(f"[RUN_POST_EVAL] pretrain       : {args.pretrain}")
    print(f"[RUN_POST_EVAL] scratch        : {args.scratch}")
    print(f"[RUN_POST_EVAL] root_dir       : {root_dir}")
    print(f"[RUN_POST_EVAL] report_root    : {report_root}")
    print(f"[RUN_POST_EVAL] selected_yaml  : {selected_base_yaml}")
    print(f"[RUN_POST_EVAL] baseline_root  : {baseline_mode}")
    print(f"[RUN_POST_EVAL] force_rerun    : {args.force_rerun}")
    print(f"[RUN_POST_EVAL] cleanup        : {args.cleanup_keep_best_seed}")
    print(f"[RUN_POST_EVAL] seeds          : {plan.get('seeds')}")
    print(f"[RUN_POST_EVAL] num_samples    : {plan.get('fixed', {}).get('dynamics.num_samples')}")
    if args.phase == "measurement_sensitivity":
        print(f"[RUN_POST_EVAL] resolved grid : {plan['grid']}")
    print("=" * 80)

    if args.phase == "num_sample" and baseline_mode and args.base_yaml is None and not args.pretrain and not args.scratch:
        run_num_sample_for_baseline(
            plan=plan,
            root_dir=root_dir,
            force_rerun=args.force_rerun,
            cleanup=args.cleanup_keep_best_seed,
        )
        return

    print_base_yaml_info(selected_base_yaml)
    run_single_phase(
        plan=plan,
        phase=args.phase,
        root_dir=root_dir,
        base_yaml=selected_base_yaml,
        force_rerun=args.force_rerun,
        cleanup=args.cleanup_keep_best_seed,
    )


if __name__ == "__main__":
    main()