import argparse
import shutil

from swap.src.utils.yaml_utils import load_yaml
from swap.src.utils.plan_utils import (
    apply_cli_to_plan,
    get_default_plan_path,
    resolve_swap_dir,
    resolve_configs_dir,
    get_measurement_root,
    get_phase_root,
    get_report_root,
    force_all_n_trials_to_one,
)
from swap.src.utils.phase_utils import ALL_PHASES, PHASE_ORDER
from swap.src.tuning.run_dispatch import run_phase_main
from swap.src.utils.cleanup import cleanup_trial_dirs_keep_best
from swap.make_phase_ranking_tables import run_phase_ranking_tables



def apply_measurement_cli_overrides(plan: dict, args) -> dict:
    plan.setdefault("fixed", {})

    if args.num_samples is not None:
        plan["fixed"]["dynamics.num_samples"] = args.num_samples

    if args.stride is not None:
        plan["fixed"]["measurement.stride"] = args.stride

    if args.hole_ratio is not None:
        plan["fixed"]["measurement.hole_ratio"] = args.hole_ratio

    if args.scale_factor is not None:
        plan["fixed"]["measurement.scale_factor"] = args.scale_factor

    if args.alpha is not None:
        plan["fixed"]["measurement.alpha"] = args.alpha

    return plan


def print_measurement_override_info(plan: dict):
    fixed = plan.get("fixed", {})
    if not isinstance(fixed, dict):
        return

    if "dynamics.num_samples" in fixed:
        print(f"[PIPELINE] num_samples : {fixed['dynamics.num_samples']}")
    if "measurement.stride" in fixed:
        print(f"[PIPELINE] stride      : {fixed['measurement.stride']}")
    if "measurement.hole_ratio" in fixed:
        print(f"[PIPELINE] hole_ratio  : {fixed['measurement.hole_ratio']}")
    if "measurement.scale_factor" in fixed:
        print(f"[PIPELINE] scale_factor: {fixed['measurement.scale_factor']}")
    if "measurement.alpha" in fixed:
        print(f"[PIPELINE] alpha       : {fixed['measurement.alpha']}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--phase", choices=ALL_PHASES, default=None)
    parser.add_argument("--from_phase", "--from", dest="from_phase", choices=PHASE_ORDER, default=None)
    parser.add_argument("--until", choices=PHASE_ORDER, default=None)

    parser.add_argument("--out_dir", "-o", default=None)

    parser.add_argument("--dynamic_type", "-d", default="kolmogorov")
    parser.add_argument("--measurement_type", default="grid_mask")
    parser.add_argument("--method_type", default="ours")
    parser.add_argument("--nonlinear_type", default=None)
    parser.add_argument("--dim", type=int, default=128)

    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--hole_ratio", type=float, default=None)
    parser.add_argument("--scale_factor", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)

    parser.add_argument("--test", action="store_true")

    parser.add_argument("--seed_eval", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", required=False)
    parser.add_argument("--top_k", type=int, default=3)

    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--cleanup_method_dir_keep_best", "--cleanup", action="store_true")

    args = parser.parse_args()

    if args.phase is None and args.until is None:
        raise ValueError("Provide either --phase or --until")
    if args.phase is not None and args.until is not None:
        raise ValueError("Use either --phase or --until, not both")
    if args.phase == "baseline_tuning" and args.method_type not in ["enkf", "letkf"]:
        raise ValueError("baseline_tuning requires --method_type enkf or letkf")
    if args.phase is not None and args.from_phase is not None:
        raise ValueError("--from_phase cannot be used with --phase")

    if args.measurement_type != "grid_mask" and args.stride is not None:
        raise ValueError("--stride is only valid with --measurement_type grid_mask")
    if args.measurement_type != "center_mask" and args.hole_ratio is not None:
        raise ValueError("--hole_ratio is only valid with --measurement_type center_mask")
    if args.measurement_type != "low_resolution" and args.scale_factor is not None:
        raise ValueError("--scale_factor is only valid with --measurement_type low_resolution")
    if args.measurement_type != "nonlinear" and args.alpha is not None:
        raise ValueError("--alpha is only valid with --measurement_type nonlinear")
    if args.measurement_type == "nonlinear" and args.nonlinear_type is None:
        raise ValueError("--measurement_type nonlinear requires --nonlinear_type")

    swap_dir = resolve_swap_dir()
    project_dir = swap_dir.parent
    configs_dir = resolve_configs_dir(swap_dir)

    if args.phase is not None:
        phases_to_run = [args.phase]
    else:
        start_idx = 0 if args.from_phase is None else PHASE_ORDER.index(args.from_phase)
        end_idx = PHASE_ORDER.index(args.until)

        if start_idx > end_idx:
            raise ValueError(f"--from_phase ({args.from_phase}) must be earlier than or equal to --until ({args.until})")

        phases_to_run = PHASE_ORDER[start_idx:end_idx + 1]

    for phase in phases_to_run:
        plan_path = get_default_plan_path(
            configs_dir=configs_dir,
            phase=phase,
            dynamic_type=args.dynamic_type,
            dim=args.dim,
            method_type=args.method_type,
        )

        if not plan_path.exists():
            raise FileNotFoundError(f"Missing default plan for {phase}: {plan_path}")

        plan = load_yaml(plan_path)
        plan = apply_cli_to_plan(plan, args)
        plan = apply_measurement_cli_overrides(plan, args)

        if args.rerun:
            plan["force_rerun"] = True

        if args.test:
            force_all_n_trials_to_one(plan)

        phase_root = get_phase_root(plan, project_dir, phase)
        measurement_root = get_measurement_root(plan, project_dir)
        report_root = get_report_root(plan, project_dir)
        docs_dir = report_root / "docs"

        if args.rerun and phase_root.exists():
            print(f"[PIPELINE] rerun delete: {phase_root}")
            shutil.rmtree(phase_root)

        print("=" * 80)
        print(f"[PIPELINE] phase        : {phase}")
        print(f"[PIPELINE] phase_root   : {phase_root}")
        print(f"[PIPELINE] dynamic      : {plan['dynamic_name']}")
        print(f"[PIPELINE] method       : {plan['method_type']}")
        print(f"[PIPELINE] measure      : {plan['measurement_type']}")
        if plan.get("nonlinear_type") is not None:
            print(f"[PIPELINE] nonlinear   : {plan['nonlinear_type']}")
        print_measurement_override_info(plan)
        print(f"[PIPELINE] out_dir      : {plan['output_dir']}")
        print(f"[PIPELINE] cleanup      : {args.cleanup_method_dir_keep_best}")
        print("=" * 80)

        run_phase_main(phase, plan, project_dir)

        print("dir is cleaned...")
        if args.cleanup_method_dir_keep_best:
            cleanup_trial_dirs_keep_best(
                phase_root=phase_root,
                phase=phase,
                method_dir_name=plan["method_type"],
            )

        if phase == "baseline_tuning":
            run_phase_ranking_tables(root_dir=str(measurement_root), phase=phase)
        else:
            run_phase_ranking_tables(root_dir=str(measurement_root), until=phase)

        print(f"[PIPELINE] done: phase={phase} report={docs_dir / 'index.html'}")


if __name__ == "__main__":
    main()