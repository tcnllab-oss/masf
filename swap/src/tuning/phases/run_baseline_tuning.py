from swap.src.tuning.run_phase import run_phase
from swap.src.utils.grid_utils import expand_plan_grid, describe_grid_values


def main(plan, project_dir="."):
    expanded_plans = expand_plan_grid(plan)

    if len(expanded_plans) == 1:
        run_phase(
            plan=expanded_plans[0],
            project_dir=project_dir,
            phase_name="baseline_tuning",
        )
        return

    print("=" * 80)
    print(f"[BASELINE_GRID] total grid runs: {len(expanded_plans)}")
    print("=" * 80)

    for idx, plan_i in enumerate(expanded_plans, start=1):
        print("-" * 80)
        print(f"[BASELINE_GRID] run {idx}/{len(expanded_plans)}")
        print(f"[BASELINE_GRID] grid values: {describe_grid_values(plan_i)}")
        print("-" * 80)

        run_phase(
            plan=plan_i,
            project_dir=project_dir,
            phase_name="baseline_tuning",
        )