from swap.src.tuning.run_phase import run_phase

def main(plan, project_dir="."):
    run_phase(
        plan=plan,
        project_dir=project_dir,
        phase_name="normalization_selection"
    )