from swap.src.tuning.phases import (
    run_normalization_selection,
    run_model_selection,
    run_pretraining,
    run_sample_sensitivity,
    run_finetuning,
    run_baseline_tuning,
)


PHASE_RUNNERS = {
    "normalization_selection": run_normalization_selection.main,
    "model_selection": run_model_selection.main,
    "pretraining": run_pretraining.main,
    "sample_sensitivity": run_sample_sensitivity.main,
    "finetuning": run_finetuning.main,
    "baseline_tuning": run_baseline_tuning.main,
}


def run_phase_main(phase: str, plan: dict, project_dir):
    try:
        runner = PHASE_RUNNERS[phase]
    except KeyError:
        raise ValueError(f"Unknown phase: {phase}")
    runner(plan, project_dir)