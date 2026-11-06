import os
import click
import torch
from utils.train_load import load_trained_model
from utils.run_utils import load_yaml, print_run_info, set_seed, measurement_update, time_update
from utils.build import build_workdir, build_steps, build_method, build_dynamics, build_measurement, build_dataset
from utils.summary import finalize_and_save

CONFIG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


@click.command()
@click.option("-d", "--dynamic_type", default="kolmogorov_128", show_default=True, type=click.Choice(sorted(os.listdir(CONFIG_ROOT))))
@click.option("--measurement_type", default="grid_mask", show_default=True, type=click.Choice(["grid_mask", "center_mask", "speed", "cubic"]))
@click.option("--seed", default=None, type=int)
@click.option("--exp", default=None, type=str)
@click.option("-c", "--config", default=None, type=str, help="Custom yaml (overrides -d/--measurement_type)")
def main(dynamic_type, measurement_type, seed, exp, config):
    cfg = load_yaml(config or os.path.join(CONFIG_ROOT, dynamic_type, f"{measurement_type}.yaml"))
    if seed is not None:
        cfg["system"]["seed"] = seed

    if not torch.cuda.is_available():
        cfg["system"]["device"] = "cpu"
        
    device = cfg["system"]["device"]
    print_run_info(cfg)
    set_seed(cfg)

    workdir = build_workdir(cfg, exp or os.path.join(dynamic_type, measurement_type, f"seed{cfg['system']['seed']}"))
    steps, total_step = build_steps(cfg)
    dynamics, measurement, method = build_dynamics(cfg), build_measurement(cfg), build_method(cfg)

    prior, trajectory, total_observations, observations = build_dataset(cfg, dynamics, measurement, steps)
    assimilated_states = torch.empty((len(trajectory), *prior.shape), device=device)  # [T, B, C, H, W]

    for index, i in enumerate(steps):
        print(f"{i}th step")
        model = load_trained_model(cfg, method, measurement, dynamics, i, index, steps, prior, device, workdir)
        posterior = measurement_update(method, cfg, model, measurement, i, total_step, prior, observations, device)
        prior, assimilated_states = time_update(dynamics, posterior, index, steps, total_step, assimilated_states)

    rmse, csi = finalize_and_save(cfg, workdir, dynamics, trajectory, observations, total_observations, assimilated_states, steps, total_step)
    click.echo(f"FINAL_RMSE {float(rmse)}")
    click.echo(f"FINAL_CSI {float(csi)}")


if __name__ == "__main__":
    main()
