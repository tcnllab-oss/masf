
import click 
import torch
from utils.train_load import load_trained_model
from utils.run_utils import (load_yaml, merge_cfg, set_seed,
                             measurement_update, time_update)
from utils.build import (build_workdir, build_steps, build_method,
                         build_dynamics, build_measurement, build_dataset)
from utils.summary import finalize_and_save             



@click.command()
@click.option("--base", default="base", show_default=True)

@click.option("-d", "--dynamic_type", default="kolmogorov_128", show_default=True)
@click.option( "--measurement_type", type=click.Choice(["linear", "grid_mask", "center_mask", "blur", "low_resolution", "nonlinear"]), 
               default="grid_mask", show_default=True)

@click.option("-m", "--method_type", type=click.Choice(["apf", "enkf", "letkf", "sf", "ssls", "ours", "nonlinear_ours"], 
              case_sensitive=False), default = "ours")

@click.option("--seed", default=None, type=int)
@click.option("--exp", default=None, type=str)
@click.option("--config", type=str, default=None)

def main(base, method_type, 
         dynamic_type, measurement_type, 
         seed, exp, config):
    
    # Configure merging & update
    if config is not None:
        print(f"[INFO] Loading config from: {config}")
        cfg = load_yaml(config)
    else:
        cfg = merge_cfg(base, method_type, dynamic_type, measurement_type, seed)

    # Setting seed
    set_seed(cfg)
    device = cfg["system"]["device"] if not cfg["system"]["device"].startswith("cuda") or torch.cuda.is_available() else "cpu"
    
    # override
    cfg["system"]["device"] = device
    print('cfg["system"]["device"]', cfg["system"]["device"]) 

    # Building workdir
    workdir = build_workdir(cfg, exp=exp, make_ckpt=True)

    # Defining step
    steps, total_step = build_steps(cfg)


    # Defining dynamic
    dynamics = build_dynamics(cfg)
    measurement = build_measurement(cfg)

    # Definining method
    method = build_method(cfg)

    # Constructing prior X0, trajectory X_T, total measurement Z, measurement Z
    prior, trajectory, total_observations, observations = build_dataset(cfg, dynamics, measurement, steps)
    assimilated_states = torch.empty((len(trajectory), *prior.shape), device=device) # [T, B, C, H, W]

    start_idx = 0
    for index in range(start_idx, len(steps)):
        i = steps[index]
        print(f"{i}th step")

        # Train or load model 
        #if enkf, letkf,.. -> model=None, need_normalization=None
        #else -> model=score_fn, need_normalization=True
        model, need_normalization = load_trained_model(cfg, method, measurement, dynamics, 
                                                 i, index, steps, prior, device, workdir,
                                                 map_location="cpu", strict=True)
        
        # measurement-update step
        posterior = measurement_update(method, cfg, i, total_step,
                                       prior, observations,
                                       need_normalization=need_normalization,
                                       model=model, measurement=measurement, 
                                       device=device, path=None)
        
        # time-update step
        prior, assimilated_states = time_update(dynamics, posterior,
                                                index, steps, total_step,
                                                assimilated_states)

 
    rmse, csi = finalize_and_save(
            cfg, workdir,
            dynamics,
            measurement,
            trajectory,
            observations,
            total_observations,
            assimilated_states,
            steps,
            total_step,
            dynamic_type
        )

    click.echo(f"FINAL_RMSE {float(rmse)}")
    click.echo(f"FINAL_CSI {float(csi)}")

    return rmse, csi
            

if __name__ == "__main__":
    main()

