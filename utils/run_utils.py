import yaml
import numpy as np
import torch


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_run_info(cfg):
    st = cfg["steps"]
    print("=" * 60)
    print(f"[RUN CONFIG] method={cfg['method']['name']} dynamics={cfg['dynamics']['type']} measurement={cfg['measurement']['type']}")
    print(f"  steps initial={st['initial']} end={st['end']} gap={st['gap']} seed={cfg['system']['seed']} device={cfg['system']['device']}")
    print("=" * 60)


def set_seed(cfg):
    seed = cfg["system"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def measurement_update(method, cfg, model, measurement, i, total_step, prior, observations, device):
    z = observations[int(np.where(total_step == i)[0][0])].unsqueeze(0).expand(prior.shape[0], *observations.shape[1:]).to(device)
    posterior_norm = method.sample(cfg, model, measurement.norm.norm(prior), measurement.norm.norm_z(z), measurement)
    
    return measurement.norm.denorm(posterior_norm)


def time_update(dynamics, posterior, index, steps, total_step, assimilated_states):
    if index >= len(steps) - 1:
        assimilated_states[-1] = posterior
        return posterior, assimilated_states
    
    p_step = int(np.where(total_step == steps[index])[0][0])
    next_step = int(np.where(total_step == steps[index + 1])[0][0])
    assimilated_states[p_step] = posterior
    next_prior = dynamics.states_at(posterior, np.arange(1, next_step - p_step + 1))
    assimilated_states[p_step + 1: next_step + 1] = next_prior
    
    return next_prior[-1], assimilated_states
