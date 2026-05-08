import os
import yaml
import re
import numpy as np
import torch
import click


HERE = os.path.dirname(os.path.abspath(__file__))           # .../project/utils
PROJECT_ROOT = os.path.dirname(HERE)                        # .../project
CONFIG_ROOT = os.path.join(PROJECT_ROOT, "configs") 


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

#upload yaml
def _load_yaml(path):
    if path is None:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

#config update
def _deep_update(dst, src):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

# cfg override
def _set_if_not_none(cfg, key_path, value):
    if value is None:
        return
    cur = cfg
    for k in key_path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[key_path[-1]] = value


def _cfg_path(kind, name):
    if name is None:
        return None
    name = name + ".yaml"
    if kind is None or kind == "":
        return os.path.join(CONFIG_ROOT, name)
    return os.path.join(CONFIG_ROOT, kind, name)

def strip_numeric_suffix(name):
    if name is None:
        return None
    return re.sub(r'_.*$', '', name)


# merge yaml
def merge_cfg(base=None, method_type=None, dynamic_type=None,
              measurement_type=None, seed=None):
    cfg = {}
    dynamic_type_clean = strip_numeric_suffix(dynamic_type)

    # 1) base / method / dynamics loading
    _deep_update(cfg, _load_yaml(_cfg_path(None, base)))
    _deep_update(cfg, _load_yaml(_cfg_path("methods", method_type)))
    _deep_update(cfg, _load_yaml(_cfg_path("measurements", measurement_type)))
    _deep_update(cfg, _load_yaml(_cfg_path("dynamics", dynamic_type)))

    # seed / dynamics overrides
    _set_if_not_none(cfg, ["system", "seed"], seed)

    print_run_info(cfg)
    return cfg

# print info
def print_run_info(cfg):
    dyn = cfg.get("dynamics", {}) or {}
    meas = cfg.get("measurement", {}) or {}
    syscfg = cfg.get("system", {}) or {}
    steps = cfg.get("steps", {}) or {}
    model = cfg.get("model", {}) or {}
    method_type = cfg["method"]["name"]

    click.echo("============================================================")
    click.echo("[RUN CONFIG]")
    click.echo(f"  method        = {method_type}")
    click.echo(f"  dynamics      = {dyn.get('type')}")
    click.echo(f"  measurement   = {meas.get('type')}")
    click.echo(f"  steps         = initial={steps.get('initial')} end={steps.get('end')} gap={steps.get('gap')}")
    click.echo(f"  seed          = {syscfg.get('seed')}")
    click.echo(f"  device        = {syscfg.get('device')}")

    if model.get("dim") is not None:
        click.echo(f"  Lorenz96.dim     = {model.get('dim')}")
    if dyn.get("rho") is not None:
        click.echo(f"  Lorenz63 rho= {dyn.get('rho')}")
    click.echo("============================================================")

# set seed 
def set_seed(cfg):
    seed = cfg["system"]["seed"]
    device = cfg["system"]["device"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def measurement_update(method, cfg, i, total_step,
                       prior, observations,
                       need_normalization =True, 
                       model=None, measurement=None, 
                       device=None, path=None):

    idx = int(np.where(total_step == i)[0][0])
    B = prior.shape[0]
    z = observations[idx].unsqueeze(0).expand(B, *observations.shape[1:])
    
    if device is not None:
        z = z.to(device)

    if need_normalization:
        z_norm = measurement.norm.norm_z(z)
        prior_norm = measurement.norm.norm(prior)
    else:
        z_norm = z
        prior_norm = prior

    posterior_norm = method.sample(cfg, model, prior_norm, z_norm, measurement, path=path)

    if need_normalization:
       posterior = measurement.norm.denorm(posterior_norm)
    else:
       posterior = posterior_norm

    return posterior


def time_update(dynamics, posterior,
                index, steps, total_step,
                assimilated_states):
    if index >= len(steps) - 1:
        prior = posterior
        assimilated_states[-1] = posterior
        return prior, assimilated_states

    p_step = int(np.where(total_step == steps[index])[0][0])
    assimilated_states[p_step : p_step +1] = posterior

    next_step = int(np.where(total_step == steps[index + 1])[0][0])

    next_step_len = next_step - p_step
    nexts = np.arange(1, next_step_len + 1)
    next_prior = dynamics.states_at(posterior, nexts)
    
    assimilated_states[p_step + 1: next_step + 1] = next_prior
    prior = next_prior[-1]

    return prior, assimilated_states