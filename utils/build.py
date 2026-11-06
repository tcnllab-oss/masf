import os
import numpy as np
import torch
import yaml
from torch.utils.data import TensorDataset, DataLoader, random_split
from dynamics import KolmogorovFlow, Lorenz96
from measurements import GridMask, CenterMask, Speed, Cubic
from methods import Ours, Nonlinear_Ours
from models import UNetModel, DualHeadUNetModel, DualHeadUNet1D
from .dataloder import find_reusable_dataset_dir, load_or_create_state

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_pretrained_path(cfg):
    path = os.path.join(REPO_ROOT, cfg["pretrain"]["ckpt"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def build_workdir(cfg, exp):
    workdir = os.path.join(cfg["exp"]["workdir_root"], exp)
    os.makedirs(os.path.join(workdir, "ckpt"), exist_ok=True)
    print("Workdir path is:", workdir)
    with open(os.path.join(workdir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return workdir


def build_steps(cfg):
    steps = np.arange(cfg["steps"]["initial"], cfg["steps"]["end"] + 1, cfg["steps"]["gap"])
    return steps, np.arange(steps[0], steps[-1] + 1)


def build_dynamics(cfg):
    dyn, seed = cfg["dynamics"], cfg["system"]["seed"]
    if dyn["type"] == "kolmogorov":
        return KolmogorovFlow(grid_size=dyn["dim"], reynolds=dyn["reynolds"], dt=dyn["dt"], seed=seed)
    return Lorenz96(dim=dyn["dim"], prior_mean=dyn["prior_mean"], prior_std=dyn["prior_std"], dt=dyn["dt"], forcing=dyn["forcing"], perturb_std=dyn["perturb_std"], solver=dyn["solver"], seed=seed)


def build_measurement(cfg):
    meas, dim = cfg["measurement"], cfg["dynamics"]["dim"]
    common = dict(noise_std=meas["noise_std"], device=cfg["system"]["device"], seed=cfg["system"]["seed"] + 1, same_normalization=meas["same_normalization"], normalization_form=meas["normalization_form"],
                  stats_mode=meas["stats_mode"], stats_update_mode=meas["stats_update_mode"], momentum=meas.get("momentum"))
    if meas["type"] == "grid_mask":
        return GridMask(stride=meas["stride"], grid_size=dim, **common)
    if meas["type"] == "center_mask":
        return CenterMask(grid_size=dim, hole_ratio=meas["hole_ratio"], **common)
    if meas["type"] == "speed":
        return Speed(tau=meas["tau"], sharpness=meas["sharpness"], **common)
    return Cubic(alpha=meas["alpha"], **common)

def build_method(cfg):
    return {"ours": Ours, "nonlinear_ours": Nonlinear_Ours}[cfg["method"]["name"]]()

def build_dataset(cfg, dynamics, measurement, steps):
    device, dyn = cfg["system"]["device"], cfg["dynamics"]
    num_samples, root = dyn["num_samples"], os.path.join(REPO_ROOT, "datasets", dyn["type"])
   
    out_dir = find_reusable_dataset_dir(root, dyn["dim"], num_samples) or os.path.join(root, f"dim{dyn['dim']}_num_samples{num_samples}")
    stored_num_samples = int(os.path.basename(out_dir).split("num_samples")[-1])
    prior = load_or_create_state(dynamics, out_dir, int(steps[0]), stored_num_samples, num_samples, device)
    
    total_step = np.arange(steps[0], steps[-1] + 1)
    trajectory = dynamics.states_at(dynamics.prior(1).to(device), total_step).squeeze(1).to(device)
    total_observations = measurement.observation_phy(trajectory).to(device)
    observations = torch.zeros_like(trajectory)
    observations[steps - steps[0]] = total_observations[steps - steps[0]]
    
    print(f"[DATASET] prior {tuple(prior.shape)} trajectory {tuple(trajectory.shape)} dir {out_dir}")
    return prior, trajectory, total_observations, observations


def build_pretraining_dataset(cfg, dynamics, steps):
    dyn, n = cfg["dynamics"], cfg["pretrain"]["num_samples"]
    out_dir = os.path.join(REPO_ROOT, "datasets", dyn["type"], f"dim{dyn['dim']}_num_samples{n}")
    prior = load_or_create_state(dynamics, out_dir, int(steps[0]), n, n, cfg["system"]["device"])
    print(f"[DATASET for pretraining] prior {tuple(prior.shape)} dir {out_dir}")
    return prior


def build_model(cfg):
    m, device = cfg["model"], cfg["system"]["device"]
    if m["type"] == "dual_unet1d":
        return DualHeadUNet1D(dim=m["model_dim"], init_dim=m["init_dim"], out_dim_1=m["out_dim_1"], out_dim_2=m["out_dim_2"], dim_mults=tuple(m["dim_mults"]), channels=m["channels"], dropout=m["dropout"],
                              sinusoidal_pos_emb_theta=m["sinusoidal_pos_emb_theta"], attn_dim_head=m["attn_dim_head"], attn_heads=m["attn_heads"], circular_padding=m["circular_padding"]).to(device)

    kw = dict(in_channels=m["in_channels"], model_channels=m["model_channels"], num_res_blocks=m["num_res_blocks"], attention_resolutions=tuple(m["attention_resolutions"]), dropout=m["dropout"],
              channel_mult=tuple(m["channel_mult"]), conv_resample=m["conv_resample"], dims=m["dims"], num_heads=m["num_heads"], num_head_channels=m["num_head_channels"],
              num_heads_upsample=m["num_heads_upsample"], use_scale_shift_norm=m["use_scale_shift_norm"], resblock_updown=m["resblock_updown"], use_new_attention_order=m["use_new_attention_order"])
    
    if m["type"] == "dual_unet":
        return DualHeadUNetModel(out_channels_1=m["out_channels1"], out_channels_2=m["out_channels2"], **kw).to(device)
    
    return UNetModel(image_size=cfg["dynamics"]["dim"], out_channels=m["out_channels"], **kw).to(device)


def build_dataloader(prior, batch_size, val_ratio):
    dataset = TensorDataset(prior)
    n_val = int(len(dataset) * val_ratio)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
