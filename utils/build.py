import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset, DataLoader, random_split

from dynamics import KolmogorovFlow
from measurements import GridMask, Nonlinear, CenterMask
from methods import Ours, EnKF, LETKF, SF, SSLS, Nonlinear_Ours
from .dataloder import save_trajectory_by_step, iter_load_states_from_folder, find_reusable_dataset_dir
from models import UNetModel, DualHeadUNetModel


def _list_to_str(x):
    if isinstance(x, (list, tuple)):
        return "-".join(map(str, x))
    return str(x)


def build_dynamic_name(cfg):
    dynamic_type = cfg["dynamics"]["type"]
    dim = cfg["dynamics"]["dim"]
    measurement_type = cfg["measurement"]["type"]

    name = f"{dynamic_type}_{dim}"
    if measurement_type == "nonlinear":
        name = f"{name}_nonlinear"
    return name


def build_measurement_setting_name(cfg):
    meas = cfg.get("measurement", {}) or {}
    measurement_type = meas.get("type", "grid_mask")
    nonlinear_type = meas.get("nonlinear_type", None)

    if measurement_type == "grid_mask":
        stride = meas.get("stride")
        return f"stride_{stride}" if stride is not None else "default"

    if measurement_type == "center_mask":
        hole_ratio = meas.get("hole_ratio")
        return f"hole_{hole_ratio}" if hole_ratio is not None else "default"

    if measurement_type == "nonlinear":
        alpha = meas.get("alpha")
        if nonlinear_type is not None and alpha is not None:
            return f"{nonlinear_type}_alpha_{alpha}"
        if nonlinear_type is not None:
            return str(nonlinear_type)
        return "default"

    return "default"


# -------------------------------------------------
# build base_workdir for dataset
# -------------------------------------------------
def build_base_workdir(cfg):
    """
    datasets/{dynamic_type}/dim{dim}_num_samples{num_samples}/{measurement_type}/{measurement_setting}
    """
    dynamic_type = cfg["dynamics"]["type"]
    dim = cfg["dynamics"]["dim"]
    num_samples = cfg["pretrain"]["num_samples"]
    measurement_type = cfg["measurement"]["type"]
    method_name = cfg["method"]["name"]
    measurement_setting = build_measurement_setting_name(cfg)
    print('method_name',method_name)
    if method_name in ["ours", "nonlinear_ours"]:
        base_workdir = os.path.join(
            "datasets",
            str(dynamic_type).lower(),
            f"dim{dim}_num_samples{num_samples}",
            measurement_type,
            measurement_setting,
        )
        os.makedirs(base_workdir, exist_ok=True)
        return base_workdir
    
    else:  
        base_workdir = os.path.join(
            "datasets",
            str(dynamic_type).lower(),
            str(method_name).lower(),
            f"dim{dim}_num_samples{num_samples}",
            measurement_type,
            measurement_setting,
        )
        print('base_workdir',base_workdir)
        os.makedirs(base_workdir, exist_ok=True)
        return base_workdir


def _list_to_str(xs):
    return "-".join(map(str, xs))

# -------------------------------------------------
# build pretrained model dir
# -------------------------------------------------
def build_model_dir(cfg):
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]

    if model_type in {"UNet", "dual_unet"}:
        model_channels = model_cfg["model_channels"]
        num_res_blocks = model_cfg["num_res_blocks"]
        attention_resolutions = _list_to_str(model_cfg["attention_resolutions"])
        channel_mult = _list_to_str(model_cfg["channel_mult"])

        model_dir = (
            f"model_mc{model_channels}_"
            f"rb{num_res_blocks}_"
            f"attn{attention_resolutions}_"
            f"cm{channel_mult}"
        )
        return model_dir

    elif model_type == "TimeCondMLP":
        hidden_dim = model_cfg["hidden_dim"]
        depth = model_cfg["depth"]

        model_dir = f"model_hd{hidden_dim}_depth{depth}"
        return model_dir

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
# -------------------------------------------------
# build pretrained_workdir
# -------------------------------------------------
def build_pretrained_workdir(cfg):
    """
    base_workdir/norm_*/model_*
    """
    base_workdir = build_base_workdir(cfg)
    print('innter base', base_workdir)

    same_normalization = str(
        cfg["measurement"].get("same_normalization", False)
    ).lower()
    normalization_form = str(
        cfg["measurement"].get("normalization_form", "none")
    )
    stats_mode = str(cfg["measurement"].get("stats_mode", "none"))

    norm_dir = (
        f"norm_{same_normalization}_"
        f"{normalization_form}_"
        f"{stats_mode}"
    )

    model_dir = build_model_dir(cfg)
    pretrained_workdir = os.path.join(
        base_workdir,
        norm_dir,
        model_dir,
    )

    os.makedirs(pretrained_workdir, exist_ok=True)
    return pretrained_workdir


# -------------------------------------------------
# build pretrained_path
# -------------------------------------------------
def build_pretrained_path(cfg):
    pretrain_cfg = cfg["pretrain"]
    pretrained_workdir = build_pretrained_workdir(cfg)

    filename = (
        f"batch{pretrain_cfg['batch_size']}_"
        f"epoch{pretrain_cfg['epoch']}_"
        f"lr{pretrain_cfg['lr']}"
    )

    sigma = cfg["method"].get("sigma")
    if sigma is not None:
        filename += f"_sigma{sigma}"

    filename += "_ckpt.pt"
    path = os.path.join(pretrained_workdir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def build_workdir(cfg, exp=None, make_ckpt=True):
    root = cfg["exp"]["workdir_root"]
    exp_name = exp if exp else "run"

    workdir = os.path.join(root, exp_name)
    if make_ckpt:
        os.makedirs(os.path.join(workdir, "ckpt"), exist_ok=True)
    else:
        os.makedirs(workdir, exist_ok=True)

    print("Workdir path is:", workdir)

    cfg_path = os.path.join(workdir, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    return workdir

def build_steps(cfg):
    initial = cfg["steps"]["initial"]
    end = cfg["steps"]["end"]
    gap = cfg["steps"]["gap"]
    steps = np.arange(initial, end + 1, gap)
    total_step = np.arange(steps[0], steps[-1] + 1)
    return steps, total_step

# -------------------------------------------------
# build dynamics
# -------------------------------------------------
def build_dynamics(cfg):
    dyn = cfg.get("dynamics", {}) or {}
    dyn_type = dyn.get("type")
    seed = (cfg.get("system", {}) or {}).get("seed", 42)
    dt = dyn.get("dt", 0.01)

    if dyn_type == "kolmogorov":
        grid_size = dyn.get("image_size", dyn.get("dim", 64))
        reynolds = dyn.get("reynolds", 1000)
        return KolmogorovFlow(
            grid_size=grid_size,
            reynolds=reynolds,
            dt=dt,
            seed=seed,
        )
    raise ValueError(f"Unknown dynamics.type: {dyn_type}")


# -------------------------------------------------
# build measurement
# -------------------------------------------------
def build_measurement(cfg):
    same_normalization = cfg["measurement"].get("same_normalization", True)
    normalization_form = cfg["measurement"].get("normalization_form", "affine")
    stats_mode = cfg["measurement"].get("stats_mode", "robust")
    stats_update_mode = cfg["measurement"].get("stats_update_mode", "fixed")
    momentum = cfg["measurement"].get("momentum", 0.05)

    meas = (cfg.get("measurement", {}) or {})
    syscfg = (cfg.get("system", {}) or {})
    dyn = (cfg.get("dynamics", {}) or {})

    measure_type = meas.get("type", "linear")
    std = meas.get("noise_std", 0.1)
    device = syscfg.get("device", "cpu")
    seed = syscfg.get("seed", 42) + 1

    dim = dyn.get("dim")

    if measure_type == "grid_mask":
        stride = cfg["measurement"]["stride"]
        measurement = GridMask(
            noise_std=std,
            stride=stride,
            grid_size=dim,
            device=device,
            seed = seed,
            same_normalization=same_normalization,
            normalization_form=normalization_form,
            stats_mode=stats_mode,
            stats_update_mode=stats_update_mode,
            momentum=momentum,
        )

    elif measure_type == "center_mask":
        hole_ratio = cfg["measurement"].get("hole_ratio", 0.6)
        measurement = CenterMask(
            noise_std=std,
            grid_size=dim,
            hole_ratio=hole_ratio,
            device=device,
            seed = seed,
            same_normalization=same_normalization,
            normalization_form=normalization_form,
            stats_mode=stats_mode,
            stats_update_mode=stats_update_mode,
            momentum=momentum,
        )

    elif measure_type == "nonlinear":
        measurement = Nonlinear(
            noise_std=std,
            type=meas.get("nonlinear_type", "sigmoid"),
            alpha=cfg["measurement"].get("alpha", 2),
            device=device,
            seed = seed,
            same_normalization=same_normalization,
            normalization_form=normalization_form,
            stats_mode=stats_mode,
            stats_update_mode=stats_update_mode,
            momentum=momentum,
        )

    else:
        raise ValueError(f"Unknown measurement type: {measure_type}")

    return measurement


# -------------------------------------------------
# build method
# -------------------------------------------------
def build_method(cfg):
    m = (cfg.get("method") or {}).get("name")
    if m is None:
        raise ValueError("cfg['method']['name'] is required (enkf/letkf/sf/ssls/ours/nonlinear_ours).")
    m = str(m).lower()
    if m == "enkf":
        return EnKF()
    elif m == "letkf":
        return LETKF()
    elif m == "sf":
        return SF()
    elif m == "ssls":
        return SSLS()
    elif m == "ours":
        return Ours()
    elif m == "nonlinear_ours":
        return Nonlinear_Ours()
    raise ValueError("unknown method: " + str(m))

# -------------------------------------------------
# build dataset
# -------------------------------------------------
def build_dataset(cfg, dynamics, measurement, steps):
    device = cfg["system"]["device"]
    dyn = cfg["dynamics"]
    dynamic_type = dyn["type"]

    num_samples = int(dyn.get("num_samples", 400))
    dim = int(dyn["dim"])

    project_root = Path(__file__).resolve().parent.parent
    dataset_root = project_root / "datasets" / str(dynamic_type).lower()

    reusable_dir = find_reusable_dataset_dir(str(dataset_root), dim, num_samples)
    if reusable_dir is None:
        out_dir = dataset_root / f"dim{dim}_num_samples{num_samples}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stored_num_samples = num_samples
    else:
        out_dir = Path(reusable_dir).resolve()
        stored_num_samples = int(out_dir.name.split("num_samples")[-1])

    initial_step = int(steps[0])
    final_step = int(steps[-1])
    total_step = np.arange(initial_step, final_step + 1)

    step_path = out_dir / f"step_{initial_step:06d}.pt"
    print("step_path =", step_path)

    if not step_path.exists():
        x0 = dynamics.prior(stored_num_samples).to(device)
        save_trajectory_by_step(
            dynamics,
            x0=x0,
            steps=[initial_step],
            out_dir=str(out_dir),
            save_dtype="fp32",
            cpu_store=True,
            overwrite=False,
            meta_extra={"note": "experiment A"},
        )

    _, prior = next(
        iter(
            iter_load_states_from_folder(
                str(out_dir),
                steps=[initial_step],
                map_device=device,
                out_dtype=torch.float32,
                return_step=True,
                max_samples=num_samples,
            )
        )
    )

    initial = dynamics.prior(1).to(device)
    trajectory = dynamics.states_at(initial, total_step).squeeze(1).to(device)
    total_observations = measurement.observation_phy(trajectory).to(device)

    relative_steps = steps - steps[0]
    observations = torch.zeros_like(trajectory).to(device)
    observations[relative_steps] = total_observations[relative_steps].to(device)

    print("[DATASET for training]")
    print("  requested num_samples =", num_samples)
    print("  stored num_samples    =", stored_num_samples)
    print("  prior shape           =", tuple(prior.shape))
    print("  trajectory shape      =", tuple(trajectory.shape))
    print("  dataset dir           =", out_dir)
    print("============================================================")

    return prior, trajectory, total_observations, observations


def build_pretraining_dataset(cfg, dynamics, steps):
    device = cfg["system"]["device"]

    dyn = cfg["dynamics"]
    pre = cfg["pretrain"]

    dynamic_type = dyn["type"]                   
    dim = int(dyn["dim"])                        
    num_samples = int(pre.get("num_samples", 1000))

    project_root = Path(__file__).resolve().parent.parent
    dataset_root = project_root / "datasets" / str(dynamic_type).lower()
    out_dir = dataset_root / f"dim{dim}_num_samples{num_samples}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # e.g. /.../datasets/kolmogorov/dim256_num_samples1000

    initial_step = int(steps[0])                  # e.g. 50
    step_path = out_dir / f"step_{initial_step:06d}.pt"
    # e.g. /.../dim256_num_samples1000/step_000050.pt

    print("pretrain_step_path =", step_path)

    # Create the requested dataset only when the target step file is missing.
    if not step_path.exists():
        x0 = dynamics.prior(num_samples).to(device)   # [num_samples, ...]
        save_trajectory_by_step(
            dynamics,
            x0=x0,
            steps=[initial_step],
            out_dir=str(out_dir),
            save_dtype="fp32",
            cpu_store=True,
            overwrite=False,
            meta_extra={
                "dynamics_type": str(dynamic_type),
                "dim": dim,
                "num_samples": num_samples,
                "saved_for": "pretraining",
                "initial_step": initial_step,
            },
        )

    # Load exactly the requested number of samples from the saved step file.
    _, prior = next(
        iter(
            iter_load_states_from_folder(
                str(out_dir),
                steps=[initial_step],
                map_device=device,
                out_dtype=torch.float32,
                return_step=True,
                max_samples=num_samples,
            )
        )
    )

    print("[DATASET for pretraining]")
    print("  requested num_samples =", num_samples)
    print("  prior shape           =", tuple(prior.shape))
    print("  out_dir               =", out_dir)
    print("=" * 60)

    return prior

def _as_tuple(x, default=None):
    if x is None:
        return default
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, str):
        parts = [p.strip() for p in x.split(",") if p.strip() != ""]
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                out.append(p)
        return tuple(out)
    return default



def build_model(cfg):
    method_cfg = cfg.get("method") or {}
    method_name = method_cfg.get("name")
    if not bool(method_cfg.get("train", False)):
        return None

    syscfg = cfg.get("system") or {}
    device = syscfg.get("device", "cuda:0")

    dyn = cfg.get("dynamics") or {}
    dynamic_type = (dyn.get("type") or "").lower()
    if not dynamic_type:
        raise ValueError("cfg['dynamics']['type'] is required.")

    model_cfg = cfg.get("model") or {}
    model_type = (model_cfg.get("type") or "").lower()
    if not model_type:
        raise ValueError("cfg['model']['type'] is required.")

    dyn_dim = dyn.get("dim")

    if model_type == "dual_unet" and method_name=="nonlinear_ours":
        model = DualHeadUNetModel(
            in_channels=model_cfg.get("in_channels", 2),
            model_channels=model_cfg.get("model_channels", 128),
            out_channels_1=model_cfg.get("out_channels1", 2),
            out_channels_2=model_cfg.get("out_channels2", 2),
            num_res_blocks=model_cfg.get("num_res_blocks", 2),
            attention_resolutions=model_cfg.get("attention_resolutions", (16, 8)),
            dropout=model_cfg.get("dropout", 0),
            channel_mult=model_cfg.get("channel_mult", (1, 2, 4, 8)),
            conv_resample=model_cfg.get("conv_resample", True),
            dims=model_cfg.get("dims", 2),
            num_classes=model_cfg.get("num_classes", None),
            use_checkpoint=model_cfg.get("use_checkpoint", False),
            use_fp16=model_cfg.get("use_fp16", False),
            num_heads=model_cfg.get("num_heads", 1),
            num_head_channels=model_cfg.get("num_head_channels", -1),
            num_heads_upsample=model_cfg.get("num_heads_upsample", -1),
            use_scale_shift_norm=model_cfg.get("use_scale_shift_norm", False),
            resblock_updown=model_cfg.get("resblock_updown", False),
            use_new_attention_order=model_cfg.get("use_new_attention_order", False),
        )
        return model

    else:
        in_channels = model_cfg.get("in_channels", 2)
        out_channels = model_cfg.get("out_channels", 2)
        model_channels = model_cfg.get("model_channels", 64)
        num_res_blocks = model_cfg.get("num_res_blocks", 2)
        attention_resolutions = _as_tuple(model_cfg.get("attention_resolutions"), default=(4, 8))
        dropout = model_cfg.get("dropout", 0.0)

        channel_mult = _as_tuple(model_cfg.get("channel_mult"), default=(1, 2, 4))
        conv_resample = bool(model_cfg.get("conv_resample", True))
        dims = int(model_cfg.get("dims", 2))

        num_heads = int(model_cfg.get("num_heads", 1))
        num_head_channels = int(model_cfg.get("num_head_channels", -1))
        num_heads_upsample = int(model_cfg.get("num_heads_upsample", -1))
        use_scale_shift_norm = bool(model_cfg.get("use_scale_shift_norm", True))
        resblock_updown = bool(model_cfg.get("resblock_updown", True))
        use_new_attention_order = bool(model_cfg.get("use_new_attention_order", False))

        image_size = int(dyn_dim)

        model = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(int(x) for x in attention_resolutions),
            dropout=float(dropout),
            channel_mult=tuple(int(x) for x in channel_mult),
            conv_resample=conv_resample,
            dims=dims,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        ).to(device)
        return model


def build_optimizer(cfg, model):
    method_cfg = cfg.get("method") or {}
    if (model is None) or (not bool(method_cfg.get("train", False))):
        return None
    train_cfg = cfg.get("train") or {}
    lr = float(train_cfg.get("lr"))
    betas = (0.9, 0.99)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)


def build_ckpt_paths(workdir, model):
    if model is None:
        return None
    ckpt_dir = os.path.join(workdir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, "last.pth")


def build_dataloader(prior, batch_size, val_ratio):
    dataset = TensorDataset(prior)
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader