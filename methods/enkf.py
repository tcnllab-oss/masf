from .base import Method
import torch


def make_gen_from_measurement(measurement, seed_offset=1):
    device = measurement.device
    if isinstance(device, str):
        device = torch.device(device)

    base_seed = getattr(measurement, "seed", 0)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(base_seed) + int(seed_offset))
    return gen


def _get_mask(measurement, y_like):
    if not hasattr(measurement, "mask"):
        return None
    m = measurement.mask
    if not torch.is_tensor(m):
        m = torch.tensor(m, device=y_like.device)
    m = m.to(y_like.device)
    while m.ndim < y_like.ndim:
        m = m.unsqueeze(0)
    return m


def _flatten_batch(x):
    return x.reshape(x.shape[0], -1)


def _unflatten_batch(x_flat, ref_shape):
    return x_flat.reshape(ref_shape)


@torch.no_grad()
def enkf_update(
    prior,
    z_one,
    measurement,
    gen,
    stochastic=True,
    enkf_mode="full",   # "diag" | "full"
    inflation=1.0,
    eps=1e-6,
):
    device = prior.device
    B = prior.shape[0]

    if not hasattr(measurement, "observation"):
        raise AttributeError("measurement must have observation(x).")

    y_hat = measurement.observation_clean_phy(prior)

    # observation: use a single observed sample, then repeat across ensemble
    z = z_one[0:1]  # shape: (1, ...)
    z = z.expand_as(y_hat)  # shape: (B, ...)

    mask = _get_mask(measurement, y_hat)
    obs_idx = None

    if mask is not None:
        mask_flat = _flatten_batch(mask.expand_as(y_hat))
        obs_idx = (mask_flat[0] > 0.5).nonzero(as_tuple=False).squeeze(1)

    X = _flatten_batch(prior)
    Y = _flatten_batch(y_hat)
    Z = _flatten_batch(z)

    if obs_idx is not None:
        Y = Y[:, obs_idx]
        Z = Z[:, obs_idx]

    x_mean = X.mean(dim=0, keepdim=True)
    y_mean = Y.mean(dim=0, keepdim=True)

    X_a = X - x_mean
    Y_a = Y - y_mean

    if inflation != 1.0:
        X_a = X_a * inflation
        Y_a = Y_a * inflation
        X = x_mean + X_a
        Y = y_mean + Y_a

    noise_std = float(getattr(measurement, "noise_std", 1.0))
    r2 = noise_std * noise_std

    # stochastic EnKF: independent perturbed observations for each ensemble member
    if stochastic:
        obs_noise = torch.randn(
            Z.shape, device=Z.device, generator=gen, dtype=Z.dtype
        ) * noise_std
        Z_tilde = Z + obs_noise
    else:
        Z_tilde = Z

    denom = max(B - 1, 1)
    Pxy = (X_a.t() @ Y_a) / denom
    innov = Z_tilde - Y

    if enkf_mode == "full":
        Pyy = (Y_a.t() @ Y_a) / denom
        S = Pyy + torch.eye(Pyy.shape[0], device=device, dtype=Pyy.dtype) * r2
        S = S + eps * torch.eye(Pyy.shape[0], device=device, dtype=Pyy.dtype)
        try:
            K = torch.linalg.solve(S.t(), Pxy.t()).t()
        except torch.OutOfMemoryError:
            print("[EnKF] CUDA OOM in full solve, fallback to CPU")
            S_cpu = S.detach().cpu()
            Pxy_cpu = Pxy.detach().cpu()
            K = torch.linalg.solve(S_cpu.t(), Pxy_cpu.t()).t().to(device)

        X_post = X + (innov @ K.t())
    elif enkf_mode == "diag":
        var_y = (Y_a.pow(2).sum(dim=0) / denom)
        S_diag = var_y + r2 + eps
        K = Pxy / S_diag.unsqueeze(0)
        X_post = X + (innov @ K.t())
    else:
        raise ValueError("enkf_mode must be 'full' or 'diag'")

    post = _unflatten_batch(X_post, prior.shape)
    return post


class EnKF(Method):
    name = "EnKF"

    def sample(self, cfg, model, prior, z, measurement, device=None, path=None):
        # prefer cfg["seed"] if present, otherwise fall back to cfg["system"]["seed"]
        seed = cfg.get("seed", cfg.get("system", {}).get("seed", 0))

        gen = make_gen_from_measurement(measurement, seed_offset=seed)

        posterior = enkf_update(
            prior=prior,
            z_one=z,
            measurement=measurement,
            gen=gen,
            stochastic=cfg["method"].get("stochastic", True),
            enkf_mode=cfg["method"].get("enkf_mode", "diag"),
            inflation=cfg["method"].get("inflation", 1.0),
            eps=cfg["method"].get("eps", 1e-6),
        )
        return posterior