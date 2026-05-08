from __future__ import annotations
import math
from typing import List, Tuple
import torch
from .base import Method


def _get_mask(measurement, y_like: torch.Tensor):
    if not hasattr(measurement, "mask"):
        return None

    m = measurement.mask
    if m is None:
        return None

    if not torch.is_tensor(m):
        m = torch.tensor(m, device=y_like.device)

    m = m.to(device=y_like.device, dtype=y_like.dtype)

    while m.ndim < y_like.ndim:
        m = m.unsqueeze(0)

    return m


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def _unflatten_batch(x_flat: torch.Tensor, ref_shape) -> torch.Tensor:
    return x_flat.reshape(ref_shape)


def _flat_idx_to_coord(idx: int, state_shape: Tuple[int, ...]):
    if len(state_shape) == 3:
        c, hw = divmod(idx, state_shape[1] * state_shape[2])
        h, w = divmod(hw, state_shape[2])
        return int(c), int(h), int(w)

    if len(state_shape) == 2:
        c, h = divmod(idx, state_shape[1])
        return int(c), int(h)

    if len(state_shape) == 1:
        return (int(idx),)

    raise ValueError(f"Unsupported state shape: {state_shape}")


def _periodic_delta(a: int, b: int, n: int) -> int:
    d = abs(a - b)
    return min(d, n - d)


def gaspari_cohn(dist: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Gaspari-Cohn taper with compact support 2*radius.
    dist >= 0
    """
    if radius <= 0:
        return torch.ones_like(dist)

    r = dist / radius
    w = torch.zeros_like(r)

    m1 = r <= 1
    x = r[m1]
    w[m1] = (((-0.25 * x + 0.5) * x + 0.625) * x - 5.0 / 3.0) * x * x + 1.0

    m2 = (r > 1) & (r <= 2)
    x = r[m2]
    w[m2] = ((((x / 12.0 - 0.5) * x + 0.625) * x + 5.0 / 3.0) * x - 5.0) * x + 4.0 - 2.0 / (3.0 * x)

    return torch.clamp(w, min=0.0, max=1.0)


def _build_local_obs_info(
    mask_flat_1d: torch.Tensor,
    state_shape: Tuple[int, ...],
    loc_radius: float,
    same_channel_only: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns for each state flat index j:
      (obs_indices_j, localization_weights_j)

    obs_indices_j: LongTensor [m_j]
    localization_weights_j: Tensor [m_j] in [0, 1]
    """
    device = mask_flat_1d.device
    dtype = mask_flat_1d.dtype
    obs_bool = mask_flat_1d > 0.5
    D = mask_flat_1d.numel()

    local_info: List[Tuple[torch.Tensor, torch.Tensor]] = []

    if len(state_shape) == 3:
        C, H, W = state_shape

        obs_points = []
        for idx in range(D):
            if not obs_bool[idx]:
                continue
            c, h, w = _flat_idx_to_coord(idx, state_shape)
            obs_points.append((idx, c, h, w))

        for j in range(D):
            cj, hj, wj = _flat_idx_to_coord(j, state_shape)

            idxs = []
            dists = []

            for obs_idx, co, ho, wo in obs_points:
                if same_channel_only and co != cj:
                    continue

                dh = _periodic_delta(hj, ho, H)
                dw = _periodic_delta(wj, wo, W)
                dist = math.sqrt(dh * dh + dw * dw)

                if dist <= 2.0 * loc_radius:
                    idxs.append(obs_idx)
                    dists.append(dist)

            if len(idxs) == 0:
                local_info.append(
                    (
                        torch.empty(0, dtype=torch.long, device=device),
                        torch.empty(0, dtype=dtype, device=device),
                    )
                )
            else:
                idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
                dists_t = torch.tensor(dists, dtype=dtype, device=device)
                weights_t = gaspari_cohn(dists_t, loc_radius)
                local_info.append((idxs_t, weights_t))

        return local_info

    if len(state_shape) == 2:
        C, H = state_shape

        obs_points = []
        for idx in range(D):
            if not obs_bool[idx]:
                continue
            c, h = _flat_idx_to_coord(idx, state_shape)
            obs_points.append((idx, c, h))

        for j in range(D):
            cj, hj = _flat_idx_to_coord(j, state_shape)

            idxs = []
            dists = []

            for obs_idx, co, ho in obs_points:
                if same_channel_only and co != cj:
                    continue

                dh = _periodic_delta(hj, ho, H)
                dist = float(dh)

                if dist <= 2.0 * loc_radius:
                    idxs.append(obs_idx)
                    dists.append(dist)

            if len(idxs) == 0:
                local_info.append(
                    (
                        torch.empty(0, dtype=torch.long, device=device),
                        torch.empty(0, dtype=dtype, device=device),
                    )
                )
            else:
                idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
                dists_t = torch.tensor(dists, dtype=dtype, device=device)
                weights_t = gaspari_cohn(dists_t, loc_radius)
                local_info.append((idxs_t, weights_t))

        return local_info

    if len(state_shape) == 1:
        obs_points = [idx for idx in range(D) if obs_bool[idx]]

        for j in range(D):
            idxs = []
            dists = []

            for obs_idx in obs_points:
                dist = abs(j - obs_idx)
                if dist <= 2.0 * loc_radius:
                    idxs.append(obs_idx)
                    dists.append(float(dist))

            if len(idxs) == 0:
                local_info.append(
                    (
                        torch.empty(0, dtype=torch.long, device=device),
                        torch.empty(0, dtype=dtype, device=device),
                    )
                )
            else:
                idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
                dists_t = torch.tensor(dists, dtype=dtype, device=device)
                weights_t = gaspari_cohn(dists_t, loc_radius)
                local_info.append((idxs_t, weights_t))

        return local_info

    raise ValueError(f"Unsupported state shape for LETKF: {state_shape}")


@torch.no_grad()
def letkf_update(
    prior: torch.Tensor,
    z_one: torch.Tensor,
    measurement,
    inflation: float = 1.0,
    eps: float = 1e-8,
    loc_radius: float = 3.0,
    same_channel_only: bool = False,
):
    """
    Local ensemble transform Kalman filter (deterministic square-root LETKF).

    prior: [B, ...]
    z_one: single observation, broadcastable to prior[0]
    """
    device = prior.device
    dtype = prior.dtype

    B = prior.shape[0]
    state_shape = prior.shape[1:]
    D = prior[0].numel()

    if B < 2:
        raise ValueError("LETKF requires ensemble size B >= 2.")

    mask = _get_mask(measurement, prior)
    if mask is None:
        mask = torch.ones_like(prior, device=device, dtype=dtype)

    while mask.ndim < prior.ndim:
        mask = mask.unsqueeze(0)
    mask_b = mask.expand_as(prior)

    y_hat = measurement.observation_clean_phy(prior)

    z = z_one.to(device=device, dtype=dtype)
    while z.ndim < prior.ndim:
        z = z.unsqueeze(0)
    z = z.expand_as(prior)

    noise_std = getattr(measurement, "noise_std", 1.0)
    if not torch.is_tensor(noise_std):
        noise_std = torch.tensor(noise_std, device=device, dtype=dtype)
    noise_std = noise_std.to(device=device, dtype=dtype)

    while noise_std.ndim < prior.ndim:
        noise_std = noise_std.unsqueeze(0)
    noise_std_b = noise_std.expand_as(prior)

    X = _flatten_batch(prior)          # [B, D]
    Y = _flatten_batch(y_hat)          # [B, D]
    Z = _flatten_batch(z)              # [B, D]
    M = _flatten_batch(mask_b)[0]      # [D]
    Rstd = _flatten_batch(noise_std_b)[0]  # [D]

    x_mean = X.mean(dim=0, keepdim=True)   # [1, D]
    y_mean = Y.mean(dim=0, keepdim=True)   # [1, D]

    Xp = X - x_mean                        # [B, D]
    Yp = Y - y_mean                        # [B, D]

    if inflation != 1.0:
        Xp = Xp * inflation
        Yp = Yp * inflation

    local_obs_info = _build_local_obs_info(
        mask_flat_1d=M,
        state_shape=state_shape,
        loc_radius=loc_radius,
        same_channel_only=same_channel_only,
    )

    X_post = X.clone()
    I_B = torch.eye(B, device=device, dtype=dtype)

    for j in range(D):
        obs_idx, loc_w = local_obs_info[j]

        if obs_idx.numel() == 0:
            X_post[:, j] = X[:, j]
            continue

        Yp_loc = Yp[:, obs_idx]            # [B, m]
        y_mean_loc = y_mean[0, obs_idx]    # [m]
        z_loc = Z[0, obs_idx]              # [m]
        r_loc = Rstd[obs_idx]              # [m]

        valid = (
            torch.isfinite(z_loc)
            & torch.isfinite(r_loc)
            & (r_loc > 0)
            & torch.isfinite(loc_w)
            & (loc_w > 0)
        )

        if valid.sum() == 0:
            X_post[:, j] = X[:, j]
            continue

        Yp_loc = Yp_loc[:, valid]
        y_mean_loc = y_mean_loc[valid]
        z_loc = z_loc[valid]
        r_loc = r_loc[valid]
        loc_w = loc_w[valid]

        d = z_loc - y_mean_loc                              # [m]
        rinv_sqrt = torch.sqrt(loc_w) / torch.sqrt(r_loc * r_loc + eps)   # [m]

        S = Yp_loc * rinv_sqrt.unsqueeze(0)                 # [B, m]
        d_tilde = d * rinv_sqrt                             # [m]

        # LETKF transform-space posterior
        # Pa_tilde^{-1} = (B-1) I + S S^T
        Pa_inv = (B - 1) * I_B + (S @ S.t())
        Pa_inv = 0.5 * (Pa_inv + Pa_inv.t()) + eps * I_B

        rhs = S @ d_tilde
        w_mean = torch.linalg.solve(Pa_inv, rhs)            # [B]

        Pa = torch.linalg.solve(Pa_inv, I_B)
        Pa = 0.5 * (Pa + Pa.t())

        # symmetric square root of (B-1) Pa
        eigvals, eigvecs = torch.linalg.eigh((B - 1) * Pa)
        eigvals = torch.clamp(eigvals, min=0.0)
        W_a = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.t()   # [B, B]

        x_mean_j = x_mean[0, j]
        x_pert_j = Xp[:, j]                                  # [B]

        xa_mean_j = x_mean_j + torch.dot(x_pert_j, w_mean)
        xa_pert_j = W_a.t() @ x_pert_j

        X_post[:, j] = xa_mean_j + xa_pert_j

    return _unflatten_batch(X_post, prior.shape)


class LETKF(Method):
    name = "LETKF"

    def sample(
        self,
        cfg,
        model,
        prior,
        z,
        measurement,
        prior_mean=0,
        prior_std=1,
        device=None,
        path=None,
    ):
        posterior = letkf_update(
            prior=prior,
            z_one=z,
            measurement=measurement,
            inflation=cfg["method"].get("inflation", 1.0),
            eps=cfg["method"].get("eps", 1e-6),
            loc_radius=cfg["method"].get("loc_radius", 3.0),
            same_channel_only=cfg["method"].get("same_channel_only", False),
        )
        return posterior