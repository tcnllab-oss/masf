import os
import math
import numpy as np
import torch
import seaborn as sns

from .visualizations import (
    save_vorticity_pairs_color,
    save_vorticity_video_pairs,
    plot_rmse_with_measurements)


@torch.no_grad()
def _safe_quantile_large(
    x: torch.Tensor,
    q: float,
    max_sample_size: int = 2_000_000,
    force_cpu: bool = True,
) -> float:
    """
    Safe quantile for very large tensors.

    Steps:
      1. flatten
      2. subsample if too large
      3. optionally move to CPU
      4. compute quantile in float32
    """
    if x is None:
        raise ValueError("x is None")

    x = x.detach().reshape(-1)

    if x.numel() == 0:
        return 0.0

    n = x.numel()
    if n > max_sample_size:
        step = int(np.ceil(n / max_sample_size))
        x = x[::step]

    if force_cpu and x.device.type != "cpu":
        x = x.cpu()

    return float(torch.quantile(x.float(), q).item())


@torch.no_grad()
def _compute_csi(
    truth: torch.Tensor,
    pred: torch.Tensor,
    threshold: float = None,
    quantile_q: float = 0.95,
    max_quantile_sample_size: int = 2_000_000,
):
    """
    Critical Success Index (CSI)

    Default:
      - if threshold is None:
          use quantile_q percentile of truth as event threshold

    event := value >= threshold
    CSI = TP / (TP + FP + FN)

    Returns:
        float
    """
    truth_flat = truth.detach().reshape(-1).float()
    pred_flat = pred.detach().reshape(-1).float()

    if threshold is None:
        threshold = _safe_quantile_large(
            truth_flat,
            q=quantile_q,
            max_sample_size=max_quantile_sample_size,
            force_cpu=True,
        )
    else:
        threshold = float(threshold)

    truth_event = truth_flat >= threshold
    pred_event = pred_flat >= threshold

    tp = torch.logical_and(truth_event, pred_event).sum().item()
    fp = torch.logical_and(~truth_event, pred_event).sum().item()
    fn = torch.logical_and(truth_event, ~pred_event).sum().item()

    denom = tp + fp + fn
    if denom == 0:
        return float("nan")

    return float(tp / denom)


def finalize_and_save(
    cfg,
    workdir,
    dynamics,
    measurement,
    trajectory,
    observations,
    total_observations,
    assimilated_states,
    steps,
    total_step,
    dynamic_type,
):
    gap = (cfg.get("steps") or {}).get("gap") if isinstance(cfg, dict) else None
    pred_trajectory = assimilated_states.mean(dim=1)

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------
    save_vorticity_pairs_color(
        trajectory,
        pred_trajectory,
        os.path.join(workdir, "last.png"),
        cmap=sns.cm.icefire,
        noisy_uv=observations,
        periodic=True,
        quantile_max_sample_size_each=300_000,
    )

    save_vorticity_video_pairs(
        x_seq=trajectory,
        y_seq=pred_trajectory,
        out_path=os.path.join(workdir, "vorticity_pairs.gif"),
        norm_scope="frame_robust",
        robust_pct=99.5,
        cmap=sns.cm.icefire,
        fps=4,
        crf=32,
        noisy_uv=observations,
        preset="slow",
        dpi=140,
    )


    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    err = trajectory - pred_trajectory
    rmse_overall = torch.sqrt((err ** 2).mean()).item()

    T = err.shape[0]
    mse_t = (err ** 2).reshape(T, -1).mean(dim=1)
    rmse_curve = torch.sqrt(mse_t).detach().cpu().numpy()

    metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg, dict) else {}
    csi_threshold = metrics_cfg.get("csi_threshold", None)
    csi_overall = _compute_csi(
        trajectory,
        pred_trajectory,
        threshold=csi_threshold,
        quantile_q=0.95,
        max_quantile_sample_size=2_000_000,
    )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    fname = "results.npz"
    save_path = os.path.join(workdir, fname)

    np.savez_compressed(
        save_path,
        states=trajectory.detach().cpu().numpy().astype(np.float32),
        observations=observations.detach().cpu().numpy().astype(np.float32),
        total_observations=total_observations.detach().cpu().numpy().astype(np.float32),
        pred_trajectory=pred_trajectory.detach().cpu().numpy().astype(np.float32),
        steps=steps,
        total_step=total_step,
        rmse_overall=np.float32(rmse_overall),
        csi_overall=np.float32(csi_overall) if not math.isnan(csi_overall) else np.nan,
        rmse_curve=rmse_curve.astype(np.float32),
    )

    # RMSE curve plot
    rmse_fig_path = os.path.join(workdir, f"RMSE_mean{rmse_overall:.2f}.png")
    plot_rmse_with_measurements(
        rmse_overall,
        rmse_curve,
        total_step,
        steps,
        dynamics.dt,
        name=f"{dynamic_type}-gap{gap}",
        save_path=rmse_fig_path,
    )

    return rmse_overall, csi_overall