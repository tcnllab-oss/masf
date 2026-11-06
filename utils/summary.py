import os
import numpy as np
import torch
import seaborn as sns

from .visualizations import save_vorticity_pairs_color, save_vorticity_video, plot_line_1d, plot_triplet, plot_rmse_with_measurements


@torch.no_grad()
def _compute_csi(truth, pred, q=0.95, max_sample_size=2_000_000):
    truth, pred = truth.detach().reshape(-1).float(), pred.detach().reshape(-1).float()
    sub = truth[::int(np.ceil(truth.numel() / max_sample_size))]
    threshold = float(torch.quantile(sub.cpu(), q))
    t_ev, p_ev = truth >= threshold, pred >= threshold
    tp, fp, fn = (t_ev & p_ev).sum().item(), (~t_ev & p_ev).sum().item(), (t_ev & ~p_ev).sum().item()
    return float("nan") if tp + fp + fn == 0 else tp / (tp + fp + fn)


@torch.no_grad()
def _ensemble_metrics(truth, members):
    n, se, var, crps = members.shape[1], 0.0, 0.0, 0.0
    w = 2 * torch.arange(1, n + 1, device=members.device, dtype=torch.float64) - n - 1
    for x, y in zip(members, truth):
        x, y = x.double().flatten(1), y.double().flatten()
        se += ((x.mean(0) - y) ** 2).sum().item()
        var += x.var(0).sum().item()
        crps += ((x - y).abs().mean(0) - w @ x.sort(0).values / (n * (n - 1))).sum().item()
    return np.sqrt(se / truth.numel()), crps / truth.numel(), np.sqrt((n + 1) / n * var / se)


def finalize_and_save(cfg, workdir, dynamics, trajectory, observations, total_observations, assimilated_states, steps, total_step):
    pred_trajectory = assimilated_states.mean(dim=1)

    # Visualization
    if trajectory.ndim == 4 and trajectory.shape[1] == 2:  # 2D velocity fields only
        save_vorticity_pairs_color(trajectory, pred_trajectory, observations, os.path.join(workdir, "last.png"), cmap=sns.cm.icefire)
        save_vorticity_video(trajectory, pred_trajectory, observations, os.path.join(workdir, "vorticity_pairs.gif"), cmap=sns.cm.icefire)
    elif trajectory.ndim == 3:  # 1D states
        gt, post, obs, T = trajectory.cpu().numpy(), pred_trajectory.cpu().numpy(), observations.cpu().numpy(), trajectory.shape[0]
        for b in (0, T // 2, T - 1):
            plot_line_1d(gt, post, obs, b, os.path.join(workdir, f"total_{b}_variable.png"))
        plot_triplet(gt, post, obs, np.asarray(total_step) * dynamics.dt, os.path.join(workdir, "total.png"))

    # Metrics
    err = trajectory - pred_trajectory
    rmse_overall = torch.sqrt((err ** 2).mean()).item()

    T = err.shape[0]
    mse_t = (err ** 2).reshape(T, -1).mean(dim=1)
    rmse_curve = torch.sqrt(mse_t).detach().cpu().numpy()

    csi_overall = _compute_csi(trajectory, pred_trajectory)

    # Ensemble RMSE, fair CRPS, SSR
    m = np.isin(total_step, steps) if cfg.get("eval", "all") == "measurement" else np.ones(len(total_step), dtype=bool)
    m = torch.as_tensor(m, device=trajectory.device)
    rmse, fcrps, ssr = _ensemble_metrics(trajectory[m], assimilated_states[m])
    print(f"RMSE {rmse} FCRPS {fcrps} SSR {ssr}")
    ensemble = {"assimilated_states": assimilated_states.detach().cpu().numpy().astype(np.float32)} if trajectory.ndim == 3 else {}

    # Save results
    np.savez_compressed(os.path.join(workdir, "results.npz"), states=trajectory.detach().cpu().numpy().astype(np.float32),
                        observations=observations.detach().cpu().numpy().astype(np.float32),
                        total_observations=total_observations.detach().cpu().numpy().astype(np.float32),
                        pred_trajectory=pred_trajectory.detach().cpu().numpy().astype(np.float32), steps=steps, total_step=total_step,
                        rmse_overall=np.float32(rmse_overall), csi_overall=np.float32(csi_overall),
                        rmse_curve=rmse_curve.astype(np.float32), rmse=rmse, fcrps=fcrps, ssr=ssr, **ensemble)

    # RMSE curve plot
    rmse_fig_path = os.path.join(workdir, f"RMSE_mean{rmse_overall:.2f}.png")
    plot_rmse_with_measurements(rmse_curve, total_step, steps, dynamics.dt, name=f"{cfg['dynamics']['type']}{cfg['dynamics']['dim']}-gap{cfg['steps']['gap']}", save_path=rmse_fig_path)

    return rmse, csi_overall