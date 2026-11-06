import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as tvu
import imageio.v2 as iio
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm


# Robust quantile helpers
@torch.no_grad()
def _robust_abs_quantile(tensors, q, max_sample_size_each):
    pools = []
    for x in tensors:
        a = x.detach().abs().reshape(-1)
        pools.append(a[::int(np.ceil(a.numel() / max_sample_size_each))].cpu())
    return max(float(torch.quantile(torch.cat(pools), q)), 1e-8)


# Vorticity
@torch.no_grad()
def vorticity_from_uv(x, dx=1.0, dy=1.0):
    # [B, 2, H, W] -> [B, H, W], periodic
    y = F.pad(x, pad=(1, 1, 1, 1), mode="circular")
    (du,) = torch.gradient(y[:, 0], spacing=(dx,), dim=-1)
    (dv,) = torch.gradient(y[:, 1], spacing=(dy,), dim=-2)
    return (du - dv)[:, 1:-1, 1:-1]


# Scalar -> RGB
@torch.no_grad()
def colorize_scalar(w, cmap, vmin, vmax):
    # [B, H, W] -> [B, 3, H, W]
    wn = ((w - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1)
    return torch.from_numpy(cmap(wn.cpu().numpy())[..., :3]).permute(0, 3, 1, 2).contiguous()


# PNG save helper
@torch.no_grad()
def save_vorticity_pairs_color(gt_uv, pred_uv, noisy_uv, png_path, cmap, q=0.995, max_sample_size_each=300_000):
    # rows of [GT | Measurement | Pred]
    ws = [vorticity_from_uv(x) for x in (gt_uv, noisy_uv, pred_uv)]
    vmax = _robust_abs_quantile(ws, q, max_sample_size_each)
    tiles = torch.stack([colorize_scalar(w, cmap, -vmax, vmax) for w in ws], dim=1).flatten(0, 1)
    tvu.save_image(tiles, png_path, nrow=3, padding=0)


# Video helpers
def _fig_to_rgb(fig):
    fig.canvas.draw()
    try:
        w, h = fig.canvas.get_width_height()
        return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3).copy()
    except AttributeError:
        return np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()


def _draw_frame(omegas, names, L, cmap, vmax, dpi):
    ncol = len(omegas)
    fig = plt.figure(figsize=(3.0 * ncol + 0.35, 3.0), dpi=dpi)
    gs = fig.add_gridspec(1, ncol + 1, width_ratios=[1] * ncol + [0.05], wspace=0.0, hspace=0.0)
    for i, om in enumerate(omegas):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(om, origin="lower", extent=[0, L, 0, L], cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), interpolation="nearest")
        ax.set_axis_off()
        ax.text(0.02, 0.98, names[i], transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold", color="white").set_path_effects([pe.withStroke(linewidth=3, foreground="black")])
    fig.colorbar(im, cax=fig.add_subplot(gs[0, -1]), orientation="vertical").set_label(r"vorticity $\omega$")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    frame = _fig_to_rgb(fig)
    plt.close(fig)
    return frame


@torch.no_grad()
def save_vorticity_video(x_seq, y_seq, z_seq, out_path, cmap, fps=4, dpi=140, robust_pct=99.5, L=2 * np.pi):
    omegas = [vorticity_from_uv(s, dx=L / s.shape[-1], dy=L / s.shape[-2]).cpu().numpy() for s in (x_seq, y_seq, z_seq)]
    writer = iio.get_writer(out_path, mode="I", duration=1.0 / fps)
    try:
        for t in range(x_seq.shape[0]):
            frame_omegas = [o[t] for o in omegas]
            vmax = max(max(float(np.percentile(np.abs(o), robust_pct)) for o in frame_omegas), 1e-8)
            writer.append_data(_draw_frame(frame_omegas, ["GT", "Posterior", "Measurement"], L, cmap, vmax, dpi))
    finally:
        writer.close()
    print(f"[saved] {out_path} (frames={x_seq.shape[0]}, fps={fps})")


# 1D plots
def plot_line_1d(gt, post, obs, b, path):
    plt.figure()
    for x, marker, ls, label in zip((gt, post, obs), "os^", ("-", "--", ":"), ("GT", "Posterior", "Measurement")):
        plt.plot(x[b].reshape(-1), marker=marker, linestyle=ls, label=label)
    plt.xlabel("dimension (index)")
    plt.ylabel("value")
    plt.title(f"Time {b} 1D Line")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_triplet(gt, post, obs, time, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, x, title in zip(axes, (gt, post, obs), ("Ground Truth", "Posterior", "Measurement")):
        ax.imshow(x[:, 0].T, aspect="auto", origin="lower", cmap="winter", extent=(time[0], time[-1], 0, x.shape[-1] - 1))
        ax.set_title(title)
        ax.set_xlabel("time")
    axes[0].set_ylabel("grid index")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# RMSE plot helper
def plot_rmse_with_measurements(rmse_curve, total_step, steps, dt, name, save_path):
    time, meas_mask = np.asarray(total_step) * dt, np.isin(total_step, steps)
    plt.figure()
    plt.plot(time, rmse_curve, label="RMSE", linestyle="--", alpha=0.5)
    plt.scatter(time, rmse_curve, marker="x", label="step")
    plt.scatter(time[meas_mask], rmse_curve[meas_mask], marker="o", label="measurement step")
    plt.xlabel("time")
    plt.ylabel("RMSE")
    plt.title(name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
