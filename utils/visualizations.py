import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as tvu
import imageio.v2 as iio

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm


# =========================================================
# 0. Robust quantile helpers
# =========================================================
@torch.no_grad()
def _safe_flat_abs(x: torch.Tensor) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().abs().reshape(-1)


@torch.no_grad()
def _robust_abs_quantile_multi(
    tensors: list[Optional[torch.Tensor]],
    q: float = 0.995,
    max_sample_size_each: int = 1_000_000,
    force_cpu: bool = True,
) -> float:
    """
    Safe robust quantile over multiple tensors without concatenating gigantic vectors.
    """
    pools = []

    for x in tensors:
        if x is None:
            continue

        a = _safe_flat_abs(x)
        if a is None or a.numel() == 0:
            continue

        n = a.numel()
        if n > max_sample_size_each:
            step = int(np.ceil(n / max_sample_size_each))
            a = a[::step]

        if force_cpu and a.device.type != "cpu":
            a = a.cpu()

        pools.append(a)

    if len(pools) == 0:
        return 1.0

    a = torch.cat(pools, dim=0)
    val = torch.quantile(a, q).item()
    return max(float(val), 1e-8)


# =========================================================
# 1. Vorticity
# =========================================================
@torch.no_grad()
def vorticity_from_uv(
    x: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    periodic: bool = True,
) -> torch.Tensor:
    """
    x: [B, 2, H, W]
    return: [B, H, W], omega = du/dy - dv/dx

    NOTE:
    This keeps your original implementation convention:
      du = gradient(u, x-direction)
      dv = gradient(v, y-direction)
      omega = du - dv
    """
    assert x.ndim == 4 and x.size(1) == 2, f"Expected [B,2,H,W], got {tuple(x.shape)}"

    *batch, _, h, w = x.shape
    y = x.reshape(-1, 2, h, w)

    if periodic:
        y = F.pad(y, pad=(1, 1, 1, 1), mode="circular")
    else:
        y = F.pad(y, pad=(1, 1, 1, 1), mode="replicate")

    (du,) = torch.gradient(y[:, 0], spacing=(dx,), dim=-1)
    (dv,) = torch.gradient(y[:, 1], spacing=(dy,), dim=-2)

    omega = du - dv
    omega = omega[:, 1:-1, 1:-1]
    omega = omega.reshape(*batch, h, w)

    return omega


vorticity_2d = vorticity_from_uv


# =========================================================
# 2. Scalar -> RGB
# =========================================================
@torch.no_grad()
def colorize_scalar(
    w: torch.Tensor,
    cmap: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> torch.Tensor:
    """
    [B, H, W] -> [B, 3, H, W]
    """
    assert w.dim() == 3, f"Expected [B,H,W], got {tuple(w.shape)}"

    if vmin is None or vmax is None:
        vmax = float(w.detach().abs().max().item())
        vmax = max(vmax, 1e-8)
        vmin = -vmax

    wn = (w - vmin) / (vmax - vmin + 1e-8)
    wn = wn.clamp(0, 1)

    cmap_fn = cm.get_cmap(cmap)
    rgb = cmap_fn(wn.detach().cpu().numpy())[..., :3]  # [B, H, W, 3]
    rgb = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()

    return rgb


# =========================================================
# 3. PNG save helper
# =========================================================
@torch.no_grad()
def save_vorticity_pairs_color(
    gt_uv: torch.Tensor,
    pred_uv: torch.Tensor,
    png_path: str,
    denorm=None,
    noisy_uv: Optional[torch.Tensor] = None,
    cmap: str = "turbo",
    periodic: bool = True,
    dx: float = 1.0,
    dy: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    robust: bool = True,
    q: float = 0.995,
    quantile_max_sample_size_each: int = 1_000_000,
):
    """
    Save vorticity comparison image.

    If noisy_uv is None:
        saves [GT | Pred]

    If noisy_uv is given:
        saves [GT | Measurement | Pred]
    """
    if denorm is not None:
        gt_uv = denorm(gt_uv)
        pred_uv = denorm(pred_uv)

        if noisy_uv is not None:
            noisy_uv = denorm(noisy_uv)

    w_gt = vorticity_from_uv(gt_uv, dx=dx, dy=dy, periodic=periodic)
    w_pred = vorticity_from_uv(pred_uv, dx=dx, dy=dy, periodic=periodic)

    if noisy_uv is None:
        w_noisy = None
    else:
        w_noisy = vorticity_from_uv(noisy_uv, dx=dx, dy=dy, periodic=periodic)

    if vmin is None or vmax is None:
        if robust:
            vmax = _robust_abs_quantile_multi(
                [w_gt, w_pred, w_noisy],
                q=q,
                max_sample_size_each=quantile_max_sample_size_each,
                force_cpu=True,
            )
            vmin = -vmax
        else:
            vals_min = [w_gt.min().item(), w_pred.min().item()]
            vals_max = [w_gt.max().item(), w_pred.max().item()]

            if w_noisy is not None:
                vals_min.append(w_noisy.min().item())
                vals_max.append(w_noisy.max().item())

            vmin = float(min(vals_min))
            vmax = float(max(vals_max))

            if abs(vmax - vmin) < 1e-8:
                vmax = vmin + 1e-8

    w_gt_rgb = colorize_scalar(w_gt, cmap=cmap, vmin=vmin, vmax=vmax)
    w_pred_rgb = colorize_scalar(w_pred, cmap=cmap, vmin=vmin, vmax=vmax)

    if w_noisy is None:
        tiles = torch.stack([w_gt_rgb, w_pred_rgb], dim=1).flatten(0, 1)
        nrow = 2
    else:
        w_noisy_rgb = colorize_scalar(w_noisy, cmap=cmap, vmin=vmin, vmax=vmax)
        tiles = torch.stack([w_gt_rgb, w_noisy_rgb, w_pred_rgb], dim=1).flatten(0, 1)
        nrow = 3

    dirname = os.path.dirname(png_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    tvu.save_image(tiles.cpu(), png_path, nrow=nrow, padding=0)


# =========================================================
# 4. Video helpers
# =========================================================
def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()

    try:
        buf = fig.canvas.tostring_rgb()
        w, h = fig.canvas.get_width_height()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    except AttributeError:
        arr = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]

    return arr.copy()


def _canon_seq(
    x: Optional[torch.Tensor],
    batch_index: int = 0,
) -> Optional[torch.Tensor]:
    """
    Convert input sequence to [T, 2, H, W].

    Accepted:
      [T, 2, H, W]
      [T, B, 2, H, W]
      [B, T, 2, H, W]
      [2, H, W]
    """
    if x is None:
        return None

    if x.ndim == 4 and x.size(0) == 2:
        x = x.unsqueeze(0)
    elif x.ndim == 5 and x.size(2) == 2:
        x = x[:, batch_index]
    elif x.ndim == 5 and x.size(1) == 2:
        x = x[batch_index]
    elif not (x.ndim == 4 and x.size(1) == 2):
        raise ValueError(f"Unsupported shape for sequence: {tuple(x.shape)}")

    return x


def _omega_seq(
    x_seq: torch.Tensor,
    Lx: float,
    Ly: float,
) -> np.ndarray:
    with torch.no_grad():
        omega = vorticity_2d(
            x_seq,
            dx=Lx / x_seq.shape[-1],
            dy=Ly / x_seq.shape[-2],
        )
        return omega.detach().cpu().numpy()


def _sym_vrange(
    arr_list: list[np.ndarray],
    robust_pct: Optional[float],
):
    if robust_pct is None or robust_pct >= 100:
        amax = max(float(np.max(np.abs(a))) for a in arr_list)
    else:
        amax = max(float(np.percentile(np.abs(a), robust_pct)) for a in arr_list)

    amax = max(amax, 1e-8)
    return -amax, +amax


def _draw_triplet_frame(
    omegas: list[np.ndarray],
    uvs: list[Optional[torch.Tensor]],
    names: list[str],
    Lx: float,
    Ly: float,
    cmap: str,
    vmin: float,
    vmax: float,
    quiver: bool,
    quiver_step: int,
    quiver_scale: float,
    dpi: int = 140,
) -> np.ndarray:
    ncol = len(omegas)
    extent = [0, Lx, 0, Ly]

    fig = plt.figure(
        figsize=(3.0 * ncol + 0.35, 3.0),
        dpi=dpi,
        constrained_layout=False,
    )

    gs = fig.add_gridspec(
        1,
        ncol + 1,
        width_ratios=[1] * ncol + [0.05],
        wspace=0.0,
        hspace=0.0,
    )

    im_axes = [fig.add_subplot(gs[0, i]) for i in range(ncol)]
    cax = fig.add_subplot(gs[0, -1])

    im_last = None

    for i, ax in enumerate(im_axes):
        om = omegas[i]

        im = ax.imshow(
            om,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
            interpolation="nearest",
        )
        im_last = im

        ax.set_axis_off()

        txt = ax.text(
            0.02,
            0.98,
            names[i],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="white",
        )
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="black")])

        if quiver and uvs[i] is not None:
            uv = uvs[i].detach().cpu()
            u, v = uv[0], uv[1]

            H, W = om.shape
            xs = np.linspace(0, Lx, W, endpoint=False)
            ys = np.linspace(0, Ly, H, endpoint=False)
            XX, YY = np.meshgrid(xs, ys, indexing="xy")

            step = max(1, quiver_step)

            ax.quiver(
                XX[::step, ::step],
                YY[::step, ::step],
                u.numpy()[::step, ::step],
                v.numpy()[::step, ::step],
                pivot="mid",
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                color="k",
                alpha=0.55,
            )

    cb = fig.colorbar(im_last, cax=cax, orientation="vertical")
    cb.set_label(r"vorticity $\omega$")

    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace=0,
        hspace=0,
    )

    frame = _fig_to_rgb(fig)
    plt.close(fig)

    return frame


def save_vorticity_video_pairs(
    x_seq: torch.Tensor,
    y_seq: torch.Tensor,
    out_path: str,
    noisy_uv: Optional[torch.Tensor] = None,
    batch_index: int = 0,
    Lx: float = 2 * np.pi,
    Ly: float = 2 * np.pi,
    cmap: str = "RdBu_r",
    fps: int = 12,
    dpi: int = 140,
    quiver: bool = False,
    quiver_step: int = 6,
    quiver_scale: float = 1.0,
    max_frames: Optional[int] = None,
    norm_scope: str = "frame_robust",
    robust_pct: float = 99.0,
    crf: int = 28,
    preset: str = "slow",
):
    """
    Save vorticity video.

    x_seq: GT sequence
    y_seq: posterior/pred sequence
    noisy_uv: optional measurement/noisy sequence

    Supported extensions:
      .webm / .mp4 / .gif
    """
    x_seq = _canon_seq(x_seq, batch_index)
    y_seq = _canon_seq(y_seq, batch_index)
    z_seq = _canon_seq(noisy_uv, batch_index) if noisy_uv is not None else None

    assert x_seq.shape == y_seq.shape, "x_seq and y_seq must have the same [T,2,H,W] shape."

    if z_seq is not None and z_seq.shape != x_seq.shape:
        raise ValueError("noisy_uv sequence shape differs from x_seq.")

    T = x_seq.size(0)

    if max_frames is not None:
        T = min(T, max_frames)

    omega_x = _omega_seq(x_seq[:T], Lx, Ly)
    omega_y = _omega_seq(y_seq[:T], Lx, Ly)
    omega_z = _omega_seq(z_seq[:T], Lx, Ly) if z_seq is not None else None

    dirname = os.path.dirname(out_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    ext = os.path.splitext(out_path)[1].lower()

    if ext == ".webm":
        writer = iio.get_writer(
            out_path,
            fps=fps,
            codec="libvpx-vp9",
            ffmpeg_params=[
                "-b:v",
                "0",
                "-crf",
                str(crf),
                "-row-mt",
                "1",
            ],
        )
    elif ext == ".mp4":
        writer = iio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            ffmpeg_params=[
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-pix_fmt",
                "yuv444p",
            ],
            macro_block_size=None,
        )
    elif ext == ".gif":
        writer = iio.get_writer(
            out_path,
            mode="I",
            duration=1.0 / max(fps, 1),
        )
    else:
        raise ValueError("Supported extensions: .webm / .mp4 / .gif")

    names = ["GT", "Posterior"] + (["Measurement"] if omega_z is not None else [])

    if norm_scope == "global":
        vmin_global, vmax_global = _sym_vrange(
            [omega_x, omega_y] + ([omega_z] if omega_z is not None else []),
            robust_pct=None,
        )
    elif norm_scope == "global_robust":
        vmin_global, vmax_global = _sym_vrange(
            [omega_x, omega_y] + ([omega_z] if omega_z is not None else []),
            robust_pct=robust_pct,
        )
    elif norm_scope not in ("frame", "frame_robust"):
        raise ValueError(
            "norm_scope must be one of: global, global_robust, frame, frame_robust"
        )

    try:
        for t in range(T):
            if norm_scope in ("frame", "frame_robust"):
                arrays = [omega_x[t], omega_y[t]] + (
                    [omega_z[t]] if omega_z is not None else []
                )
                vmin, vmax = _sym_vrange(
                    arrays,
                    robust_pct if norm_scope.endswith("robust") else None,
                )
            else:
                vmin, vmax = vmin_global, vmax_global

            omegas = [omega_x[t], omega_y[t]] + (
                [omega_z[t]] if omega_z is not None else []
            )
            uvs = [x_seq[t], y_seq[t]] + (
                [z_seq[t]] if z_seq is not None else []
            )

            frame = _draw_triplet_frame(
                omegas=omegas,
                uvs=uvs,
                names=names,
                Lx=Lx,
                Ly=Ly,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                quiver=quiver,
                quiver_step=quiver_step,
                quiver_scale=quiver_scale,
                dpi=dpi,
            )

            writer.append_data(frame)

    finally:
        writer.close()

    print(
        f"[saved] {out_path} "
        f"(frames={T}, fps={fps}, norm={norm_scope}, robust={robust_pct})"
    )


# =========================================================
# 5. RMSE plot helper
# =========================================================
def plot_rmse_with_measurements(
    rmse_overall,
    rmse_curve,
    total_step,
    steps,
    dt,
    name: str = "RMSE",
    save_path: Optional[str] = None,
):
    total_step = np.asarray(total_step)
    rmse_curve = np.asarray(rmse_curve)
    steps = np.asarray(steps)

    time = total_step * dt
    meas_mask = np.isin(total_step, steps)

    plt.figure()

    plt.plot(
        time,
        rmse_curve,
        label="RMSE",
        linestyle="--",
        alpha=0.5,
    )
    plt.scatter(
        time,
        rmse_curve,
        marker="x",
        label="step",
    )
    plt.scatter(
        time[meas_mask],
        rmse_curve[meas_mask],
        marker="o",
        label="measurement step",
    )

    plt.xlabel("time")
    plt.ylabel("RMSE")
    plt.title(f"{name} (mean RMSE={rmse_overall:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()