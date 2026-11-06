import sys
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


def precond_x0(raw, predicted_h, x_t, t, measurement):
    a_t = measurement.a(t).view(-1, *([1] * (x_t.ndim - 1)))
    sigma_t = torch.sqrt(measurement.Cov(t).abs())
    q = torch.sqrt(a_t ** 2 + sigma_t ** 2).clamp_min(1e-5)
    return a_t * (x_t - (1 - a_t) * predicted_h) / q ** 2 + sigma_t * raw / q


def predict(model, measurement, x, t, target):
    raw, predicted_h = model(x, t)
    return (precond_x0(raw, predicted_h, x, t, measurement) if target == "x0_precond" else raw), predicted_h


def sampler(model, measurement, x_s, s, t, z, device, target="x0", solver="euler"):
    # move to device
    x_s, s, t, z = x_s.to(device), s.to(device), t.to(device), z.to(device)

    # coefficient
    Sigma_s = measurement.Cov(s)
    a_t = measurement.a(t).view(-1, *([1] * (x_s.ndim - 1)))
    a_s = measurement.a(s).view(-1, *([1] * (x_s.ndim - 1)))

    # prediction
    with torch.no_grad():
        predicted_x, predicted_h = predict(model, measurement, x_s, s, target)
        predicted_h_s = a_s * predicted_x + (1 - a_s) * predicted_h

    # score
    score = -1 / (Sigma_s + 1e-5) * (x_s - predicted_h_s)

    if solver == "exp":
        guided_score = score + measurement.guidance(z, x_s, model, s)
        M = a_t / a_s.clamp_min(1e-6)
        sigma_norm = measurement.sigma_norm()
        mean = M * x_s + (1 - M) * predicted_h + 2 * sigma_norm ** 2 * (M - 1) * guided_score
        return mean + sigma_norm * torch.sqrt((M ** 2 - 1).abs()) * torch.randn_like(x_s)

    Sigma_s_t = measurement.Sigma(s, t)
    guided_score = score + measurement.guidance(z, x_s, model, t)
    mean = x_s + (a_t - a_s) * (predicted_x - predicted_h) - Sigma_s_t * guided_score
    return mean + torch.sqrt(-Sigma_s_t) * torch.randn_like(x_s)


class Nonlinear_Ours:
    def train(self, cfg, device, model, optimizer, train_loader, val_loader, n_epoch, step, measurement):
        loss_ft = nn.MSELoss()
        target = cfg["method"].get("target", "x0")
        clean_h = cfg["method"].get("clean_h", False)

        # time configuration
        terminal_time = float(cfg["time"]["terminal_time"])
        T = float(cfg["time"]["train_T"])

        pbar = tqdm(range(n_epoch), mininterval=5.0, maxinterval=50.0, leave=False, desc=f"train(step={step})", file=sys.stdout)

        for epoch in pbar:
            model.train()
            epoch_train_loss = epoch_train_loss_x = epoch_train_loss_h = 0.0

            for (x,) in train_loader:
                x0 = x.to(device)

                # forward process
                t = torch.rand(x0.size(0), device=device) * (T - terminal_time) + terminal_time
                xt, _ = measurement.forward_process(x0, t)
                z = measurement.H(x0) if clean_h else measurement.observation(x0)

                predicted_x, predicted_h = predict(model, measurement, xt, t, target)

                # loss
                loss_x = loss_ft(predicted_x, x0)  # E[X_0|X_t]
                loss_h = loss_ft(predicted_h, z)  # E[H(X_0)|X_t]
                loss = loss_x + loss_h

                # loss update
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += float(loss.item())
                epoch_train_loss_x += float(loss_x.item())
                epoch_train_loss_h += float(loss_h.item())

            n = max(len(train_loader), 1)
            pbar.set_postfix({"epoch": epoch + 1, "train": epoch_train_loss / n, "train_x": epoch_train_loss_x / n, "train_h": epoch_train_loss_h / n}, refresh=False)

        return model

    def sample(self, cfg, model, x0, z, measurement):
        device, sc = cfg["system"]["device"], cfg["sample"]
        nfe, target = sc["nfe"], cfg["method"].get("target", "x0")

        terminal_time = float(cfg["time"]["terminal_time"])
        T = float(cfg["time"]["sample_T"])

        model.eval()
        x0 = x0.to(device)
        z = z.to(device)

        u = torch.linspace(1, 0, nfe + 1, device=device, dtype=torch.float32)
        times = u ** 2 * (T - terminal_time) + terminal_time

        t1 = torch.ones(x0.shape[0], device=device) * T
        x, _ = measurement.forward_process(x0, t1)

        for j in range(nfe):
            s = times[j].expand(x0.shape[0])
            t = times[j + 1].expand(x0.shape[0])

            x = sampler(model, measurement, x, s, t, z, device, target, sc.get("solver", "euler"))

        m, f = x.mean(0, keepdim=True), sc.get("inflation", 1.0)
        return x if f == 1 else m + f * (x - m)
