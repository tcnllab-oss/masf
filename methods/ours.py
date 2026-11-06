import sys
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


def precond_terms(x_t, t, measurement):
    a_t, sigma_t = measurement.A(t), torch.sqrt(measurement.Cov(t).abs())
    q = torch.sqrt(a_t ** 2 + sigma_t ** 2).clamp_min(1e-5)
    return a_t * (x_t - measurement.B(t)) / q ** 2, (sigma_t / q).clamp_min(1e-5)


def predict_x0(model, x_t, t, measurement, target):
    raw = model(x_t, t)
    if target != "x0_precond":
        return raw
    skip, out = precond_terms(x_t, t, measurement)
    return skip + out * raw


@torch.no_grad()
def sampler(model, measurement, x_s, s, t, z, device, target="x0"):

    # move to device
    x_s, s, t, z = x_s.to(device), s.to(device), t.to(device), z.to(device)

    # prediction
    predicted_x0 = predict_x0(model, x_s, s, measurement, target)

    # coefficients
    U_s_t = measurement.U(s, t)
    Sigma_s = torch.abs(measurement.Cov(s))
    a_x = measurement.A_apply(predicted_x0, s)

    # score
    B_s = measurement.B(s)
    score = -(x_s - a_x - B_s) / (Sigma_s + 1e-5)

    # guidance
    guidance = measurement.guidance(z, x_s, s)
    guided_score = score + guidance
    score_mean = measurement.Sigma_apply(guided_score, s, t)

    # mean
    mean = measurement.M_apply(x_s, s, t) + U_s_t - score_mean

    # noise
    noise = torch.randn_like(x_s)
    noise_term = measurement.Sigma_root_apply(noise, s, t)

    return mean + noise_term


class Ours:
    def train(self, cfg, device, model, optimizer, train_loader, val_loader, n_epoch, step, measurement):

        loss_ft = nn.MSELoss()
        global_step = 0
        target = cfg["method"].get("target", "x0")

        # time configuration
        terminal_time = float(cfg["time"]["terminal_time"])
        T = float(cfg["time"]["train_T"])

        pbar = tqdm(range(n_epoch), mininterval=5.0, maxinterval=50.0, leave=False, desc=f"train(step={step})", file=sys.stdout)

        for epoch in pbar:
            model.train()
            epoch_train_loss = 0.0

            for batch in train_loader:
                (x,) = batch
                x0 = x.to(device)

                # forward process
                t = torch.rand(x0.size(0), device=device) * (T - terminal_time) + terminal_time
                xt, _ = measurement.forward_process(x0, t)

                predicted = model(xt, t)

                if target == "x0_precond":
                    skip, out = precond_terms(xt, t, measurement)
                    loss = loss_ft(predicted, (x0 - skip) / out)
                else:
                    loss = loss_ft(predicted, x0)

                # loss update
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += float(loss.item())
                global_step += 1

            epoch_train_loss /= max(len(train_loader), 1)

            pbar.set_postfix({"epoch": epoch + 1, "train": epoch_train_loss}, refresh=False)

        return model

    @torch.no_grad()
    def sample(self, cfg, model, x0, z, measurement):
        device, nfe = cfg["system"]["device"], cfg["sample"]["nfe"]

        terminal_time = float(cfg["time"]["terminal_time"])
        T = float(cfg["time"]["sample_T"])

        model.eval()
        x0 = x0.to(device)
        z = z.to(device)

        # sample time
        times = torch.linspace(1, 0, nfe + 1, device=device, dtype=torch.float32) ** 2
        times = times * (T - terminal_time) + terminal_time

        t1 = torch.ones(x0.shape[0], device=device) * T
        x, _ = measurement.forward_process(x0, t1)

        for j in range(nfe):
            s = times[j].expand(x0.shape[0])
            t = times[j + 1].expand(x0.shape[0])

            x = sampler(model, measurement, x, s, t, z, device, cfg["method"].get("target", "x0"))

        return x