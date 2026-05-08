import sys
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from .base import Method
from measurements.scheduler import a, score_scale_fn, guidance_scale_fn

def sampler(model, measurement, x_s, s, t, z, device, 
            g_min_scale=0.5, g_max_scale=2.0, g_power=1,
            s_min_scale=0.5, s_max_scale=1.5, s_power=1):
    
    # move to device
    x_s, s, t, z = x_s.to(device), s.to(device), t.to(device), z.to(device)

    # coefficient
    Sigma_s_t = measurement.Sigma(s, t)
    Sigma_s = measurement.Cov(s)
    a_t = a(t).view(-1, *([1] * (x_s.ndim - 1)))
    a_s = a(s).view(-1, *([1] * (x_s.ndim - 1)))

    # score estimation
    scale = guidance_scale_fn(s, g_min_scale, g_max_scale, g_power).view(-1, *([1] * (x_s.ndim - 1)))
    scale_s = score_scale_fn(s, s_min_scale, s_max_scale, s_power).view(-1, *([1] * (x_s.ndim - 1)))

    # prediction
    with torch.no_grad():
        predicted_x, predicted_h = model(x_s, s)
        predicted_h_s = a_s * predicted_x + (1 - a_s) * predicted_h

    # score
    score = -1 / (Sigma_s + 1e-5) * (x_s - predicted_h_s)

    # guidance
    guidance = measurement.guidance(z, x_s, model, t)
    guided_score = score * scale_s + guidance * scale

    # mean 
    mean = x_s + (a_t - a_s) * (predicted_x - predicted_h) - Sigma_s_t * guided_score 

    # noise 
    noise_term = torch.randn_like(x_s) * torch.sqrt(-Sigma_s_t)
    return mean + noise_term
                

class Nonlinear_Ours(Method):
    name = "ours"
    def train(self, cfg, device, model,
              optimizer, train_loader, val_loader,
              n_epoch, step, measurement, workdir):

        loss_ft = nn.MSELoss()
        global_step = 0

        # time configuration
        terminal_time = float(cfg.get("sample", {}).get("terminal_time", 1e-5))
        T = float(cfg.get("sample", {}).get("T", 0.992))

        pbar = tqdm(
            range(n_epoch),
            mininterval=5.0,
            maxinterval=50.0,
            leave=False,
            desc=f"train(step={step})",
            file=sys.stdout)

        for epoch in pbar:
            model.train()
            epoch_train_loss = 0.0
            epoch_train_loss_x = 0.0
            epoch_train_loss_h = 0.0

            for batch in train_loader:
                (x,) = batch
                x0 = x.to(device)

                # forward process
                t = torch.rand(x0.size(0), device=device) * (T - terminal_time) + terminal_time
                xt, _ = measurement.forward_process(x0, t)
                z = measurement.observation(x0)

                predicted_x, predicted_h = model(xt, t)

                # loss 
                loss_x = loss_ft(predicted_x, x0) # E[X_0|X_t]
                loss_h = loss_ft(predicted_h, z) # E[H(X_0)|X_t]
                loss = loss_x +  loss_h 

                # loss update
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += float(loss.item())
                epoch_train_loss_x += float(loss_x.item())
                epoch_train_loss_h += float(loss_h.item())
                global_step += 1

            epoch_train_loss /= max(len(train_loader), 1)
            epoch_train_loss_x /= max(len(train_loader), 1)
            epoch_train_loss_h /= max(len(train_loader), 1)

            pbar.set_postfix({
                "epoch": epoch + 1,
                "train": epoch_train_loss,
                "train_x": epoch_train_loss_x,
                "train_h": epoch_train_loss_h,
            }, refresh=False)


        return model


    def sample(self, cfg, model, x0, z, measurement, device=None, path=None):
        if device is None:
            device = (cfg.get("system") or {}).get("device", "cuda:0")

        # sample configuration
        nfe = ((cfg.get("sample") or {}).get("nfe")) or 500

        g_scale_min = cfg.get("sample", {}).get("g_scale_min")
        g_scale_max = cfg.get("sample", {}).get("g_scale_max")
        g_scale_power = cfg.get("sample", {}).get("g_scale_power")
        s_scale_min = cfg.get("sample", {}).get("s_scale_min")
        s_scale_max = cfg.get("sample", {}).get("s_scale_max")
        s_scale_power = cfg.get("sample", {}).get("s_scale_power")

        terminal_time = float(cfg.get("sample", {}).get("terminal_time", 1e-5))
        T = float(cfg.get("sample", {}).get("T", 0.992))

        model.eval()
        x0 = x0.to(device)
        z = z.to(device)

        # sample time
        times = torch.linspace(1, 0, nfe + 1, device=device, dtype=torch.float32) ** 2
        times = times * (T - terminal_time) + terminal_time


        t1 = torch.ones(x0.shape[0], device=device) * T
        x, _ = measurement.forward_process(x0, t1)

        for j in range(nfe):
            s = times[j] * t1
            t = times[j + 1] * t1

            x = sampler(model, measurement, x, s, t, z, device, 
                        g_scale_min, g_scale_max, g_scale_power,
                        s_scale_min, s_scale_max, s_scale_power)


        return x