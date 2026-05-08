import sys
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from .base import Method


@torch.no_grad()
def _reshape_like_std(x, v):
    if x.dim() == 4:
        return v[:, None, None, None]
    elif x.dim() == 3:
        return v[:, None, None]
    else:
        while v.dim() < x.dim():
            v = v.unsqueeze(-1)
        return v

def marginal_prob_std(t, sigma =25.0):
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma = 25.0):
    return sigma**t


def loss_fn(model, x, eps=1e-5):
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps #(eps,1)
    z = torch.randn_like(x)
    
    if x.dim() == 4:
        std = marginal_prob_std(random_t)[:, None, None, None ]
    elif x.dim() == 3:
        std = marginal_prob_std(random_t)[:, None, None]
    
    perturbed_x = x + z * std 
    score = model(perturbed_x, random_t)

    loss = torch.mean(torch.sum((score  + z)**2, dim=1))
    return loss



class SF(Method):
    name = "sf"
    def train(self, cfg, device, model,
            optimizer, train_loader, val_loader,
            n_epoch, step, measurement, workdir):

        global_step = 0
        pbar = tqdm(
            range(n_epoch),
            mininterval=5.0,
            maxinterval=50.0,
            leave=False,
            desc=f"train(step={step})",
            file=sys.stdout,
        )

        for epoch in pbar:
            model.train()
            epoch_train_loss = 0.0

            for batch in train_loader:
                (x,) = batch
                x0 = x.to(device)
                
                # loss 
                loss = loss_fn(model, x0)

                # loss update
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += float(loss.item())
                global_step += 1


            epoch_train_loss /= max(len(train_loader), 1)

            pbar.set_postfix({
                "epoch": epoch + 1,
                "train": epoch_train_loss}, refresh=False)
            
        return model

    @torch.no_grad()
    def sample(self, cfg, model, x0, z, measurement, device=None, path=None):
        if device is None:
            device = (cfg.get("system") or {}).get("device", "cpu")
        device = torch.device(device)

        scfg = cfg.get("sample") or {}

        nfe = scfg.get("nfe", 500)
        eps = scfg.get("eps", 1e-5)
        tol = scfg.get("tol", 100)


        model.eval()
        z = z.to(device)
        x0 = x0.to(device)

        # init
        t0 = torch.ones(z.shape[0], device=device)
        std0 = _reshape_like_std(z, marginal_prob_std(t0))
        x = torch.randn_like(z, device=device) * std0

        time_steps = torch.linspace(1.0, float(eps), int(nfe), device=device) **2
        
        if len(time_steps) >= 2:
            step_size = (time_steps[0] - time_steps[1]).item()
        else:
            step_size = (1.0 - float(eps))

        step_size_t = torch.tensor(step_size, device=device)

        mean_x = x
        for i, time_step in enumerate(tqdm(time_steps, desc="sample", leave=False)):
            batch_t = torch.ones(z.shape[0], device=device) * time_step

            std = _reshape_like_std(x, marginal_prob_std(batch_t))
            g   = _reshape_like_std(x, diffusion_coeff(batch_t))

            score = model(x, batch_t) / (std + 1e-5)
            like_score = measurement.likelihood_score(z, x)

            w = torch.relu(torch.ones_like(batch_t) - 2.0 * batch_t)
            w = _reshape_like_std(x, w)

            guided_score = score + like_score * w 

            reduce_dims = tuple(range(1, x.dim()))
            score_norm = torch.sqrt(torch.mean(guided_score ** 2, dim=reduce_dims))
            adj_ratio = torch.ones_like(score_norm)
            adj_ratio[score_norm>tol] = tol/score_norm[score_norm>tol]
      
            while adj_ratio.dim() < guided_score.dim():
                adj_ratio = adj_ratio.unsqueeze(-1)
            guided_score = guided_score * adj_ratio

            # Euler-Maruyama
            mean_x = x + (g ** 2) * guided_score * step_size_t
            x = mean_x + torch.sqrt(step_size_t) * g * torch.randn_like(x)

        return mean_x