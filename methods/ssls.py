import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from .base import Method



class SSLS(Method):
    name = "ssls"

    def train(self, cfg, device, model,
              optimizer, train_loader, val_loader, 
              n_epoch, step, measurement, workdir='.'):

        loss_ft = nn.MSELoss()
        global_step = 0
        sigma = cfg["method"]["sigma"]

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
                z = torch.randn_like(x0, device=device)
                x = x0 + sigma * z
                
                score = model(x)

                # loss 
                loss = loss_ft(score, -z)
                # loss = loss_ft(score, x0)

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
            
            # model.eval()
            # epoch_val_loss = 0.0
            # with torch.no_grad():
            #     for batch in val_loader:
            #         (x,) = batch
            #         x0 = x.to(device)
            #         z = torch.randn_like(x0, device=device)
            #         xt = x0 + sigma * z
            #         score = model(xt)
            #         val_loss = loss_ft(score, -z)
            #         epoch_val_loss += float(val_loss.item())

            # epoch_val_loss /= max(len(val_loader), 1)

        return model

    @torch.no_grad()
    def sample(self, cfg, model, x0, z, measurement, device=None, path=None):
        if device is None:
            device = (cfg.get("system") or {}).get("device", "cuda:0")
        
        # sample configuration
        sigma = cfg["method"]["sigma"]
        lam = cfg["sample"]["anneal_init"]
        gamma = cfg["sample"]["anneal_decay"]
        tol = cfg["sample"]["tol"]

        anneal_steps = cfg["sample"]["anneal_steps"]
        nfe = cfg["sample"]["nfe"]
        dt = float(cfg["sample"]["stepsize"])

        model.eval()
        x = x0.to(device) 
        z = z.to(device)

        phi = lambda x: (np.exp(x) - 1) / x

        for _ in range(anneal_steps):
            for i in range(nfe):
                noise = torch.randn_like(x0, device=x0.device) * (2 * dt * phi(-2 * lam * dt)) ** 0.5
                # noise = torch.randn_like(x0, device=x0.device) * (2 * dt ) ** 0.5

                score = model(x) / (sigma + 1e-5)
                s = torch.ones((x.shape[0],), device=x.device) * (nfe- i) / nfe
                
                # guided score 
                grad =  measurement.likelihood_score(z, x)  + score 
           
                # # score normalziation
                grad_norm = torch.sqrt(torch.mean(grad**2, dim=tuple(range(1, x.ndim)), keepdim=True))
                adj_ratio = torch.ones_like(grad_norm, device=x.device)
                adj_ratio[grad_norm > tol] = tol / grad_norm[grad_norm > tol]
                grad = grad * adj_ratio
                
                # update 
                x = np.exp(-lam * dt) * x + dt * phi(-lam * dt) * grad + noise
                # x =  x + dt * grad + noise

            #     if (i % 50 == 0) or (i == nfe - 1):
            #         k_min = float(x.amin().item())
            #         k_max = float(x.amax().item())
            #         k_mean = float(x.mean().item())
            #         k_std = float(x.std(unbiased=False).item())
            #         print(f"{i:02d}/{nfe} | x range=({k_min:.4f}, {k_max:.4f}) mean={k_mean:.4f} std={k_std:.4f}")

            #         if x.dim() ==4: 
            #             save_vorticity_pairs_color(x0[:4], x[:4], 'ssls.png', noisy_uv=z[:4], cmap=sns.cm.icefire)
            #         if path:
            #             if x.dim() == 4:
            #                 save_vorticity_pairs_color(x0[:4], x[:4], path, noisy_uv=z[:4], cmap=sns.cm.icefire)
            #             elif x.dim() == 3:
            #                 plot_line_1d(
            #                     x0.detach().cpu().numpy(),
            #                     x.detach().cpu().numpy(),
            #                     z.detach().cpu().numpy(),
            #                     path=path,
            #                 )
            # if x.dim() ==4: 
            #             save_vorticity_pairs_color(x0[:4], x[:4], 'ssls.png', noisy_uv=z[:4], cmap=sns.cm.icefire)
                    
            lam = gamma * lam
        return x