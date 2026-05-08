import torch
from .base import Measurement
from .scheduler import *

def speed_to_two_channel(x, tau=3, sharpness=1, eps=1e-5):
    ux = x[:, 0]
    uy = x[:, 1]
    speed = ux**2 + uy**2
    y = torch.sigmoid(sharpness * (speed - tau))
    return y.unsqueeze(1).repeat(1, 2, 1, 1)

class Nonlinear(Measurement):
    def __init__(self, 
                 noise_std=1.0,
                 type="sigmoid", # sigmoid | tanh | speed
                 alpha=2,
                 device="cuda", 
                 seed=43,
                 T=0.992, 
                 eps=1e-5,
                 same_normalization=True,
                 normalization_form="affine",      # "affine" | "scale_only"
                 stats_mode="standard",            # "standard" | "robust" | "minmax"
                 stats_update_mode="fixed",        # None | "fixed" | "adaptive" | "moving"
                 momentum=0.05,                    # used only for moving
            ):
        super().__init__(noise_std=noise_std,
                         device=device, 
                         seed=seed,
                         T=T, 
                         eps=eps,
                         same_normalization=same_normalization,
                         normalization_form=normalization_form,
                         stats_mode=stats_mode,
                         stats_update_mode=stats_update_mode,
                         momentum=momentum
                    )

        
        self.alpha = alpha
        self.type = type

    # -------------------------------------------------
    # physical measurement
    # -------------------------------------------------
    def H_phy(self, x_phy, alpha=2):
        if self.type == "sigmoid":
            return torch.sigmoid(self.alpha * x_phy)
        elif self.type == "tanh":
            return torch.tanh(self.alpha * x_phy)
        elif self.type == "speed":
            return speed_to_two_channel(x_phy)
        else:
            raise ValueError(f"Unknown nonlinearity type: {self.type}")

    # -------------------------------------------------
    # Sampling operator 
    # -------------------------------------------------
    def Sigma(self, s, t):
        s = s.to(self.device)
        t = t.to(self.device)
        return self.Cov(t) - self.Cov(s)

    # -------------------------------------------------
    # posterior terms
    # -------------------------------------------------
    def guidance(self, z_norm, x_norm, model, t):
        with torch.enable_grad():
            x_norm = x_norm.detach().clone().to(self.device).requires_grad_(True)
            z_norm = z_norm.to(self.device)
            t = t.to(self.device)
            t1 = t*0 + 0.992
       
            predicted_x_norm, predicted_h_norm = model(x_norm, t)
            a_t1 = a(t1).view(-1, *([1] * (x_norm.ndim - 1)))
            a_t = a(t).view(-1, *([1] * (x_norm.ndim - 1)))

            mean = x_norm +( - a_t) * (predicted_x_norm - predicted_h_norm)
            B = t.shape[0]
            t1 = t.new_full((B,), 0.992)
            Sigma_t_1 = self.Sigma(t, t1)

            residual = z_norm - mean
            denom = Sigma_t_1.float() + self.eps
            loss = 0.5 * (residual.float().pow(2) / denom).sum()

            grad_x = torch.autograd.grad(loss, x_norm, create_graph=False, retain_graph=False)[0]

            return -grad_x

