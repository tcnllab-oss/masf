import torch
from .base import Measurement
from .scheduler import a, r


class CenterMask(Measurement):
    def __init__(self,
                 noise_std=1.0,
                 grid_size=64,
                 hole_ratio=0.6,              
                 device="cuda",
                 seed=43,
                 T=0.992,
                 terminal_time=1e-5,
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
                         terminal_time=terminal_time,
                         eps=eps,
                         same_normalization=same_normalization,
                         normalization_form=normalization_form,
                         stats_mode=stats_mode,
                         stats_update_mode=stats_update_mode,
                         momentum=momentum
                    )


        self.grid_size = grid_size
        self.hole_ratio = hole_ratio

        # make mask
        mask = torch.ones((grid_size, grid_size), device=self.device)
        hole_size = int(round(grid_size * hole_ratio))
        hole_size = max(1, min(hole_size, grid_size))
        start = (grid_size - hole_size) // 2
        end = start + hole_size
        mask[start:end, start:end] = 0.0
        self.mask = mask[None, None, :, :]

    # -------------------------------------------------
    # physical measurement
    # -------------------------------------------------
    def H_phy(self, x_phy):
        return self.mask * x_phy
    
    # -------------------------------------------------
    # linear operators
    # -------------------------------------------------
    def A(self, t):
        t = t.to(self.device)
        a_t = a(t, eps=self.terminal_time, T=self.T).view(-1, *([1] * (self.mask.ndim - 1)))
        return self.norm.s / (self.norm.s_z + self.eps) * self.mask * (1.0 - a_t) + a_t

    def B(self, t):
        t = t.to(self.device)
        b = (self.mask * self.norm.u - self.norm.u_z ) / (self.norm.s_z + self.eps)
        a_t = a(t, eps=self.terminal_time, T=self.T).view(-1, *([1] * (self.mask.ndim - 1)))
        return (1.0 - a_t) * b

    def M(self, s, t):
        return self.A(t) / (self.A(s) + self.eps)

    def U(self, s, t):
        M_s_t = self.M(s, t)
        return self.B(t) - M_s_t * self.B(s)

    def Sigma(self, s, t):
        Sigma_t = self.Cov(t)
        Sigma_s = self.Cov(s)
        M_s_t = self.M(s, t)
        return Sigma_t - M_s_t * Sigma_s * M_s_t

    # -------------------------------------------------
    # Sampling operator 
    # -------------------------------------------------
    def A_apply(self, x, t):
        return self.A(t) * x 

    def M_apply(self, x, s, t):
        return self.M(s,t) * x 

    def Sigma_apply(self, x, s, t):
        return self.Sigma(s, t) * x 
    
    def Sigma_root_apply(self, x, s, t):
        return torch.sqrt(self.Sigma(s, t).abs()) * x 

    # -------------------------------------------------
    # posterior terms
    # -------------------------------------------------
    def guidance(self, z, x, t):
        z = z.to(self.device)
        x = x.to(self.device)
        t = t.to(self.device)

        t1 = torch.ones_like(t, device=self.device)
        M_t_1 = self.M(t, t1)
        Sigma_t_1 = self.Sigma(t, t1)

        residual = z - M_t_1 * x - self.U(t, t1)
        return M_t_1 * residual / (Sigma_t_1 + self.eps)
