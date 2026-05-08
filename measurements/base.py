import torch
from abc import ABC, abstractmethod
from .scheduler import a, r
from .normalization import NormalizationManager


class Measurement(ABC):
    def __init__(
        self,
        noise_std=1.0,
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
        super().__init__()
        self.device = torch.device(device)
        self.noise_std = noise_std
        self.seed = seed
        self.T = T
        self.terminal_time =terminal_time
        self.eps = eps

        self.norm = NormalizationManager(
            device=device,
            eps=eps,
            same_normalization=same_normalization,
            normalization_form=normalization_form,
            stats_mode=stats_mode,
            stats_update_mode=stats_update_mode,
            momentum=momentum
        )
        self._gens = {}
    # -------------------------------------------------
    # subclass must implement
    # -------------------------------------------------
    @abstractmethod
    def H_phy(self, x_phy):
        raise NotImplementedError

    @abstractmethod
    def guidance(self, z_norm, x_norm, t):
        raise NotImplementedError

    @abstractmethod
    def Sigma(self, s, t):
        raise NotImplementedError

    # -------------------------------------------------
    # measurement operators
    # -------------------------------------------------
    def observation_phy(self, x):
        x = x.to(self.device)
        gen = self._get_gen(self.device)
        z_clean= self.H_phy(x)
        noise = torch.randn(z_clean.shape, device=self.device, generator=gen)
        return z_clean + self.noise_std * noise
    
    def observation_clean_phy(self, x_phy):
        x_phy = x_phy.to(self.device)
        return self.H_phy(x_phy)
    
    # -------------------------------------------------
    # normalization entrypoint
    # -------------------------------------------------
    def update_stats(self, prior):
        prior = prior.to(self.device)
        self.norm.update_stats(prior, H_phy=self.H_phy)
    # -------------------------------------------------
    # helpers
    # -------------------------------------------------
    def _expand_t_like_x(self, t, x):
        return t.view(x.shape[0], *([1] * (x.dim() - 1)))

    def _get_gen(self, device):
        key = (device.type, device.index)
        if key not in self._gens:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            self._gens[key] = gen
        return self._gens[key]

    # -------------------------------------------------
    # measurement operators in normalized space
    # z = H_{phy}(x) + sigma epsilon 
    # z_norm = norm_z(H_phy(denorm_x(x_norm)))+ sigma/s_z * epsilon = H(x_norm) + tilde(sigma) epsilon
    # z_norm = H(x_norm)+ sigma_norm * epsilon, H = norm_z(H_phy(denorm))
    # x_t = a(t)x_norm + (1- a(t))H(x_norm) + ~
    # -------------------------------------------------
    def H(self, x_norm):
        x_norm = x_norm.to(self.device)
        x_phy = self.norm.denorm(x_norm)
        z_phy = self.H_phy(x_phy)
        return self.norm.norm_z(z_phy)

    def observation_clean(self, x_norm):
        x_norm = x_norm.to(self.device)
        return self.H(x_norm)

    def observation(self, x_norm):
        x_norm = x_norm.to(self.device)
        gen = self._get_gen(self.device)

        z_clean_norm = self.H(x_norm)
        sigma_norm = self.noise_std / (self.norm.s_z + self.eps)
        noise = torch.randn(z_clean_norm.shape, device=self.device, generator=gen)

        return z_clean_norm + sigma_norm * noise

    def likelihood_score(self, z_norm, x_norm):
        with torch.enable_grad():
            x_norm = x_norm.detach().requires_grad_(True)
            z_norm = z_norm.to(self.device)
            x_norm = x_norm.to(self.device)
            residual = z_norm - self.H(x_norm) 

            sigma_norm = self.noise_std / (self.norm.s_z + self.eps)
            log_p = -residual ** 2 / (2 * sigma_norm**2)
            return torch.autograd.grad(log_p.sum(), x_norm)[0]
    # -------------------------------------------------
    # forward process
    # -------------------------------------------------
    def h(self, x_norm, t):
        t = t.to(self.device)
        x_norm = x_norm.to(self.device)
        a_t = self._expand_t_like_x(a(t, eps=self.terminal_time, T= self.T).to(self.device), x_norm)
        return a_t * x_norm + (1.0 - a_t) * self.H(x_norm)

    def Cov(self, t):
        t = t.to(self.device)
        gamma2 = r(t).to(self.device) ** 2 #[B,]
        sigma_norm = self.noise_std / (self.norm.s_z + self.eps) #[1, C, 1, 1]
        gamma2 = gamma2.view(-1, *([1] * (sigma_norm.ndim - 1)))
        cov = (sigma_norm ** 2) * gamma2
        return cov #[ B, C, 1, 1]

    def forward_process(self, x_norm, t=None):
        x_norm = x_norm.to(self.device)
        if t is None:
            t = torch.ones(x_norm.shape[0], device=self.device) * self.T
        else:
            t = t.to(self.device)
        gen = self._get_gen(self.device)
        noise = torch.randn(x_norm.shape, device=x_norm.device, generator=gen)
        Sigma_t = self.Cov(t) #[B, C, 1 ,1]
        x_t = self.h(x_norm, t) + torch.sqrt(torch.clamp(Sigma_t, min=0.0)) * noise
        return x_t, noise