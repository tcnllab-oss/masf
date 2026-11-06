import torch
from .scheduler import a, r
from .normalization import NormalizationManager


class Measurement:
    def __init__(self, noise_std, device, seed, same_normalization, normalization_form, stats_mode, stats_update_mode, momentum=None, eps=1e-5):
        self.device = torch.device(device)
        self.noise_std = noise_std
        self.seed = seed
        self.eps = eps

        self.norm = NormalizationManager(device=device, eps=eps, same_normalization=same_normalization, normalization_form=normalization_form,
                                         stats_mode=stats_mode, stats_update_mode=stats_update_mode, momentum=momentum)
        self._gens = {}
    # measurement operators

    def observation_phy(self, x):
        x = x.to(self.device)
        gen = self._get_gen(self.device)
        z_clean= self.H_phy(x)
        noise = torch.randn(z_clean.shape, device=self.device, generator=gen)
        return z_clean + self.noise_std * noise

    # normalization entrypoint
    def update_stats(self, prior):
        prior = prior.to(self.device)
        self.norm.update_stats(prior, H_phy=self.H_phy)
        
    # helpers
    def a(self, t):
        return a(t.to(self.device))

    def sigma_norm(self):
        return self.noise_std / (self.norm.s_z + self.eps)

    def _expand_t_like_x(self, t, x):
        return t.view(x.shape[0], *([1] * (x.dim() - 1)))

    def _get_gen(self, device):
        key = (device.type, device.index)
        if key not in self._gens:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            self._gens[key] = gen
        return self._gens[key]

    # measurement operators in normalized space
    def H(self, x_norm):
        x_norm = x_norm.to(self.device)
        x_phy = self.norm.denorm(x_norm)
        z_phy = self.H_phy(x_phy)
        return self.norm.norm_z(z_phy)

    def observation(self, x_norm):
        x_norm = x_norm.to(self.device)
        gen = self._get_gen(self.device)

        z_clean_norm = self.H(x_norm)
        sigma_norm = self.sigma_norm()
        noise = torch.randn(z_clean_norm.shape, device=self.device, generator=gen)

        return z_clean_norm + sigma_norm * noise

    # forward process
    def h(self, x_norm, t):
        t = t.to(self.device)
        x_norm = x_norm.to(self.device)
        a_t = self._expand_t_like_x(self.a(t), x_norm)
        return a_t * x_norm + (1.0 - a_t) * self.H(x_norm)

    def Cov(self, t):
        t = t.to(self.device)
        gamma2 = r(t).to(self.device) ** 2 #[B,]
        sigma_norm = self.sigma_norm() #[1, C, 1, 1]
        gamma2 = gamma2.view(-1, *([1] * (sigma_norm.ndim - 1)))
        cov = (sigma_norm ** 2) * gamma2
        return cov #[ B, C, 1, 1]

    def forward_process(self, x_norm, t):
        x_norm = x_norm.to(self.device)
        t = t.to(self.device)
        gen = self._get_gen(self.device)
        noise = torch.randn(x_norm.shape, device=x_norm.device, generator=gen)
        Sigma_t = self.Cov(t) #[B, C, 1 ,1]
        x_t = self.h(x_norm, t) + torch.sqrt(torch.clamp(Sigma_t, min=0.0)) * noise
        return x_t, noise