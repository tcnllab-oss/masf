import torch


def normalization(x, device, norm_type, eps=1e-5):
    x = x.detach().to(device)
    reduce_dims = tuple(d for d in range(x.ndim) if d != 1)
    if norm_type == "standard":
        return x.mean(dim=reduce_dims, keepdim=True), x.std(dim=reduce_dims, keepdim=True).clamp_min(eps)
    center = x
    for d in sorted(reduce_dims, reverse=True):
        center = center.median(dim=d, keepdim=True).values
    scale = (x - center).abs()
    for d in sorted(reduce_dims, reverse=True):
        scale = scale.median(dim=d, keepdim=True).values
    return center, (1.4826 * scale).clamp_min(eps)


class NormalizationManager:
    def __init__(self, device, same_normalization, normalization_form, stats_mode, stats_update_mode, momentum=None, eps=1e-5):
        self.device, self.eps = torch.device(device), eps
        self.same_normalization, self.normalization_form, self.stats_mode = same_normalization, normalization_form, stats_mode
        self.stats_update_mode, self.momentum = stats_update_mode, momentum
        self.u = self.s = self.u_z = self.s_z = self.data_std_x = self.data_std_z = None
        self.initialized = False

    def _ema(self, old, new):
        return new if old is None else (1.0 - self.momentum) * old + self.momentum * new

    def update_stats(self, prior, H_phy):
        if self.stats_update_mode == "fixed" and self.initialized:
            return
        prior = prior.to(self.device)
        u, s = normalization(prior, self.device, self.stats_mode, self.eps)
        u_z, s_z = (u, s) if self.same_normalization else normalization(H_phy(prior), self.device, self.stats_mode, self.eps)
        std_x = torch.clamp(s.mean(), min=self.eps)
        std_z = std_x if self.same_normalization else torch.clamp(s_z.mean(), min=self.eps)
        new = dict(u=u, s=s, u_z=u_z, s_z=s_z, data_std_x=std_x, data_std_z=std_z)
        for k, v in new.items():
            setattr(self, k, self._ema(getattr(self, k), v) if self.stats_update_mode == "moving" else v)
        self.initialized = True

    def _stats(self, is_z):
        # (center, scale) used for x or z
        if is_z and not self.same_normalization:
            return (0.0, self.data_std_z) if self.normalization_form == "scale_only" else (self.u_z, self.s_z)
        return (0.0, self.data_std_x) if self.normalization_form == "scale_only" else (self.u, self.s)

    def norm(self, x):
        c, s = self._stats(False)
        return (x - c) / (s + self.eps)

    def denorm(self, x):
        c, s = self._stats(False)
        return x * (s + self.eps) + c

    def norm_z(self, x):
        c, s = self._stats(True)
        return (x - c) / (s + self.eps)
