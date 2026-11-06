import torch
from .base import Measurement


class Cubic(Measurement):

    def __init__(self, alpha, **kw):
        super().__init__(**kw)
        self.alpha = alpha

    def H_phy(self, x_phy):
        return x_phy + self.alpha * x_phy.pow(3)

    def Sigma(self, s, t):
        return self.Cov(t.to(self.device)) - self.Cov(s.to(self.device))

    def guidance(self, z_norm, x_norm, model, t):
        with torch.enable_grad():
            x_norm = x_norm.detach().clone().to(self.device).requires_grad_(True)
            _, predicted_h_norm = model(x_norm, t.to(self.device))
            loss = 0.5 * ((z_norm.to(self.device) - predicted_h_norm).float().pow(2) / (self.sigma_norm() ** 2 + self.eps)).sum()
            return -torch.autograd.grad(loss, x_norm)[0]
