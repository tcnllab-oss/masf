import torch
from .base import Measurement
from .scheduler import *


def speed_to_two_channel(x, tau=3, sharpness=1):
    speed = x[:, 0]**2 + x[:, 1]**2
    y = torch.sigmoid(sharpness * (speed - tau))
    return y.unsqueeze(1).repeat(1, 2, 1, 1)


class Speed(Measurement):
    def __init__(self, tau, sharpness, **kw):
        super().__init__(**kw)
        self.tau = tau
        self.sharpness = sharpness

    # physical measurement
    def H_phy(self, x_phy):
        return speed_to_two_channel(x_phy, tau=self.tau, sharpness=self.sharpness)

    # Sampling operator
    def Sigma(self, s, t):
        return self.Cov(t.to(self.device)) - self.Cov(s.to(self.device))

    # posterior terms
    def guidance(self, z_norm, x_norm, model, t):
        with torch.enable_grad():
            x_norm = x_norm.detach().clone().to(self.device).requires_grad_(True)
            z_norm = z_norm.to(self.device)
            t = t.to(self.device)
            t1 = t*0 + 0.992

            predicted_x_norm, predicted_h_norm = model(x_norm, t)
            a_t = a(t).view(-1, *([1] * (x_norm.ndim - 1)))

            mean = x_norm - a_t * (predicted_x_norm - predicted_h_norm)
            Sigma_t_1 = self.Sigma(t, t1)

            residual = z_norm - mean
            denom = Sigma_t_1.float() + self.eps
            loss = 0.5 * (residual.float().pow(2) / denom).sum()

            grad_x = torch.autograd.grad(loss, x_norm, create_graph=False, retain_graph=False)[0]
            return -grad_x
