import torch
import math

cosine_s = 0.008


def a(t):
    t = torch.clamp(t, 0.0, 1.0)
    alpha_0 = math.cos(cosine_s / (1.0 + cosine_s) * math.pi / 2.0)
    alpha = torch.clamp(torch.cos((t + cosine_s) / (1.0 + cosine_s) * (math.pi / 2.0)) / alpha_0, 0.0, 1.0)
    alpha = torch.where(t == 0.0, torch.ones_like(alpha), alpha)
    return torch.where(t == 1.0, torch.zeros_like(alpha), alpha)


def r(t):
    return torch.sqrt(torch.clamp(1.0 - a(t) ** 2, min=0.0, max=1.0))
