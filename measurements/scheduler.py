

import torch
import math

cosine_s = 0.008
def beta(t):
    t = torch.clamp(t, 1e-5, 0.99)
    beta = math.pi/2*2/(cosine_s+1)*torch.tan( (t+cosine_s)/(1+cosine_s)*math.pi/2 )
    beta = torch.clamp(beta,0,20)
    return beta

def cosine_log_alpha(t):
    log_alpha_0 = math.log(math.cos(cosine_s / (1.0 + cosine_s) * math.pi / 2.0))
    v = (t + cosine_s) / (1.0 + cosine_s) * (math.pi / 2.0)
    return torch.log(torch.cos(v)) - log_alpha_0

def a(t, eps = 1e-5, T = 0.992):
    t = torch.clamp(t, eps, T)
    return torch.exp(cosine_log_alpha(t))

def r(t):
    return torch.sqrt(1- a(t)**2)


def score_scale_fn(s, min_scale=1, max_scale=1.5, power=1):
    s = torch.clamp(s, 0.0, 1.0)
    return min_scale + (max_scale - min_scale) * s**power

def guidance_scale_fn(s, min_scale=1.1, max_scale=1.5, power=2):
    s = torch.clamp(s, 0.0, 1.0)
    return min_scale + (max_scale - min_scale) * (1.0 - s)**power
