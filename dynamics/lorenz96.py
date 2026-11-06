import torch
from .base import Dynamics


class Lorenz96(Dynamics):
    def __init__(self, dim=10, prior_mean=0.0, prior_std=1.0, dt=0.01, forcing=8.0, perturb_std=0.1, solver="Euler", seed=42):
        super().__init__(shape=(1, dim))
        self.seed = seed
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.dt = dt
        self.forcing = forcing
        self.perturb_std = perturb_std

        solvers = {"Euler": self._euler, "Heun": self._heun, "Runge-Kutta": self._runge_kutta}
        self.step_fn = solvers[solver]

        self._gens = {}

    def _get_gen(self, device):
        key = (device.type, device.index)
        if key not in self._gens:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            self._gens[key] = gen
        return self._gens[key]

    def prior(self, n_sample):
        device = torch.device("cpu")
        gen = self._get_gen(device)
        prior = self.prior_mean + torch.randn((n_sample, *self.shape), generator=gen) * self.prior_std
        return prior

    def f(self, x):
        x_p1, x_m2, x_m1 = [torch.roll(x, i, dims=-1) for i in [-1, 2, 1]]
        return (x_p1 - x_m2) * x_m1 - x + self.forcing

    def transition(self, x):
        gen = self._get_gen(x.device)
        noise = torch.randn(x.shape, device=x.device, generator=gen)
        return x + self.dt * self.step_fn(x) + self.perturb_std * noise * (self.dt ** 0.5)

    def _euler(self, x):
        return self.f(x)

    def _heun(self, x):
        k1 = self.f(x)
        k2 = self.f(x + self.dt * k1)
        return (k1 + k2) / 2

    def _runge_kutta(self, x):
        k1 = self.f(x)
        k2 = self.f(x + self.dt * k1 / 2)
        k3 = self.f(x + self.dt * k2 / 2)
        k4 = self.f(x + self.dt * k3)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6