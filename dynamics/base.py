import torch


class Dynamics:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def prior(self, n_sample):
        raise NotImplementedError

    def transition(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def states_at(self, x0, steps, progress=None):
        steps, x, out = [int(s) for s in steps], x0, {}
        for cur in range(max(steps) + 1):
            if cur > 0:
                x = self.transition(x)
                if progress is not None:
                    progress(cur, max(steps))
            if cur in steps:
                out[cur] = x.clone()
        return torch.stack([out[s] for s in steps])
