from abc import ABC, abstractmethod

class Method(ABC):
    def __init__(self):
        pass
    def need_train(self):
        return False

    def train(self, cfg, device, model,
              optimizer, train_loader,
              val_loader, n_epoch, step,
              measurement, workdir):
        return None

    @abstractmethod
    def sample(self, ctx, step_i, prior, z_i):
        return None 