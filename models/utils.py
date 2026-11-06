import torch


def save_step_ckpt(path, model):
    torch.save({"model": model.state_dict()}, path)


def load_step_ckpt(path, model, device):
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return model.to(device)
