import os
import torch
from utils.build import build_pretraining_dataset, build_model, build_dataloader, build_pretrained_path
from models.utils import load_step_ckpt, save_step_ckpt


def build_optimizer(model, opt_cfg):
    return torch.optim.AdamW(model.parameters(), lr=float(opt_cfg["lr"]), weight_decay=opt_cfg["weight_decay"], betas=tuple(opt_cfg["betas"]))


def train_model(cfg, model, opt_cfg, method, measurement, prior, device, path, epoch, step):
    measurement.update_stats(prior)
    train_loader, val_loader = build_dataloader(measurement.norm.norm(prior), opt_cfg["batch_size"], float(opt_cfg["val_ratio"]))
    model.train()

    method.train(cfg, device=device, model=model, optimizer=build_optimizer(model, opt_cfg), train_loader=train_loader, val_loader=val_loader, n_epoch=epoch, step=step, measurement=measurement)
    save_step_ckpt(path, model)
    
    return model


def load_pretrained_model(cfg, model, dynamics, method, measurement, steps, device):
    pre, path = cfg["pretrain"], build_pretrained_path(cfg)
    print("pretrain path:", path)
    prior = build_pretraining_dataset(cfg, dynamics, steps)
    measurement.update_stats(prior)

    if os.path.exists(path):
        print("pretrained model is loaded...")
        return load_step_ckpt(path, model, device)
    
    print("pretrained model is training...")
    return train_model(cfg, model, pre, method, measurement, prior, device, path, pre["epoch"], "pretrain")


def load_trained_model(cfg, method, measurement, dynamics, i, index, steps, prior, device, workdir):
    tr, model = cfg["train"], build_model(cfg)
    path = os.path.join(workdir, "ckpt", f"ckpt_{int(i)}th_step.pt")

    if index == 0:
        if tr["pretrained"]:
            print("[INIT] step 0 initialized from pretrained")
            model = load_pretrained_model(cfg, model, dynamics, method, measurement, steps, device)
            if not tr["online"].get("finetune_first", False):
                return model
            print(f"{i} th step training...")
            return train_model(cfg, model, tr, method, measurement, prior, device, path, tr["online"]["full_epoch"], i)
        print("[INIT] step 0 initialized from scratch")
        return train_model(cfg, model, tr, method, measurement, prior, device, path, tr["epoch"], i)

    prev_path = os.path.join(workdir, "ckpt", f"ckpt_{int(steps[index - 1])}th_step.pt")
    if os.path.exists(prev_path):
        model = load_step_ckpt(prev_path, model, device)
    else:
        model = load_pretrained_model(cfg, model, dynamics, method, measurement, steps, device)
    print(f"{i} th step training...")
    return train_model(cfg, model, tr, method, measurement, prior, device, path, tr["online"]["full_epoch"] if tr["pretrained"] else tr["epoch"], i)
