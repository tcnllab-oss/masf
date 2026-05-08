
import os
import random
import torch  
import numpy as np 
from utils.build import (build_pretraining_dataset, build_model, build_dataloader,
                         build_pretrained_workdir, build_pretrained_path)
from models.utils import load_step_ckpt, save_step_ckpt


def train_model(cfg, model, optimizer, 
                method, measurement,
                prior, device, 
                path, epoch, 
                batch_size, 
                step = 0, val_ratio=0.1,
                workdir = '.'):


    measurement.update_stats(prior)

    # normalization 
    n_prior = measurement.norm.norm(prior)

    # dataloader
    train_loader, val_loader = build_dataloader(n_prior, batch_size, val_ratio)

    # score_fn training
    model.train()
    method.train(cfg, device=device, model=model,
                 optimizer=optimizer,
                 train_loader=train_loader, val_loader=val_loader,
                 n_epoch=epoch, step=step, measurement=measurement,
                 workdir = workdir)
    
    # save configuraiton 
    save_step_ckpt(path, model)
    return model

def load_pretrained_model(cfg, model, dynamics, method, measurement,
                          steps, device, map_location="cpu",
                          strict=True, override=False):
    pretrain_cfg = cfg["pretrain"]

    # Build experiment directory first.
    workdir = build_pretrained_workdir(cfg)
    path = build_pretrained_path(cfg)

    print("pretrain workdir:", workdir)
    print("pretrain path:", path)

    # Load existing pretrained checkpoint 
    prior = build_pretraining_dataset(cfg, dynamics, steps)
    measurement.update_stats(prior)

    if not override and os.path.exists(path):
        print("pretrained model is loaded...")
        return load_step_ckpt(path, model,
                              map_location=map_location,
                              device=device,
                              strict=strict)

    print("pretrained model is training...")

    # Optimizer for pretraining.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(pretrain_cfg["lr"]),
        weight_decay=pretrain_cfg["weight_decay"],
        betas=tuple(pretrain_cfg["betas"]))

    model.train()

    # Train from scratch and save checkpoint to `path`.
    model= train_model(
        cfg, model, optimizer, method, measurement, prior, device,
        path=path,
        epoch=pretrain_cfg["epoch"],
        batch_size=pretrain_cfg["batch_size"],
        step="pretrain",
        val_ratio=float(pretrain_cfg["val_ratio"]),
        workdir=workdir)
    
    return model 


def load_trained_model(cfg, method, measurement, dynamics, 
                       i, index, steps, prior, device, workdir,
                       map_location="cpu", strict=True):

    # 0) no train
    need_train = cfg["method"]["train"]
    override = cfg["train"]["override"]

    if not need_train:
        return None, None

    # 1) model setting 
    model = build_model(cfg).to(device)
    path = os.path.join(workdir, 'ckpt', f"ckpt_{int(i)}th_step.pt")
    need_normalization=True
    
    # 2) load the ckpt if it has been trained
    if os.path.exists(path) and not override:
        measurement.update_stats(prior)

        print(f"[CKPT] loading current step checkpoint: {path}")
        model = load_step_ckpt(path, model, map_location=map_location, device=device, strict=strict)
        return model, need_normalization

    # 3) initialization
    print('index', index)
    loaded_pretrained = cfg["train"]["pretrained"]
    if index == 0:
        # using pretrained_model
        override =cfg["pretrain"].get("override") 
        if loaded_pretrained:
            model = load_pretrained_model(cfg, model, dynamics, method, measurement, 
                                                        steps, device, map_location=map_location, strict=strict,
                                                        override = override)
            print("[INIT] step 0 initialized from pretrained")
            print("===============================================")

            return model, need_normalization

        
        # train from scratch
        else:
            print("[INIT] step 0 initialized from scratch")
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=float(cfg["train"]["lr"]),
                                          weight_decay=cfg["train"]["weight_decay"],
                                          betas=tuple(cfg["train"]["betas"]))
            model= train_model(cfg, model, optimizer, 
                                              method, measurement,
                                              prior, device, 
                                              path, cfg["train"]["epoch"], 
                                              cfg["train"]["batch_size"], 
                                              step = i, val_ratio=float(cfg["train"]["val_ratio"]),
                                              workdir=workdir)
            return model, need_normalization

    else:
        # previous model uploaded 
        print(f"{i} th step model loaded")
        prev_i = steps[index - 1]
        prev_path = os.path.join(workdir, 'ckpt', f"ckpt_{int(prev_i)}th_step.pt") #이게 없으면 
        if os.path.exists(prev_path) :
            model = load_step_ckpt(prev_path, model, map_location=map_location, device=device,  strict=strict,
                                                 )
        else:
            model = load_pretrained_model(cfg, model, dynamics, method, measurement, 
                                        steps, device, map_location=map_location, strict=strict, override=False)

    # 4) optimizer / schedule
    train_cfg = cfg.get("train", {})
    lr = float(train_cfg["lr"])
    epoch_scratch = train_cfg.get("epoch", 200)

    if loaded_pretrained:
        full_epoch = train_cfg["online"]["full_epoch"]
    else:
        full_epoch = epoch_scratch

    n_epoch = full_epoch    
    optimizer = torch.optim.AdamW(model.parameters(),
                                        lr=lr,
                                        weight_decay=cfg["train"]["weight_decay"],
                                        betas=tuple(cfg["train"]["betas"]))

    print(f"{i} th step training...")
    model = train_model(cfg, model, optimizer, 
                                      method, measurement,
                                      prior, device, 
                                      path, n_epoch, 
                                      cfg["train"]["batch_size"], 
                                      step = i, val_ratio=float(cfg["train"]["val_ratio"]),
                                       workdir=workdir)


    return model, need_normalization