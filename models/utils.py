import os
import torch.nn as nn
import torch


def finetune_param_groups(model, cfg):
    dyn_type = (cfg.get("dynamics", {}) or {}).get("type", "").lower()
    model_ = getattr(model, "module", model)

    params = []

    if dyn_type == "lorenz63":
        # TimeCondMLP 계열
        params += list(model_.out_proj.parameters())
        params += list(model_.time_mlp.parameters())
        for blk in model_.blocks:
            params += list(blk.to_scale_shift.parameters())

    elif dyn_type == "lorenz96":
        params += list(model_.time_mlp.parameters())
        params += list(model_.final_conv.parameters())

        def collect_resnet_mlp(m):
            if hasattr(m, "mlp") and m.mlp is not None:
                return list(m.mlp.parameters())
            return []

        for blocks in model_.downs:
            for m in blocks:
                params += collect_resnet_mlp(m)
        params += collect_resnet_mlp(model_.mid_block1)
        params += collect_resnet_mlp(model_.mid_block2)
        for blocks in model_.ups:
            for m in blocks:
                params += collect_resnet_mlp(m)
        params += collect_resnet_mlp(model_.final_res_block)

    elif dyn_type == "kolmogorov":
        if hasattr(model_, "out"):
            params += list(model_.out.parameters())
        if hasattr(model_, "time_embed"):
            params += list(model_.time_embed.parameters())
        for m in model_.modules():
            if m.__class__.__name__ == "ResBlock" and hasattr(m, "emb_layers"):
                params += list(m.emb_layers.parameters())

    else:
        raise ValueError(f"unsupported dynamics.type={dyn_type}")

    seen = set()
    uniq = []
    for p in params:
        if id(p) not in seen:
            uniq.append(p)
            seen.add(id(p))
    return uniq

# def save_step_ckpt(path, model, norm, denorm):
#     payload = {
#         "model": model.state_dict(),
#         "norm": norm,
#         "denorm": denorm,
#     }
#     torch.save(payload, path)


def save_step_ckpt(path, model):
    payload = {"model": model.state_dict()}
    torch.save(payload, path)


def load_step_ckpt(path, model, map_location="cpu", device="cpu", strict=True):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=strict)

    return model.to(device)
