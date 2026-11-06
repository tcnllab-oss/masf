import json
import os
import re
import torch
from tqdm.auto import tqdm


def find_reusable_dataset_dir(root, dim, num_samples):
    if not os.path.isdir(root):
        return None

    candidates = []
    for name in os.listdir(root):
        m, meta_path = re.match(rf"^dim{dim}_num_samples(\d+)$", name), os.path.join(root, name, "meta.json")
        if m is None or int(m.group(1)) < num_samples or not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if meta["batch"] >= num_samples and meta["state_shape"][-1] == dim:
            candidates.append((int(m.group(1)), os.path.join(root, name)))
    return min(candidates)[1] if candidates else None


@torch.no_grad()
def save_state(dynamics, x0, step, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pbar = tqdm(desc="Generating trajectory", unit="step", dynamic_ncols=True)

    def progress(cur, total):
        pbar.total = total; pbar.update(cur - pbar.n)
    x_s = dynamics.states_at(x0, [step], progress)[0]
    torch.save({"step": step, "state": x_s.float().cpu()}, os.path.join(out_dir, f"step_{step:06d}.pt"))
    pbar.close()
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"batch": x0.shape[0], "state_shape": list(x0.shape[1:])}, f, indent=2)


def load_or_create_state(dynamics, out_dir, step, n_store, n_load, device):
    path = os.path.join(out_dir, f"step_{step:06d}.pt")
    if not os.path.exists(path):
        save_state(dynamics, dynamics.prior(n_store).to(device), step, out_dir)

    state = torch.load(path, map_location=device)["state"]
    assert state.shape[0] >= n_load, f"{path} has {state.shape[0]} samples < {n_load}"
    return state[:n_load].float()
