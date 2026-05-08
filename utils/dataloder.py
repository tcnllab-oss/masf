from __future__ import annotations
import json
import os
import re
import tempfile
import torch
from torch import Tensor
from tqdm.auto import tqdm
from pathlib import Path
from typing import Iterable, Literal


def _atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            print("save does not complete..")


def _atomic_json_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _load_json_safe(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _normalize_steps(steps: Iterable[int] | Tensor) -> list[int]:
    if torch.is_tensor(steps):
        steps_list = [int(s) for s in steps.detach().cpu().tolist()]
    else:
        steps_list = [int(s) for s in steps]

    if not steps_list:
        return []

    if any(s < 0 for s in steps_list):
        raise ValueError("steps must be non-negative")

    return sorted(set(steps_list))


def _meta_matches_requirements(
    meta: dict,
    *,
    dim: int,
    required_num_samples: int,
    required_steps: Iterable[int] | None = None,
    extra_match: dict | None = None,
) -> bool:
    if not isinstance(meta, dict):
        return False

    batch = meta.get("batch")
    state_shape = meta.get("state_shape", [])
    saved_steps = set(int(s) for s in meta.get("saved_steps", []))

    if batch is None or int(batch) < int(required_num_samples):
        return False

    if not state_shape:
        return False

    if len(state_shape) < 1:
        return False

    if int(state_shape[-1]) != int(dim):
        return False

    if required_steps is not None:
        required_steps_set = set(int(s) for s in required_steps)
        if not required_steps_set.issubset(saved_steps):
            return False

    if extra_match:
        for k, v in extra_match.items():
            if meta.get(k) != v:
                return False

    return True


def find_reusable_dataset_dir(
    base_root: str,
    dim: int,
    required_num_samples: int,
    *,
    required_steps: Iterable[int] | None = None,
    extra_match: dict | None = None,
    require_meta: bool = True,
) -> str | None:
    """
    Find the smallest reusable dataset directory that satisfies:
    - folder name matches dim{dim}_num_samples{N}
    - N >= required_num_samples
    - if require_meta=True, meta.json must exist and match requested conditions
    """
    root = Path(base_root)
    if not root.exists():
        return None

    pattern = re.compile(rf"^dim{int(dim)}_num_samples(\d+)$")
    candidates: list[tuple[int, str]] = []

    for p in root.iterdir():
        if not p.is_dir():
            continue

        m = pattern.match(p.name)
        if m is None:
            continue

        stored_n = int(m.group(1))
        if stored_n < int(required_num_samples):
            continue

        if require_meta:
            meta = _load_json_safe(p / "meta.json")
            if meta is None:
                continue
            if not _meta_matches_requirements(
                meta,
                dim=dim,
                required_num_samples=required_num_samples,
                required_steps=required_steps,
                extra_match=extra_match,
            ):
                continue

        candidates.append((stored_n, str(p)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def is_valid_saved_step_file(
    path: Path,
    *,
    required_num_samples: int | None = None,
    required_dim: int | None = None,
) -> bool:
    if not path.exists():
        return False

    try:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        return False

    if not isinstance(obj, dict) or "state" not in obj:
        return False

    t = obj["state"]
    if not torch.is_tensor(t):
        return False

    if required_num_samples is not None and t.shape[0] < required_num_samples:
        return False

    if required_dim is not None and t.shape[-1] != required_dim:
        return False

    return True


@torch.no_grad()
def save_trajectory_by_step(
    dynamics,
    x0: Tensor,
    steps: Iterable[int],
    out_dir: str,
    *,
    save_dtype: Literal["fp32", "fp16", "bf16"] = "fp32",
    cpu_store: bool = True,
    overwrite: bool = False,
    meta_extra: dict | None = None,
    show_progress: bool = True,
) -> None:
    """
    Save trajectory states as:
      out_dir/
        step_000020.pt
        step_000030.pt
        ...
        meta.json

    Each step file stores:
      {"step": int, "state": Tensor}

    Progress:
    - "Generating trajectory" shows INTERNAL trajectory progress, e.g. 1/50 ... 50/50
    - target file saving still happens only at requested steps
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _cast(t: Tensor) -> Tensor:
        if save_dtype == "fp16":
            return t.to(torch.float16)
        if save_dtype == "bf16":
            return t.to(torch.bfloat16)
        return t.to(torch.float32)

    if x0.ndim == len(dynamics.shape):
        x0n = x0.unsqueeze(0)
    else:
        x0n = x0

    N = int(x0n.shape[0])
    state_shape = tuple(x0n.shape[1:])

    meta_path = out / "meta.json"
    prev_saved: set[int] = set()

    meta_old = _load_json_safe(meta_path)
    if meta_old is not None:
        try:
            prev_saved = set(int(s) for s in meta_old.get("saved_steps", []))
        except Exception:
            prev_saved = set()

    targets = _normalize_steps(steps)
    if not targets:
        print("[save] no target steps were given.")
        return

    first_target_step = int(targets[0])
    last_target_step = int(targets[-1])
    trajectory_span = last_target_step

    exist_set: set[int] = set()
    todo: list[int] = []

    required_dim = int(state_shape[-1])

    for s in targets:
        step_path = out / f"step_{s:06d}.pt"
        if (not overwrite) and is_valid_saved_step_file(
            step_path,
            required_num_samples=N,
            required_dim=required_dim,
        ):
            exist_set.add(s)
        else:
            todo.append(s)

    total_targets = len(targets)
    total_existing = len(exist_set)
    total_todo = len(todo)

    def _write_meta(saved_steps: set[int]) -> None:
        meta = {
            "batch": N,
            "state_shape": list(state_shape),
            "dtype": save_dtype,
            "device": "cpu" if cpu_store else str(x0.device),
            "saved_steps": sorted(saved_steps),
        }
        if meta_extra:
            meta.update(meta_extra)
        _atomic_json_save(meta, meta_path)

    print("=" * 80)
    print("[save] trajectory save plan")
    print(f"[save] out_dir            : {out}")
    print(f"[save] batch              : {N}")
    print(f"[save] state_shape        : {state_shape}")
    print(f"[save] total_targets      : {total_targets}")
    print(f"[save] already_exists     : {total_existing}")
    print(f"[save] to_save            : {total_todo}")
    print(f"[save] first_target_step  : {first_target_step}")
    print(f"[save] last_target_step   : {last_target_step}")
    print(f"[save] trajectory_span    : 0 -> {last_target_step} (len={trajectory_span})")
    print("=" * 80)

    if not todo:
        _write_meta(prev_saved | exist_set)
        print(f"[save] nothing to do (all {total_targets} target steps already exist).")
        return

    newly_saved: set[int] = set()

    substep_pbar = None

    def _progress_callback(cur_step: int, max_step: int):
        nonlocal substep_pbar
        if not show_progress:
            return

        if substep_pbar is None:
            substep_pbar = tqdm(
                total=max_step,
                desc="Generating trajectory",
                unit="step",
                dynamic_ncols=True,
            )

        delta = cur_step - substep_pbar.n
        if delta > 0:
            substep_pbar.update(delta)

        substep_pbar.set_postfix_str(f"{cur_step}/{max_step}")

    iterator = dynamics.iter_states_at(
        x0n,
        todo,
        return_step=True,
        progress_callback=_progress_callback,
        include_step0_progress=False,
    )

    for s, x_s in iterator:
        step_i = int(s)
        t_out = _cast(x_s)
        if cpu_store:
            t_out = t_out.cpu()

        step_path = out / f"step_{step_i:06d}.pt"
        if step_path.exists() and not overwrite:
            continue

        _atomic_torch_save({"step": step_i, "state": t_out}, step_path)
        newly_saved.add(step_i)

    if substep_pbar is not None:
        substep_pbar.close()

    _write_meta(prev_saved | exist_set | newly_saved)

    print("=" * 80)
    print("[save] trajectory save done")
    print(f"[save] total_targets      : {total_targets}")
    print(f"[save] already_exists     : {total_existing}")
    print(f"[save] newly_saved        : {len(newly_saved)}")
    print(f"[save] final_saved        : {len(prev_saved | exist_set | newly_saved)}")
    print(f"[save] final_target_step  : {last_target_step}")
    print(f"[save] out_dir            : {out}")
    print("=" * 80)


@torch.no_grad()
def load_state_from_folder(
    folder: str,
    step: int,
    *,
    map_device: torch.device | str = "cpu",
    out_dtype: torch.dtype | None = None,
    max_samples: int | None = None,
) -> Tensor:
    path = Path(folder) / f"step_{int(step):06d}.pt"
    if not path.exists():
        raise FileNotFoundError(str(path))

    obj = torch.load(path, map_location=map_device)
    if not isinstance(obj, dict) or "state" not in obj:
        raise ValueError(f"Invalid step file format: {path}")

    t = obj["state"]

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        if max_samples > t.shape[0]:
            raise ValueError(
                f"requested {max_samples} samples, but stored tensor has only {t.shape[0]}"
            )
        t = t[:max_samples]

    if out_dtype is not None and t.dtype != out_dtype:
        t = t.to(out_dtype)

    return t


@torch.no_grad()
def iter_load_states_from_folder(
    folder: str,
    steps: Iterable[int],
    *,
    map_device: torch.device | str = "cpu",
    out_dtype: torch.dtype | None = None,
    return_step: bool = True,
    skip_missing: bool = True,
    max_samples: int | None = None,
):
    for s in steps:
        step_i = int(s)
        path = Path(folder) / f"step_{step_i:06d}.pt"
        if not path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(str(path))

        x_s = load_state_from_folder(
            folder,
            step_i,
            map_device=map_device,
            out_dtype=out_dtype,
            max_samples=max_samples,
        )
        yield (step_i, x_s) if return_step else x_s