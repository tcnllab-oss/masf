from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils import EntryNotFoundError

from dynamics.kolmogorov import KolmogorovFlow
from utils.dataloder import (save_trajectory_by_step, iter_load_states_from_folder,
)

def kolmogorov_flow(
    dim,
    num_samples,
    step,
    *,
    device="cpu",
    repo_id="eunbii1/Kolmogorov_flow",
    reynolds=1e3,
    dt=0.2,
    seed=42,
):

    filename = (
        f"dim{dim}/"
        f"num_samples{num_samples}/"
        f"step_{step:06d}.pt"
    )

    # -------------------------
    # Try HuggingFace dataset
    # -------------------------

    try:

        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
        )

        print(
            "[kolmogorov_flow] "
            f"Loaded from HF: {filename}"
        )

        x=torch.load(
            path,
            map_location=device,
        )
        return x["state"]

    except (
        EntryNotFoundError,
        RepositoryNotFoundError,
        FileNotFoundError,
    ):

        print(
            "[kolmogorov_flow] "
            "HF dataset missing → generating."
        )

    # -------------------------
    # Generate locally
    # -------------------------

    dynamics = KolmogorovFlow(
        grid_size=dim,
        reynolds=reynolds,
        dt=dt,
        seed=seed,
    )

    x0 = dynamics.prior(
        n_sample=num_samples,
    ).to(device)

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    dataset_root = (
        project_root
        / "datasets"
        / "kolmogorov"
    )

    out_dir = (
        dataset_root
        / f"dim{dim}_num_samples{num_samples}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    step_path = (
        out_dir
        / f"step_{step:06d}.pt"
    )

    if not step_path.exists():

        save_trajectory_by_step(
            dynamics,
            x0=x0,
            steps=[step],
            out_dir=str(out_dir),
            save_dtype="fp32",
            cpu_store=True,
            overwrite=False,
            meta_extra={
                "dim": dim,
                "num_samples": num_samples,
                "step": step,
            },
        )

    _, prior = next(
        iter(
            iter_load_states_from_folder(
                str(out_dir),
                steps=[step],
                map_device=device,
                out_dtype=torch.float32,
                return_step=True,
                max_samples=num_samples,
            )
        )
    )

    return prior
