import shutil
import statistics
import subprocess
from copy import deepcopy
from pathlib import Path

from swap.src.utils.yaml_utils import load_yaml, save_yaml, set_nested
from swap.src.utils.io_utils import parse_log_metrics, append_text, init_text_file
from swap.src.utils.log_utils import (
    write_txt_table,
    build_candidate_rows,
    write_live_summary,
)
from swap.src.utils.result_utils import (
    is_valid_number,
    load_valid_records,
    dedup_by_setting_keep_best,
    load_valid_seed_result,
)
from swap.src.utils.cleanup import cleanup_candidate_seed_outputs


def resolve_phase_dir(root_dir, phase):
    return Path(root_dir).resolve() / phase


def summarize_seed_records(seed_records):
    rmses = [r["rmse"] for r in seed_records if is_valid_number(r.get("rmse"))]
    csis = [r["csi"] for r in seed_records if is_valid_number(r.get("csi"))]
    walls = [r["wallclock"] for r in seed_records if is_valid_number(r.get("wallclock"))]

    return {
        "num_runs": len(seed_records),
        "num_valid_rmse": len(rmses),
        "rmse_mean": statistics.mean(rmses) if rmses else float("nan"),
        "rmse_std": statistics.stdev(rmses) if len(rmses) >= 2 else 0.0 if rmses else float("nan"),
        "rmse_min": min(rmses) if rmses else float("nan"),
        "rmse_max": max(rmses) if rmses else float("nan"),
        "csi_mean": statistics.mean(csis) if csis else float("nan"),
        "csi_std": statistics.stdev(csis) if len(csis) >= 2 else 0.0 if csis else float("nan"),
        "wallclock_mean": statistics.mean(walls) if walls else float("nan"),
        "wallclock_std": statistics.stdev(walls) if len(walls) >= 2 else 0.0 if walls else float("nan"),
    }


def run_single_seed(project_dir: Path, base_cfg: dict, rec: dict, candidate_name: str, rank: int, seed: int, run_dir: Path):
    outputs_dir = run_dir / "outputs"
    cfg_path = run_dir / "config.yaml"
    log_path = run_dir / "run.log"
    result_path = run_dir / "result.yaml"

    run_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    cfg = deepcopy(base_cfg)
    set_nested(cfg, "system.seed", seed)
    cfg.setdefault("exp", {})
    cfg["exp"]["workdir_root"] = str(outputs_dir)
    save_yaml(cfg, cfg_path)

    cmd = [
        "python",
        str(project_dir / "main.py"),
        "--config", str(cfg_path),
        "--exp", "run",
    ]

    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            ["/usr/bin/time", "-p"] + cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
            cwd=project_dir,
        )

    rmse, csi, real, user, sys_t = parse_log_metrics(log_path)

    result = {
        "candidate_name": candidate_name,
        "candidate_rank": rank,
        "seed": seed,
        "rmse": rmse,
        "csi": csi,
        "wallclock": real,
        "user": user,
        "sys": sys_t,
        "returncode": proc.returncode,
        "config_path": str(cfg_path),
        "log_path": str(log_path),
        "outputs_dir": str(outputs_dir),
        "source_run_name": rec.get("run_name"),
        "source_rmse": rec.get("rmse"),
        "source_cfg_file": str(rec["cfg_file"]),
        "updates": rec.get("updates"),
    }
    save_yaml(result, result_path)
    return result


def run_seed_eval_for_phase(root_dir, phase, top_k, seeds, rerun, cleanup=False):
    phase_dir = resolve_phase_dir(root_dir, phase)
    project_dir = Path(__file__).resolve().parent.parent
    trials_jsonl = phase_dir / f"{phase}_trials.jsonl"

    if not phase_dir.exists():
        raise FileNotFoundError(f"Phase dir not found: {phase_dir}")
    if not trials_jsonl.exists():
        raise FileNotFoundError(f"Trials file not found: {trials_jsonl}")

    all_records = load_valid_records(trials_jsonl)
    if not all_records:
        raise RuntimeError(f"No valid trial records found in: {trials_jsonl}")

    unique_records = dedup_by_setting_keep_best(all_records)
    unique_records.sort(key=lambda x: float(x["rmse"]))
    top_records = unique_records[:top_k]

    seed_eval_root = phase_dir / "seed_eval_topk"

    if rerun and seed_eval_root.exists():
        shutil.rmtree(seed_eval_root)

    seed_eval_root.mkdir(parents=True, exist_ok=True)
    experiments_dir = seed_eval_root / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    progress_txt = seed_eval_root / "progress.txt"
    summary_txt = seed_eval_root / "summary.txt"

    init_text_file(progress_txt, "# seed eval progress")
    init_text_file(summary_txt, "# seed eval summary")

    candidate_summaries = []

    append_text(
        progress_txt,
        "[START] phase={} root_dir={} phase_dir={} trials_jsonl={} top_k={} seeds={} seed_eval_root={} experiments_dir={} cleanup={}".format(
            phase,
            Path(root_dir).resolve(),
            phase_dir,
            trials_jsonl,
            top_k,
            seeds,
            seed_eval_root,
            experiments_dir,
            cleanup,
        )
    )

    print("=" * 80)
    print(f"[SEED_EVAL] phase    : {phase}")
    print(f"[SEED_EVAL] root_dir : {Path(root_dir).resolve()}")
    print(f"[SEED_EVAL] top_k    : {top_k}")
    print(f"[SEED_EVAL] seeds    : {seeds}")
    print(f"[SEED_EVAL] cleanup  : {cleanup}")
    print("=" * 80)

    for rank, rec in enumerate(top_records, start=1):
        src_cfg = Path(rec["cfg_file"])
        if not src_cfg.exists():
            print(f"[WARN] Missing candidate cfg: {src_cfg}")
            append_text(progress_txt, f"[WARN] missing candidate cfg: {src_cfg}")
            continue

        candidate_name = f"candidate_{rank:02d}"
        candidate_root = experiments_dir / candidate_name
        candidate_root.mkdir(parents=True, exist_ok=True)

        base_candidate_cfg = candidate_root / "base_candidate.yaml"
        if rerun or not base_candidate_cfg.exists():
            shutil.copy2(src_cfg, base_candidate_cfg)

        append_text(
            progress_txt,
            "[CANDIDATE_START] rank={} candidate={} source_run={} source_rmse={} candidate_dir={}".format(
                rank,
                candidate_name,
                rec.get("run_name"),
                rec.get("rmse"),
                candidate_root,
            )
        )

        print(f"[SEED_EVAL] candidate={candidate_name} source_run={rec.get('run_name')} source_rmse={rec.get('rmse')}")

        base_cfg = load_yaml(base_candidate_cfg)
        seed_records = []

        for seed in seeds:
            run_dir = candidate_root / f"seed_{seed:04d}"

            if rerun and run_dir.exists():
                shutil.rmtree(run_dir)

            existing_result = None if rerun else load_valid_seed_result(run_dir / "result.yaml")
            if existing_result is not None:
                seed_records.append(existing_result)
                append_text(
                    progress_txt,
                    "[SEED_REUSE] candidate={} seed={} rmse={} csi={} wallclock={}".format(
                        candidate_name,
                        seed,
                        existing_result.get("rmse"),
                        existing_result.get("csi"),
                        existing_result.get("wallclock"),
                    )
                )
                print(
                    f"[SEED_EVAL] reuse candidate={candidate_name} seed={seed} rmse={existing_result.get('rmse')}"
                )
                continue

            if run_dir.exists():
                shutil.rmtree(run_dir)

            result = run_single_seed(
                project_dir=project_dir,
                base_cfg=base_cfg,
                rec=rec,
                candidate_name=candidate_name,
                rank=rank,
                seed=seed,
                run_dir=run_dir,
            )
            seed_records.append(result)

            append_text(
                progress_txt,
                "[SEED_DONE] candidate={} seed={} rmse={} csi={} wallclock={} user={} sys={} returncode={}".format(
                    candidate_name,
                    seed,
                    result["rmse"],
                    result["csi"],
                    result["wallclock"],
                    result["user"],
                    result["sys"],
                    result["returncode"],
                )
            )
            print(f"[SEED_EVAL] done candidate={candidate_name} seed={seed} rmse={result['rmse']}")

        summary = summarize_seed_records(seed_records)
        summary.update({
            "candidate_name": candidate_name,
            "candidate_rank": rank,
            "source_run_name": rec.get("run_name"),
            "source_rmse": rec.get("rmse"),
            "source_cfg_file": str(src_cfg),
            "base_candidate_cfg": str(base_candidate_cfg),
            "updates": rec.get("updates"),
            "seed_records": seed_records,
        })

        if cleanup:
            cleanup_stats = cleanup_candidate_seed_outputs(candidate_root)
            summary["cleanup"] = cleanup_stats

            append_text(
                progress_txt,
                "[CANDIDATE_CLEANUP] candidate={} seed_count={} removed_items={} skipped={}".format(
                    candidate_name,
                    cleanup_stats["seed_count"],
                    cleanup_stats["removed_items"],
                    cleanup_stats["skipped"],
                )
            )
            print(
                "[SEED_EVAL] cleanup candidate={} seed_count={} removed_items={}".format(
                    candidate_name,
                    cleanup_stats["seed_count"],
                    cleanup_stats["removed_items"],
                )
            )

        save_yaml(summary, candidate_root / "summary.yaml")

        candidate_summaries.append(summary)
        candidate_summaries.sort(key=lambda x: float(x["rmse_mean"]))
        write_live_summary(summary_txt, candidate_summaries)

        append_text(
            progress_txt,
            "[CANDIDATE_DONE] candidate={} rmse_mean={} rmse_std={} csi_mean={} csi_std={} wallclock_mean={} wallclock_std={}".format(
                candidate_name,
                summary["rmse_mean"],
                summary["rmse_std"],
                summary["csi_mean"],
                summary["csi_std"],
                summary["wallclock_mean"],
                summary["wallclock_std"],
            )
        )

    candidate_summaries.sort(key=lambda x: float(x["rmse_mean"]))

    final_summary = {
        "root_dir": str(Path(root_dir).resolve()),
        "phase": phase,
        "phase_dir": str(phase_dir),
        "top_k": top_k,
        "seeds": seeds,
        "seed_eval_root": str(seed_eval_root),
        "experiments_dir": str(experiments_dir),
        "num_candidates": len(candidate_summaries),
        "cleanup": cleanup,
        "candidates": candidate_summaries,
    }

    save_yaml(final_summary, seed_eval_root / "ranking_summary.yaml")

    headers, rows = build_candidate_rows(candidate_summaries)
    metric_cols = {
        "source_rmse",
        "rmse_mean",
        "rmse_std",
        "csi_mean",
        "csi_std",
        "wallclock_mean",
        "wallclock_std",
    }
    write_txt_table(seed_eval_root / "ranking_summary.txt", headers, rows, metric_cols=metric_cols)
    write_live_summary(summary_txt, candidate_summaries)

    append_text(progress_txt, f"[END] phase={phase} num_candidates={len(candidate_summaries)}")
    print(f"[SEED_EVAL] done: phase={phase} saved={seed_eval_root}")