import math
import shutil
import statistics
import subprocess
from copy import deepcopy
from pathlib import Path

from swap.src.utils.yaml_utils import load_yaml, save_yaml, set_nested
from swap.src.utils.io_utils import parse_log_metrics
from swap.src.post.post_common import (
    append_text,
    init_text_file,
    write_txt_table,
    write_csv_table,
    build_ranking_rows,
    format_metric_value,
    expand_plan_grid,
)


def is_valid_number(x):
    try:
        v = float(x)
        return not math.isnan(v)
    except Exception:
        return False


def load_valid_seed_result(result_path: Path):
    if not result_path.exists():
        return None

    try:
        result = load_yaml(result_path)
    except Exception:
        return None

    if not isinstance(result, dict):
        return None
    if result.get("returncode", 1) != 0:
        return None
    if not is_valid_number(result.get("rmse")):
        return None
    return result


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


def resolve_project_root() -> Path:
    cur = Path(__file__).resolve()
    for p in [cur] + list(cur.parents):
        if p.name == "swap":
            return p.parent
    raise RuntimeError(f"Could not find project root from: {cur}")


def infer_measurement_root_from_base_yaml(base_yaml_path: Path) -> Path:
    """
    Example 1:
      .../stride_10/finetuning/best_finetune.yaml
      -> .../stride_10

    Example 2:
      .../stride_10/baseline_tuning/num_samples_300/best_enkf.yaml
      -> .../stride_10
    """
    base_yaml_path = base_yaml_path.resolve()
    parent = base_yaml_path.parent
    grandparent = parent.parent

    if grandparent.name == "baseline_tuning" and parent.name.startswith("num_samples_"):
        return grandparent.parent

    return parent.parent


def get_baseline_case_name_from_base_yaml(base_yaml_path: Path) -> str | None:
    base_yaml_path = base_yaml_path.resolve()
    parent = base_yaml_path.parent
    grandparent = parent.parent

    if grandparent.name == "baseline_tuning" and parent.name.startswith("num_samples_"):
        return parent.name

    return None


def resolve_phase_root(base_yaml_path: Path, phase_name: str) -> Path:
    measurement_root = infer_measurement_root_from_base_yaml(base_yaml_path)
    return measurement_root / "post" / phase_name


def resolve_output_dir(base_yaml_path: Path, phase_name: str) -> Path:
    phase_root = resolve_phase_root(base_yaml_path, phase_name)
    baseline_case_name = get_baseline_case_name_from_base_yaml(base_yaml_path)

    if baseline_case_name is not None:
        return phase_root / baseline_case_name

    return phase_root


def write_live_summary(summary_txt: Path, case_summaries):
    headers, rows = build_ranking_rows(case_summaries)
    write_txt_table(summary_txt, headers, rows)


def slugify_case_token(x):
    s = str(x).strip().lower()
    out = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def make_case_name(overrides: dict, idx: int) -> str:
    if not overrides:
        return f"case_{idx:04d}"

    parts = []
    for key, value in sorted(overrides.items()):
        short_key = key.split(".")[-1]
        short_key = slugify_case_token(short_key)
        short_val = slugify_case_token(value)
        parts.append(f"{short_key}_{short_val}")

    name = "__".join(parts)
    if not name:
        return f"case_{idx:04d}"
    return name


def ensure_unique_case_name(case_name: str, used_names: set[str], idx: int) -> str:
    if case_name not in used_names:
        used_names.add(case_name)
        return case_name

    alt = f"{case_name}__case_{idx:04d}"
    used_names.add(alt)
    return alt


def write_case_summary_txt(case_dir: Path, summary: dict):
    seed_records = summary.get("seed_records", [])
    valid_records = [r for r in seed_records if is_valid_number(r.get("rmse"))]

    if valid_records:
        best_record = min(valid_records, key=lambda r: float(r["rmse"]))
        best_seed = best_record.get("seed")
        best_rmse = best_record.get("rmse")
        best_result_yaml = case_dir / "configs" / f"seed_{int(best_seed):04d}" / "result.yaml"
    else:
        best_seed = ""
        best_rmse = ""
        best_result_yaml = ""

    lines = [
        f"case_name: {summary.get('case_name', '')}",
        f"n_seeds: {summary.get('num_runs', '')}",
        f"n_valid: {summary.get('num_valid_rmse', '')}",
        f"RMSE (mean ± std): {format_metric_value(summary.get('rmse_mean'))} ± {format_metric_value(summary.get('rmse_std'))}",
        f"CSI (mean ± std): {format_metric_value(summary.get('csi_mean'))} ± {format_metric_value(summary.get('csi_std'))}",
        f"wallclock (mean ± std): {format_metric_value(summary.get('wallclock_mean'))} ± {format_metric_value(summary.get('wallclock_std'))}",
        f"best_seed: {best_seed}",
        f"best_rmse: {best_rmse}",
        f"best_result_yaml: {best_result_yaml}",
    ]

    overrides = summary.get("overrides", {})
    for k, v in overrides.items():
        lines.append(f"{k}: {v}")

    (case_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_baseline_local_rankings(phase_root: Path):
    all_case_summaries = []

    if not phase_root.exists():
        return all_case_summaries

    for child in sorted(phase_root.iterdir()):
        if not child.is_dir():
            continue

        local_yaml = child / "ranking_local.yaml"
        if not local_yaml.exists():
            continue

        try:
            data = load_yaml(local_yaml)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        child_cases = data.get("cases", [])
        baseline_case_name = child.name

        for item in child_cases:
            if not isinstance(item, dict):
                continue
            merged = dict(item)
            merged["baseline_case"] = baseline_case_name
            all_case_summaries.append(merged)

    return all_case_summaries


def write_baseline_aggregate_outputs(phase_root: Path, phase_name: str, seeds):
    all_case_summaries = _collect_baseline_local_rankings(phase_root)
    if not all_case_summaries:
        return

    try:
        all_case_summaries.sort(key=lambda x: float(x.get("rmse_mean", float("inf"))))
    except Exception:
        pass

    ranking = {
        "phase": phase_name,
        "output_dir": str(phase_root),
        "seeds": seeds,
        "num_cases": len(all_case_summaries),
        "cases": all_case_summaries,
    }
    save_yaml(ranking, phase_root / "ranking.yaml")

    headers, rows = build_ranking_rows(all_case_summaries)
    write_txt_table(phase_root / "ranking.txt", headers, rows)
    write_csv_table(phase_root / "ranking.csv", headers, rows)
    write_txt_table(phase_root / "summary.txt", headers, rows)

    progress_lines = [
        f"phase: {phase_name}",
        f"n_cases: {len(all_case_summaries)}",
        f"n_children: {sum(1 for p in phase_root.iterdir() if p.is_dir())}",
    ]
    (phase_root / "progress.txt").write_text("\n".join(progress_lines) + "\n", encoding="utf-8")


def deep_merge_dict(dst: dict, src: dict) -> dict:

    for k, v in src.items():

        if isinstance(v, dict) and isinstance(dst.get(k), dict):

            deep_merge_dict(dst[k], v)

        else:

            dst[k] = deepcopy(v)

    return dst

def run_plan(plan: dict, force_rerun: bool = False):
    phase_name = plan["phase"]
    base_yaml_path = Path(plan["base_yaml"]).resolve()
    seeds = list(plan["seeds"])
    grid = plan["grid"]

    if not base_yaml_path.exists():
        raise FileNotFoundError(f"base_yaml not found: {base_yaml_path}")

    phase_root = resolve_phase_root(base_yaml_path, phase_name)
    output_dir = resolve_output_dir(base_yaml_path, phase_name)
    baseline_case_name = get_baseline_case_name_from_base_yaml(base_yaml_path)
    is_baseline_local_run = baseline_case_name is not None

    if force_rerun and output_dir.exists():
        shutil.rmtree(output_dir)

    phase_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_txt = output_dir / "progress.txt"
    summary_txt = output_dir / "summary.txt"

    init_text_file(progress_txt, "# post eval progress")
    init_text_file(summary_txt, "# post eval summary")

    base_cfg = load_yaml(base_yaml_path)
    cases = expand_plan_grid(grid)
    project_dir = resolve_project_root()

    append_text(
        progress_txt,
        f"START phase={phase_name} base_yaml={base_yaml_path} output_dir={output_dir} "
        f"num_cases={len(cases)} seeds={seeds}"
    )

    all_case_summaries = []
    used_case_names = set()

    print("=" * 80)
    print(f"[POST_EVAL] phase      : {phase_name}")
    print(f"[POST_EVAL] base_yaml  : {base_yaml_path}")
    print(f"[POST_EVAL] project_dir: {project_dir}")
    print(f"[POST_EVAL] phase_root : {phase_root}")
    print(f"[POST_EVAL] output_dir : {output_dir}")
    print(f"[POST_EVAL] num_cases  : {len(cases)}")
    print(f"[POST_EVAL] seeds      : {seeds}")
    if is_baseline_local_run:
        print(f"[POST_EVAL] baseline_case: {baseline_case_name}")
    print("=" * 80)

    for idx, overrides in enumerate(cases, start=1):
        raw_case_name = make_case_name(overrides, idx)
        case_name = ensure_unique_case_name(raw_case_name, used_case_names, idx)

        print(f"[DEBUG] case={idx} case_name={case_name} overrides={overrides}")

        case_dir = output_dir / case_name
        configs_dir = case_dir / "configs"
        experiments_dir = case_dir / "experiments"

        if force_rerun and case_dir.exists():
            shutil.rmtree(case_dir)

        case_dir.mkdir(parents=True, exist_ok=True)
        configs_dir.mkdir(parents=True, exist_ok=True)
        experiments_dir.mkdir(parents=True, exist_ok=True)

        append_text(progress_txt, f"CASE_START case={case_name} overrides={overrides}")

        cfg = deepcopy(base_cfg)

        # post config(fixed)가 base yaml보다 우선
        for key, value in plan.get("fixed", {}).items():
            set_nested(cfg, key, value)

        # case/grid override가 최종 우선
        for key, value in overrides.items():
            if isinstance(value, list):
                raise ValueError(
                    f"Override value for {key} must be scalar per case, but got list: {value}"
                )
            set_nested(cfg, key, value)

        save_yaml(cfg, case_dir / "config.yaml")
        save_yaml({"case_name": case_name, "overrides": overrides}, case_dir / "case_meta.yaml")

        seed_records = []

        print("-" * 80)
        print(f"[CASE {idx}] {case_name}")
        print(f"  overrides: {overrides}")
        print(f"  case_dir : {case_dir}")

        for seed in seeds:
            seed_cfg_dir = configs_dir / f"seed_{seed:04d}"
            seed_exp_dir = experiments_dir / f"seed_{seed:04d}"

            outputs_dir = seed_exp_dir / "outputs"
            cfg_path = seed_cfg_dir / "config.yaml"
            log_path = seed_exp_dir / "run.log"
            result_path = seed_cfg_dir / "result.yaml"

            if force_rerun and (seed_cfg_dir.exists() or seed_exp_dir.exists()):
                shutil.rmtree(seed_cfg_dir, ignore_errors=True)
                shutil.rmtree(seed_exp_dir, ignore_errors=True)

            existing_result = None if force_rerun else load_valid_seed_result(result_path)
            if existing_result is not None:
                seed_records.append(existing_result)
                append_text(
                    progress_txt,
                    f"SEED_REUSE case={case_name} seed={seed} "
                    f"rmse={format_metric_value(existing_result.get('rmse'))} "
                    f"csi={format_metric_value(existing_result.get('csi'))} "
                    f"wallclock={format_metric_value(existing_result.get('wallclock'))}"
                )
                print(
                    f"  [REUSE seed={seed}] "
                    f"rmse={format_metric_value(existing_result.get('rmse'))} "
                    f"csi={format_metric_value(existing_result.get('csi'))} "
                    f"wallclock={format_metric_value(existing_result.get('wallclock'))}"
                )
                continue

            if seed_cfg_dir.exists():
                shutil.rmtree(seed_cfg_dir, ignore_errors=True)
            if seed_exp_dir.exists():
                shutil.rmtree(seed_exp_dir, ignore_errors=True)

            seed_cfg_dir.mkdir(parents=True, exist_ok=True)
            seed_exp_dir.mkdir(parents=True, exist_ok=True)
            outputs_dir.mkdir(parents=True, exist_ok=True)

            run_cfg = deepcopy(cfg)
            set_nested(run_cfg, "system.seed", seed)

            run_cfg.setdefault("exp", {})
            run_cfg["exp"]["workdir_root"] = str(outputs_dir)

            save_yaml(run_cfg, cfg_path)

            cmd = [
                "python",
                str(project_dir / "main.py"),
                "--config",
                str(cfg_path),
                "--exp",
                "run",
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
            is_valid = (proc.returncode == 0) and is_valid_number(rmse)

            result = {
                "case_name": case_name,
                "seed": seed,
                "rmse": rmse,
                "csi": csi,
                "wallclock": real,
                "user": user,
                "sys": sys_t,
                "returncode": proc.returncode,
                "is_valid": is_valid,
                "config_path": str(cfg_path),
                "log_path": str(log_path),
                "outputs_dir": str(outputs_dir),
                "overrides": overrides,
            }
            save_yaml(result, result_path)
            seed_records.append(result)

            append_text(
                progress_txt,
                f"SEED_DONE case={case_name} seed={seed} "
                f"rmse={format_metric_value(rmse)} "
                f"csi={format_metric_value(csi)} "
                f"wallclock={format_metric_value(real)} "
                f"user={format_metric_value(user)} "
                f"sys={format_metric_value(sys_t)} "
                f"returncode={proc.returncode}"
            )
            print(
                f"  [DONE seed={seed}] "
                f"rmse={format_metric_value(rmse)} "
                f"csi={format_metric_value(csi)} "
                f"wallclock={format_metric_value(real)}"
            )

        summary = summarize_seed_records(seed_records)
        summary.update({
            "case_name": case_name,
            "overrides": overrides,
            "seed_records": seed_records,
        })

        save_yaml(summary, case_dir / "summary.yaml")
        write_case_summary_txt(case_dir, summary)

        all_case_summaries.append(summary)
        all_case_summaries.sort(key=lambda x: float(x["rmse_mean"]))
        write_live_summary(summary_txt, all_case_summaries)

        append_text(
            progress_txt,
            f"CASE_DONE case={case_name} "
            f"rmse_mean={format_metric_value(summary['rmse_mean'])} "
            f"rmse_std={format_metric_value(summary['rmse_std'])} "
            f"csi_mean={format_metric_value(summary['csi_mean'])} "
            f"csi_std={format_metric_value(summary['csi_std'])} "
            f"wallclock_mean={format_metric_value(summary['wallclock_mean'])} "
            f"wallclock_std={format_metric_value(summary['wallclock_std'])}"
        )

    all_case_summaries.sort(key=lambda x: float(x["rmse_mean"]))

    ranking = {
        "phase": phase_name,
        "base_yaml": str(base_yaml_path),
        "output_dir": str(output_dir),
        "seeds": seeds,
        "num_cases": len(all_case_summaries),
        "cases": all_case_summaries,
    }

    if is_baseline_local_run:
        save_yaml(ranking, output_dir / "ranking_local.yaml")

        headers, rows = build_ranking_rows(all_case_summaries)
        write_txt_table(output_dir / "ranking_local.txt", headers, rows)
        write_csv_table(output_dir / "ranking_local.csv", headers, rows)
        write_live_summary(summary_txt, all_case_summaries)

        write_baseline_aggregate_outputs(
            phase_root=phase_root,
            phase_name=phase_name,
            seeds=seeds,
        )

        print("=" * 80)
        print(f"[POST_EVAL] saved local to : {output_dir}")
        print(f"[POST_EVAL] local ranking   : {output_dir / 'ranking_local.txt'}")
        print(f"[POST_EVAL] aggregate rank : {phase_root / 'ranking.txt'}")
        print("=" * 80)
    else:
        save_yaml(ranking, output_dir / "ranking.yaml")

        headers, rows = build_ranking_rows(all_case_summaries)
        write_txt_table(output_dir / "ranking.txt", headers, rows)
        write_csv_table(output_dir / "ranking.csv", headers, rows)
        write_live_summary(summary_txt, all_case_summaries)

        print("=" * 80)
        print(f"[POST_EVAL] saved to  : {output_dir}")
        print(f"[POST_EVAL] ranking   : {output_dir / 'ranking.txt'}")
        print(f"[POST_EVAL] csv       : {output_dir / 'ranking.csv'}")
        print("=" * 80)

    append_text(progress_txt, f"END phase={phase_name} num_cases={len(all_case_summaries)}")