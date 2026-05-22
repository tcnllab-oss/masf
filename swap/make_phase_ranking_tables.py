import argparse
import json
import math
from pathlib import Path

from swap.src.utils.phase_utils import ALL_PHASES, PHASE_ORDER


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def safe_float(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def flatten_dict(d, prefix=""):
    out = {}
    if not isinstance(d, dict):
        return out

    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def stringify(v):
    if isinstance(v, list):
        return "[" + ", ".join(str(x) for x in v) + "]"
    if v is None:
        return ""
    return str(v)


def format_metric(v):
    if v is None:
        return ""
    try:
        x = float(v)
        if math.isnan(x):
            return ""
        text = f"{x:.4f}"
        return text.rstrip("0").rstrip(".")
    except Exception:
        return str(v)


def shorten_param_name(name: str):
    mapping = {
        "measurement.same_normalization": "same_normalization",
        "measurement.normalization_form": "normalization_form",
        "measurement.stats_mode": "stats_mode",
        "measurement.stats_update_mode": "stats_update_mode",
        "measurement.momentum": "momentum",
        "model.model_channels": "model_channels",
        "model.num_res_blocks": "num_res_blocks",
        "model.attention_resolutions": "attention_resolutions",
        "model.channel_mult": "channel_mult",
        "pretrain.batch_size": "batch_size",
        "pretrain.epoch": "epoch",
        "pretrain.lr": "lr",
        "sample.nfe": "nfe",
        "sample.s_scale_min": "s_scale_min",
        "sample.s_scale_max": "s_scale_max",
        "sample.s_scale_power": "s_scale_power",
        "sample.g_scale_min": "g_scale_min",
        "sample.g_scale_max": "g_scale_max",
        "sample.g_scale_power": "g_scale_power",
        "sample.terminal_time": "terminal_time",
        "train.batch_size": "batch_size",
        "train.lr": "lr",
        "train.online.warmup_steps": "warmup_steps",
        "train.online.full_epoch": "full_epoch",
        "train.online.ft_epoch": "ft_epoch",
        "train.online.ft_lr_scale": "ft_lr_scale",
        "method.name": "method_name",
        "method.enkf_mode": "enkf_mode",
        "method.inflation": "inflation",
        "method.eps": "eps",
        "method.loc_radius": "loc_radius",
        "method.train": "method_train",
        "method.sample": "method_sample",
        "method.stochastic": "stochastic",
    }
    return mapping.get(name, name)


def collect_param_columns(valid_rows):
    cols = set()
    for row in valid_rows:
        flat = flatten_dict(row.get("updates", {}))
        cols.update(flat.keys())
    return sorted(cols)


def build_phase_table_from_trials_file(trials_path: Path):
    raw_rows = load_jsonl(trials_path)

    valid_rows = []
    for row in raw_rows:
        rmse = safe_float(row.get("rmse"))
        if rmse is None:
            continue
        if row.get("returncode", 1) != 0:
            continue

        row["_rmse"] = rmse
        row["_real"] = safe_float(row.get("real", row.get("wallclock")))
        row["_user"] = safe_float(row.get("user"))
        row["_sys"] = safe_float(row.get("sys"))
        valid_rows.append(row)

    valid_rows.sort(key=lambda x: x["_rmse"])
    param_cols = collect_param_columns(valid_rows)

    headers = ["rank"] + [shorten_param_name(c) for c in param_cols] + [
        "final_rmse",
        "real_seconds",
        "user_seconds",
        "sys_seconds",
        "run_name",
    ]

    table_rows = []
    for idx, row in enumerate(valid_rows, start=1):
        flat = flatten_dict(row.get("updates", {}))
        out = {"rank": idx}
        for c in param_cols:
            out[shorten_param_name(c)] = flat.get(c, "")
        out["final_rmse"] = row["_rmse"]
        out["real_seconds"] = row["_real"]
        out["user_seconds"] = row["_user"]
        out["sys_seconds"] = row["_sys"]
        out["run_name"] = row.get("run_name", "")
        table_rows.append(out)

    return headers, table_rows


def write_txt_table(path: Path, headers, rows):
    metric_cols = {
        "best_rmse",
        "final_rmse",
        "real_seconds",
        "user_seconds",
        "sys_seconds",
        "wallclock",
    }

    widths = {h: len(str(h)) for h in headers}
    for row in rows:
        for h in headers:
            cell = format_metric(row.get(h, "")) if h in metric_cols else stringify(row.get(h, ""))
            widths[h] = max(widths[h], len(cell))

    def fmt(row):
        cells = []
        for h in headers:
            cell = format_metric(row.get(h, "")) if h in metric_cols else stringify(row.get(h, ""))
            cells.append(cell.ljust(widths[h]))
        return " | ".join(cells)

    with open(path, "w", encoding="utf-8") as f:
        f.write(fmt({h: h for h in headers}) + "\n")
        f.write("-+-".join("-" * widths[h] for h in headers) + "\n")
        for row in rows:
            f.write(fmt(row) + "\n")


def find_phase_trial_files(root_dir: Path, phase: str):
    if phase == "baseline_tuning":
        baseline_root = root_dir / "baseline_tuning"
        if not baseline_root.exists():
            return []
        return sorted(baseline_root.glob(f"*/{phase}_trials.jsonl"))

    trials_path = root_dir / phase / f"{phase}_trials.jsonl"
    if trials_path.exists():
        return [trials_path]
    return []


def extract_num_samples_from_phase_dir_label(phase_dir_label: str):
    tail = phase_dir_label.split("/")[-1]
    prefix = "num_samples_"
    if tail.startswith(prefix):
        return tail[len(prefix):]
    return tail


def baseline_summary_headers_from_rows(summary_rows):
    headers = ["num_samples", "best_rmse", "best_run_name", "n_valid_trials", "wallclock"]

    has_enkf_mode = any(str(r.get("enkf_mode", "")).strip() for r in summary_rows)
    has_loc_radius = any(str(r.get("loc_radius", "")).strip() for r in summary_rows)

    if has_enkf_mode:
        headers.append("enkf_mode")

    headers += ["inflation", "eps"]

    if has_loc_radius:
        headers.append("loc_radius")

    return headers


def make_baseline_summary_row(best_row, phase_dir_label, n_valid_trials):
    return {
        "num_samples": extract_num_samples_from_phase_dir_label(phase_dir_label),
        "best_rmse": best_row.get("final_rmse", ""),
        "best_run_name": best_row.get("run_name", ""),
        "n_valid_trials": n_valid_trials,
        "wallclock": best_row.get("real_seconds", best_row.get("wallclock", "")),
        "enkf_mode": best_row.get("enkf_mode", ""),
        "inflation": best_row.get("inflation", ""),
        "eps": best_row.get("eps", ""),
        "loc_radius": best_row.get("loc_radius", ""),
    }


def make_normal_summary_row(best_row, phase_name, phase_dir_label, n_valid_trials):
    return {
        "phase": phase_name,
        "phase_dir": phase_dir_label,
        "best_rmse": best_row.get("final_rmse", ""),
        "best_run_name": best_row.get("run_name", ""),
        "n_valid_trials": n_valid_trials,
        "wallclock": best_row.get("real_seconds", best_row.get("wallclock", "")),
    }


def run_phase_ranking_tables(root_dir: str, phase: str = None, from_phase: str = None, until: str = None):
    root_dir = Path(root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    if phase is None and until is None:
        raise ValueError("Provide either phase or until")
    if phase is not None and until is not None:
        raise ValueError("Use either phase or until, not both")
    if phase is not None and from_phase is not None:
        raise ValueError("from_phase cannot be used with phase")
    if phase == "baseline_tuning" and until is not None:
        raise ValueError("baseline_tuning must be run with phase only")

    summary_dir = root_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if phase is not None:
        phases_to_run = [phase]
    else:
        start_idx = 0 if from_phase is None else PHASE_ORDER.index(from_phase)
        end_idx = PHASE_ORDER.index(until)

        if start_idx > end_idx:
            raise ValueError(
                f"from_phase ({from_phase}) must be earlier than or equal to until ({until})"
            )

        phases_to_run = PHASE_ORDER[start_idx:end_idx + 1]

    summary_rows = []

    for phase_name in phases_to_run:
        trials_files = find_phase_trial_files(root_dir, phase_name)
        if not trials_files:
            print(f"[PHASE_RANKING] skip missing phase: {phase_name}")
            continue

        for trials_path in trials_files:
            phase_dir = trials_path.parent
            headers, rows = build_phase_table_from_trials_file(trials_path)
            if not rows:
                print(f"[PHASE_RANKING] skip empty valid rows: {trials_path}")
                continue

            if phase_name == "baseline_tuning":
                suffix_name = phase_dir.name
                out_name = f"{phase_name}__{suffix_name}_ranking.txt"
                phase_dir_label = f"{phase_name}/{suffix_name}"
            else:
                out_name = f"{phase_name}_ranking.txt"
                phase_dir_label = phase_name

            out_txt = summary_dir / out_name
            write_txt_table(out_txt, headers, rows)

            best_row = rows[0]
            if phase_name == "baseline_tuning":
                summary_rows.append(
                    make_baseline_summary_row(
                        best_row=best_row,
                        phase_dir_label=phase_dir_label,
                        n_valid_trials=len(rows),
                    )
                )
            else:
                summary_rows.append(
                    make_normal_summary_row(
                        best_row=best_row,
                        phase_name=phase_name,
                        phase_dir_label=phase_dir_label,
                        n_valid_trials=len(rows),
                    )
                )

    if phase == "baseline_tuning":
        summary_headers = baseline_summary_headers_from_rows(summary_rows)
        summary_rows.sort(
            key=lambda x: int(x["num_samples"]) if str(x.get("num_samples", "")).isdigit() else str(x.get("num_samples", ""))
        )
    else:
        summary_headers = ["phase", "phase_dir", "best_rmse", "best_run_name", "n_valid_trials", "wallclock"]
        phase_rank = {name: i for i, name in enumerate(PHASE_ORDER)}
        summary_rows.sort(key=lambda x: phase_rank.get(x.get("phase", ""), 999))

    write_txt_table(summary_dir / "phase_ranking_summary.txt", summary_headers, summary_rows)

    print("=" * 80)
    print(f"[PHASE_RANKING] root_dir    : {root_dir}")
    print(f"[PHASE_RANKING] summary_dir : {summary_dir}")
    print(f"[PHASE_RANKING] phases      : {phases_to_run}")
    print(f"[PHASE_RANKING] saved       : {summary_dir / 'phase_ranking_summary.txt'}")
    for row in summary_rows:
        if phase == "baseline_tuning":
            out_txt = summary_dir / f"baseline_tuning__num_samples_{row['num_samples']}_ranking.txt"
        else:
            out_txt = summary_dir / f"{row['phase']}_ranking.txt"
        if out_txt.exists():
            print(f"[PHASE_RANKING] saved       : {out_txt}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        required=True,
        help="e.g. /data3/dwkim/masf/0414/ours/kolmogorov_128/grid_mask/stride_10",
    )
    parser.add_argument("--phase", choices=ALL_PHASES, default=None)
    parser.add_argument("--from_phase", "--from", dest="from_phase", choices=PHASE_ORDER, default=None)
    parser.add_argument("--until", choices=PHASE_ORDER, default=None)

    args = parser.parse_args()

    run_phase_ranking_tables(
        root_dir=args.root_dir,
        phase=args.phase,
        from_phase=args.from_phase,
        until=args.until,
    )


if __name__ == "__main__":
    main()