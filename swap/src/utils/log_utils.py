import math
from pathlib import Path

# text input/ouput helper

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


def stringify(v):
    if isinstance(v, list):
        return "[" + ", ".join(str(x) for x in v) + "]"
    if v is None:
        return ""
    return str(v)


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


def shorten_param_name(name: str) -> str:
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


def collect_param_columns(candidate_summaries):
    cols = set()
    for item in candidate_summaries:
        flat = flatten_dict(item.get("updates", {}))
        cols.update(flat.keys())
    return sorted(cols)


def write_txt_table(path: Path, headers, rows, metric_cols=None):
    metric_cols = metric_cols or set()

    widths = {h: len(str(h)) for h in headers}
    for row in rows:
        for h in headers:
            if h in metric_cols:
                cell = format_metric(row.get(h, ""))
            else:
                cell = stringify(row.get(h, ""))
            widths[h] = max(widths[h], len(cell))

    def fmt(row):
        cells = []
        for h in headers:
            if h in metric_cols:
                cell = format_metric(row.get(h, ""))
            else:
                cell = stringify(row.get(h, ""))
            cells.append(cell.ljust(widths[h]))
        return " | ".join(cells)

    with open(path, "w", encoding="utf-8") as f:
        f.write(fmt({h: h for h in headers}) + "\n")
        f.write("-+-".join("-" * widths[h] for h in headers) + "\n")
        for row in rows:
            f.write(fmt(row) + "\n")


def build_candidate_rows(candidate_summaries):
    param_cols = collect_param_columns(candidate_summaries)

    headers = (
        ["final_rank"]
        + [shorten_param_name(c) for c in param_cols]
        + [
            "source_run",
            "source_rmse",
            "num_seeds",
            "rmse_mean",
            "rmse_std",
            "csi_mean",
            "csi_std",
            "wallclock_mean",
            "wallclock_std",
            "candidate_name",
        ]
    )

    rows = []
    for idx, item in enumerate(candidate_summaries, start=1):
        flat = flatten_dict(item.get("updates", {}))
        row = {
            "final_rank": idx,
            "source_run": item.get("source_run_name"),
            "source_rmse": item.get("source_rmse"),
            "num_seeds": item.get("num_runs"),
            "rmse_mean": item.get("rmse_mean"),
            "rmse_std": item.get("rmse_std"),
            "csi_mean": item.get("csi_mean"),
            "csi_std": item.get("csi_std"),
            "wallclock_mean": item.get("wallclock_mean"),
            "wallclock_std": item.get("wallclock_std"),
            "candidate_name": item.get("candidate_name"),
        }
        for c in param_cols:
            row[shorten_param_name(c)] = flat.get(c, "")
        rows.append(row)

    return headers, rows


def write_live_summary(summary_txt: Path, candidate_summaries):
    if not candidate_summaries:
        return

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
    write_txt_table(summary_txt, headers, rows, metric_cols=metric_cols)