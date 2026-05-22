import json
import math
from pathlib import Path

from swap.src.utils.yaml_utils import load_yaml


def canonicalize(obj):
    if isinstance(obj, dict):
        return {k: canonicalize(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [canonicalize(x) for x in obj]
    return obj


def make_updates_signature(updates):
    return json.dumps(canonicalize(updates), sort_keys=True, ensure_ascii=True)


def is_valid_number(x):
    try:
        v = float(x)
        return not math.isnan(v)
    except Exception:
        return False


def load_valid_records(trials_jsonl: Path):
    records = []

    with open(trials_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception:
                continue

            rmse = rec.get("rmse")
            updates = rec.get("updates")
            returncode = rec.get("returncode", 1)
            cfg_file = rec.get("cfg_file")

            if rmse is None or updates is None or cfg_file is None:
                continue
            if returncode != 0:
                continue
            if not is_valid_number(rmse):
                continue
            if not Path(cfg_file).exists():
                continue

            rec["_updates_sig"] = make_updates_signature(updates)
            records.append(rec)

    return records


def dedup_by_setting_keep_best(records):
    best_by_sig = {}

    for rec in records:
        sig = rec["_updates_sig"]
        if sig not in best_by_sig:
            best_by_sig[sig] = rec
            continue

        if float(rec["rmse"]) < float(best_by_sig[sig]["rmse"]):
            best_by_sig[sig] = rec

    return list(best_by_sig.values())


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