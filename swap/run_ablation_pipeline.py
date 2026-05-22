import argparse
import json
import math
import shutil
import subprocess
from copy import deepcopy
from itertools import product
from pathlib import Path

from swap.src.utils.yaml_utils import load_yaml, save_yaml
from swap.src.utils.io_utils import parse_log_metrics
from swap.src.utils.phase_utils import ALL_PHASES, PHASE_ORDER
from swap.src.utils.plan_utils import (
    resolve_swap_dir,
    resolve_configs_dir,
    get_default_plan_path,
)


def safe_float(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def fmt(x):
    if x is None:
        return ""
    return f"{x:.4f}".rstrip("0").rstrip(".")


def mean_std(values):
    xs = [safe_float(v) for v in values]
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    mean = sum(xs) / len(xs)
    if len(xs) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return mean, math.sqrt(var)


def fmt_mean_std(mean, std):
    if mean is None:
        return ""
    return f"{fmt(mean)} ± {fmt(std)}"


def slugify(value):
    text = str(value).strip().lower()
    out = []
    for ch in text:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    text = "".join(out)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "x"


def stringify_value(value):
    if isinstance(value, list):
        return "-".join(str(v) for v in value)
    return str(value)


def deep_get_by_dotted_key(d, dotted_key, default=None):
    cur = d
    for k in dotted_key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def deep_set(d, dotted_key, value):
    cur = d
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def current_cfg_to_flat(cfg):
    flat = {}

    def _rec(x, prefix=""):
        if isinstance(x, dict):
            for k, v in x.items():
                key = f"{prefix}.{k}" if prefix else k
                _rec(v, key)
        else:
            flat[prefix] = x

    _rec(cfg)
    return flat


def is_active_param(spec, current_flat):
    active_if = spec.get("active_if")
    if not active_if:
        return True
    for cond_key, expected_value in active_if.items():
        if current_flat.get(cond_key) != expected_value:
            return False
    return True


def resolve_output_root(config_path: Path, root_dir: str | None) -> Path:
    if root_dir is not None:
        return Path(root_dir).resolve()
    return config_path.parent.resolve()


def load_phase_param_specs(base_cfg: dict, phase: str, ablation_cfg: dict | None = None) -> dict:
    # 1) config 내부 ablation override 우선
    if ablation_cfg is not None:
        phase_block = ablation_cfg.get(phase, {})
        params = phase_block.get("params")
        if params:
            return params

    # 2) 없으면 default plan 사용
    swap_dir = resolve_swap_dir()
    configs_dir = resolve_configs_dir(swap_dir)

    dynamic_type = base_cfg["dynamics"]["type"]
    dim = int(base_cfg["dynamics"]["dim"])
    method_name = base_cfg["method"]["name"]

    plan_path = get_default_plan_path(
        configs_dir=configs_dir,
        phase=phase,
        dynamic_type=dynamic_type,
        dim=dim,
        method_type=method_name,
    )
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing default plan for {phase}: {plan_path}")

    plan = load_yaml(plan_path)
    return plan["search"][phase]["params"]


def load_ablation_phase_mode(phase: str, ablation_cfg: dict | None = None) -> str:
    if ablation_cfg is None:
        return "one_factor"
    phase_block = ablation_cfg.get(phase, {})
    mode = str(phase_block.get("mode", "one_factor")).strip().lower()
    if mode not in {"one_factor", "grid"}:
        raise ValueError(f"Unsupported ablation mode for phase={phase}: {mode}")
    return mode


def make_compare_cases(base_cfg: dict, phase: str, ablation_cfg: dict | None = None):
    param_specs = load_phase_param_specs(base_cfg, phase, ablation_cfg=ablation_cfg)
    current_flat = current_cfg_to_flat(base_cfg)

    cases = [
        {
            "name": "current_config",
            "updates": {},
            "changed_param": "-",
            "changed_from": "-",
            "changed_to": "-",
        }
    ]

    added_names = {"current_config"}

    def add_case(name, updates, changed_param, changed_from, changed_to):
        if name in added_names:
            return
        cases.append(
            {
                "name": name,
                "updates": updates,
                "changed_param": changed_param,
                "changed_from": changed_from,
                "changed_to": changed_to,
            }
        )
        added_names.add(name)

    for dotted_key, spec in param_specs.items():
        if spec.get("type") != "categorical":
            continue

        current_value = deep_get_by_dotted_key(base_cfg, dotted_key, default=None)
        active_if = spec.get("active_if")

        if is_active_param(spec, current_flat):
            if current_value is None:
                continue

            for cand in spec.get("values", []):
                if cand == current_value:
                    continue

                short_name = dotted_key.split(".")[-1]
                case_name = f"{short_name}_{slugify(stringify_value(cand))}"

                add_case(
                    name=case_name,
                    updates={dotted_key: cand},
                    changed_param=dotted_key,
                    changed_from=current_value,
                    changed_to=cand,
                )
            continue

        if active_if:
            for cond_key, expected_value in active_if.items():
                cond_current = deep_get_by_dotted_key(base_cfg, cond_key, default=None)

                parent_updates = {}
                if cond_current != expected_value:
                    parent_updates[cond_key] = expected_value

                virtual_cfg = deepcopy(base_cfg)
                for k, v in parent_updates.items():
                    deep_set(virtual_cfg, k, v)
                virtual_flat = current_cfg_to_flat(virtual_cfg)

                if not is_active_param(spec, virtual_flat):
                    continue

                virtual_current_value = deep_get_by_dotted_key(virtual_cfg, dotted_key, default=None)

                for cand in spec.get("values", []):
                    if virtual_current_value is not None and cand == virtual_current_value:
                        continue

                    short_name = dotted_key.split(".")[-1]

                    if parent_updates:
                        parent_name = "__".join(
                            f"{k.split('.')[-1]}_{slugify(stringify_value(v))}"
                            for k, v in parent_updates.items()
                        )
                        case_name = f"{parent_name}__{short_name}_{slugify(stringify_value(cand))}"
                        updates = dict(parent_updates)
                        updates[dotted_key] = cand
                        changed_param = f"{cond_key} + {dotted_key}"
                        changed_from = f"{cond_current} + {virtual_current_value}"
                        changed_to = f"{expected_value} + {cand}"
                    else:
                        case_name = f"{short_name}_{slugify(stringify_value(cand))}"
                        updates = {dotted_key: cand}
                        changed_param = dotted_key
                        changed_from = virtual_current_value
                        changed_to = cand

                    add_case(
                        name=case_name,
                        updates=updates,
                        changed_param=changed_param,
                        changed_from=changed_from,
                        changed_to=changed_to,
                    )

    return cases


def make_grid_cases(base_cfg: dict, phase: str, ablation_cfg: dict | None = None):
    param_specs = load_phase_param_specs(base_cfg, phase, ablation_cfg=ablation_cfg)

    keys = []
    values_list = []
    for dotted_key, spec in param_specs.items():
        if spec.get("type") != "categorical":
            continue
        keys.append(dotted_key)
        values_list.append(spec.get("values", []))

    cases = [
        {
            "name": "current_config",
            "updates": {},
            "changed_param": "-",
            "changed_from": "-",
            "changed_to": "-",
        }
    ]
    seen = {"current_config"}

    for combo in product(*values_list):
        updates = dict(zip(keys, combo))

        virtual_cfg = deepcopy(base_cfg)
        for k, v in updates.items():
            deep_set(virtual_cfg, k, v)
        virtual_flat = current_cfg_to_flat(virtual_cfg)

        valid = True
        for k, spec in param_specs.items():
            if spec.get("type") != "categorical":
                continue
            if not is_active_param(spec, virtual_flat):
                value = deep_get_by_dotted_key(virtual_cfg, k, default=None)
                current_value = deep_get_by_dotted_key(base_cfg, k, default=None)
                if value != current_value:
                    valid = False
                    break

        if not valid:
            continue

        same = True
        changed_params = []
        changed_from = []
        changed_to = []

        for k, v in updates.items():
            cur = deep_get_by_dotted_key(base_cfg, k, default=None)
            if cur != v:
                same = False
                changed_params.append(k)
                changed_from.append(str(cur))
                changed_to.append(str(v))

        if same:
            continue

        case_name = "__".join(
            f"{k.split('.')[-1]}_{slugify(stringify_value(v))}"
            for k, v in updates.items()
        )
        if case_name in seen:
            continue
        seen.add(case_name)

        cases.append(
            {
                "name": case_name,
                "updates": updates,
                "changed_param": " + ".join(changed_params),
                "changed_from": " + ".join(changed_from),
                "changed_to": " + ".join(changed_to),
            }
        )

    return cases


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_lines(path: Path, lines):
    write_text(path, "\n".join(lines) + "\n")


def summarize_records(records):
    valid = [r for r in records if safe_float(r.get("rmse")) is not None]
    rmse_mean, rmse_std = mean_std([r.get("rmse") for r in valid])
    csi_mean, csi_std = mean_std([r.get("csi") for r in valid])
    wall_mean, wall_std = mean_std([r.get("wallclock") for r in valid])

    return {
        "n_seeds": len(records),
        "n_valid": len(valid),
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "csi_mean": csi_mean,
        "csi_std": csi_std,
        "wall_mean": wall_mean,
        "wall_std": wall_std,
    }


def cleanup_experiments_keep_best(setting_dir: Path, records):
    valid = [r for r in records if safe_float(r.get("rmse")) is not None and r.get("returncode", 1) == 0]
    exp_root = setting_dir / "experiments"
    if not exp_root.exists():
        return
    if not valid:
        return

    best = min(valid, key=lambda r: safe_float(r.get("rmse")))
    keep_name = best["run_name"]

    for child in sorted(exp_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == keep_name:
            continue
        shutil.rmtree(child, ignore_errors=True)


def run_one_seed(project_root: Path, input_cfg: dict, setting_dir: Path, seed: int):
    cfg = deepcopy(input_cfg)
    deep_set(cfg, "system.seed", seed)

    seed_name = f"seed_{seed:04d}"
    configs_root = setting_dir / "configs"
    experiments_root = setting_dir / "experiments"

    seed_cfg_dir = configs_root / seed_name
    seed_exp_dir = experiments_root / seed_name
    seed_cfg_dir.mkdir(parents=True, exist_ok=True)
    experiments_root.mkdir(parents=True, exist_ok=True)

    cfg_path = seed_cfg_dir / "config.yaml"
    result_yaml = seed_cfg_dir / "result.yaml"
    record_json = seed_cfg_dir / "record.json"
    run_log = seed_exp_dir / "run.log"

    if result_yaml.exists():
        result = load_yaml(result_yaml)
        record = {
            "seed": seed,
            "run_name": seed_name,
            "rmse": safe_float(result.get("rmse")),
            "csi": safe_float(result.get("csi")),
            "wallclock": safe_float(result.get("wallclock", result.get("real"))),
            "returncode": int(result.get("returncode", 0)),
            "config_path": str(cfg_path),
            "result_yaml": str(result_yaml),
            "run_log": str(run_log),
            "cached": True,
        }
        return record

    cfg.setdefault("exp", {})
    cfg["exp"]["workdir_root"] = str(experiments_root)

    save_yaml(cfg, cfg_path)

    cmd = [
        "python",
        str(project_root / "main.py"),
        "--config",
        str(cfg_path),
        "--exp",
        seed_name,
    ]

    seed_exp_dir.mkdir(parents=True, exist_ok=True)
    with open(run_log, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            ["/usr/bin/time", "-p"] + cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
            check=False,
        )

    rmse, csi, real, user, sys_t = parse_log_metrics(run_log)
    is_valid = (proc.returncode == 0)
    try:
        is_valid = is_valid and (rmse is not None) and (not math.isnan(rmse))
    except Exception:
        is_valid = False

    result_dict = {
        "seed": seed,
        "run_name": seed_name,
        "rmse": rmse,
        "csi": csi,
        "wallclock": real,
        "real": real,
        "user": user,
        "sys": sys_t,
        "returncode": proc.returncode,
        "is_valid": is_valid,
        "config_path": str(cfg_path),
        "run_log": str(run_log),
        "experiment_dir": str(seed_exp_dir),
    }
    save_yaml(result_dict, result_yaml)

    record = {
        "seed": seed,
        "run_name": seed_name,
        "rmse": rmse,
        "csi": csi,
        "wallclock": real,
        "returncode": proc.returncode,
        "config_path": str(cfg_path),
        "result_yaml": str(result_yaml),
        "run_log": str(run_log),
        "cached": False,
    }
    record_json.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def write_setting_progress(setting_dir: Path, records):
    valid = [r for r in records if safe_float(r.get("rmse")) is not None]
    best_rmse = min([safe_float(r.get("rmse")) for r in valid], default=None)

    lines = []
    lines.append(f"done_seeds: [{len(records)}/{len(records)}]")
    lines.append(f"success: {sum(1 for r in records if r.get('returncode', 1) == 0)}")
    lines.append(f"failed: {sum(1 for r in records if r.get('returncode', 1) != 0)}")
    lines.append(f"best_rmse_so_far: {fmt(best_rmse)}")
    lines.append("")
    for idx, r in enumerate(records, start=1):
        lines.append(
            f"[{idx}/{len(records)}] "
            f"{r.get('run_name','')} "
            f"cached={r.get('cached', False)} "
            f"returncode={r.get('returncode','')} "
            f"rmse={r.get('rmse','')} "
            f"csi={r.get('csi','')} "
            f"wallclock={r.get('wallclock','')}"
        )
    write_lines(setting_dir / "progress.txt", lines)


def write_setting_summary(setting_dir: Path, setting_name: str, changed_param: str, changed_from, changed_to, records):
    valid = [r for r in records if safe_float(r.get("rmse")) is not None]
    rmse_mean, rmse_std = mean_std([r.get("rmse") for r in valid])
    csi_mean, csi_std = mean_std([r.get("csi") for r in valid])
    wall_mean, wall_std = mean_std([r.get("wallclock") for r in valid])

    best_record = None
    if valid:
        best_record = min(valid, key=lambda r: safe_float(r.get("rmse")))

    lines = []
    lines.append(f"setting: {setting_name}")
    lines.append(f"changed_param: {changed_param}")
    lines.append(f"changed_from: {changed_from}")
    lines.append(f"changed_to: {changed_to}")
    lines.append(f"n_seeds: {len(records)}")
    lines.append(f"n_valid: {len(valid)}")
    lines.append(f"RMSE (mean ± std): {fmt_mean_std(rmse_mean, rmse_std)}")
    lines.append(f"CSI (mean ± std): {fmt_mean_std(csi_mean, csi_std)}")
    lines.append(f"wallclock (mean ± std): {fmt_mean_std(wall_mean, wall_std)}")
    lines.append("")
    if best_record is not None:
        lines.append(f"best_seed: {best_record['seed']}")
        lines.append(f"best_run_name: {best_record['run_name']}")
        lines.append(f"best_rmse: {best_record['rmse']}")
        lines.append(f"best_result_yaml: {best_record['result_yaml']}")
    else:
        lines.append("best_seed: ")
        lines.append("best_run_name: ")
        lines.append("best_rmse: ")
        lines.append("best_result_yaml: ")

    write_lines(setting_dir / "summary.txt", lines)
    write_setting_progress(setting_dir, records)


def write_setting_delta(setting_dir: Path, setting_name: str, current_summary: dict, this_summary: dict):
    cur_rmse = current_summary.get("rmse_mean")
    cur_csi = current_summary.get("csi_mean")
    cur_wall = current_summary.get("wall_mean")

    rmse = this_summary.get("rmse_mean")
    csi = this_summary.get("csi_mean")
    wall = this_summary.get("wall_mean")

    delta_rmse = None if cur_rmse is None or rmse is None else rmse - cur_rmse
    delta_csi = None if cur_csi is None or csi is None else csi - cur_csi
    delta_wall = None if cur_wall is None or wall is None else wall - cur_wall

    lines = [
        "reference_setting: current_config",
        f"setting: {setting_name}",
        f"delta_RMSE: {fmt(delta_rmse)}",
        f"delta_CSI: {fmt(delta_csi)}",
        f"delta_wallclock: {fmt(delta_wall)}",
    ]
    write_lines(setting_dir / "delta_summary.txt", lines)


def write_bundle_summary(bundle_dir: Path, rows):
    header = [
        "setting",
        "changed_param",
        "changed_from",
        "changed_to",
        "n_seeds",
        "RMSE (mean ± std)",
        "CSI (mean ± std)",
        "wallclock (mean ± std)",
    ]
    widths = {h: len(h) for h in header}
    for row in rows:
        for h in header:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    def fmt_row(row):
        return " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in header)

    lines = [fmt_row({h: h for h in header}), "-+-".join("-" * widths[h] for h in header)]
    for row in rows:
        lines.append(fmt_row(row))
    write_lines(bundle_dir / "summary.txt", lines)

    current = next((r for r in rows if r["setting"] == "current_config"), None)
    cur_rmse = safe_float(current.get("_rmse_mean")) if current else None
    cur_csi = safe_float(current.get("_csi_mean")) if current else None
    cur_wall = safe_float(current.get("_wall_mean")) if current else None

    delta_header = ["setting", "changed_to", "delta_RMSE", "delta_CSI", "delta_wallclock"]
    delta_rows = []
    for row in rows:
        rmse = safe_float(row.get("_rmse_mean"))
        csi = safe_float(row.get("_csi_mean"))
        wall = safe_float(row.get("_wall_mean"))
        delta_rows.append(
            {
                "setting": row["setting"],
                "changed_to": row["changed_to"],
                "delta_RMSE": fmt(None if cur_rmse is None or rmse is None else rmse - cur_rmse),
                "delta_CSI": fmt(None if cur_csi is None or csi is None else csi - cur_csi),
                "delta_wallclock": fmt(None if cur_wall is None or wall is None else wall - cur_wall),
            }
        )

    widths = {h: len(h) for h in delta_header}
    for row in delta_rows:
        for h in delta_header:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    def fmt_row2(row):
        return " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in delta_header)

    lines = [fmt_row2({h: h for h in delta_header}), "-+-".join("-" * widths[h] for h in delta_header)]
    for row in delta_rows:
        lines.append(fmt_row2(row))
    write_lines(bundle_dir / "delta_summary.txt", lines)

    valid_rows = [r for r in rows if safe_float(r.get("_rmse_mean")) is not None]
    best_row = min(valid_rows, key=lambda r: safe_float(r.get("_rmse_mean")), default=None)

    progress_lines = [
        f"done_settings: [{len(rows)}/{len(rows)}]",
        "reference_setting: current_config",
        f"best_setting_by_rmse: {best_row['setting'] if best_row else ''}",
    ]
    write_lines(bundle_dir / "progress.txt", progress_lines)


def run_setting(project_root: Path, bundle_dir: Path, base_cfg: dict, case: dict, seeds):
    setting_dir = bundle_dir / case["name"]
    setting_dir.mkdir(parents=True, exist_ok=True)

    cfg = deepcopy(base_cfg)
    for dotted_key, value in case["updates"].items():
        deep_set(cfg, dotted_key, value)

    records = []
    for seed_idx, seed in enumerate(seeds, start=1):
        print(f"[ABLATION] setting={case['name']} seed [{seed_idx}/{len(seeds)}] seed={seed}")

        rec = run_one_seed(
            project_root=project_root,
            input_cfg=cfg,
            setting_dir=setting_dir,
            seed=seed,
        )
        records.append(rec)

        print(
            f"[ABLATION] done setting={case['name']} "
            f"seed [{seed_idx}/{len(seeds)}] "
            f"run={rec.get('run_name', '')} "
            f"rmse={rec.get('rmse', '')} "
            f"csi={rec.get('csi', '')} "
            f"wallclock={rec.get('wallclock', '')} "
            f"cached={rec.get('cached', False)} "
            f"returncode={rec.get('returncode', '')}"
        )

        write_setting_summary(
            setting_dir=setting_dir,
            setting_name=case["name"],
            changed_param=case["changed_param"],
            changed_from=case["changed_from"],
            changed_to=case["changed_to"],
            records=records,
        )

    cleanup_experiments_keep_best(setting_dir, records)
    return setting_dir, cfg, records


def pick_best_case(case_results):
    best_name = "current_config"
    best_rmse = None
    best_cfg = None

    for item in case_results:
        summary = item["summary"]
        rmse = summary["rmse_mean"]
        if rmse is None:
            continue
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_name = item["case"]["name"]
            best_cfg = item["cfg"]

    return best_name, best_rmse, best_cfg


def write_best_from_ablation(bundle_dir: Path, best_cfg: dict):
    if best_cfg is not None:
        save_yaml(best_cfg, bundle_dir / "best_from_ablation.yaml")


def run_bundle(project_root: Path, bundle_dir: Path, base_cfg: dict, compare_cases, seeds):
    bundle_dir.mkdir(parents=True, exist_ok=True)

    case_results = []
    for idx, case in enumerate(compare_cases, start=1):
        print("-" * 80)
        print(f"[ABLATION] bundle={bundle_dir.name} setting [{idx}/{len(compare_cases)}] {case['name']}")
        print("-" * 80)

        setting_dir, cfg, records = run_setting(
            project_root=project_root,
            bundle_dir=bundle_dir,
            base_cfg=base_cfg,
            case=case,
            seeds=seeds,
        )

        summary = summarize_records(records)
        case_results.append(
            {
                "case": case,
                "setting_dir": setting_dir,
                "cfg": cfg,
                "records": records,
                "summary": summary,
            }
        )

        current = next((x for x in case_results if x["case"]["name"] == "current_config"), None)
        current_summary = current["summary"] if current else {}

        rows = []
        for item in case_results:
            s = item["summary"]
            rows.append(
                {
                    "setting": item["case"]["name"],
                    "changed_param": item["case"]["changed_param"],
                    "changed_from": item["case"]["changed_from"],
                    "changed_to": item["case"]["changed_to"],
                    "n_seeds": s["n_seeds"],
                    "RMSE (mean ± std)": fmt_mean_std(s["rmse_mean"], s["rmse_std"]),
                    "CSI (mean ± std)": fmt_mean_std(s["csi_mean"], s["csi_std"]),
                    "wallclock (mean ± std)": fmt_mean_std(s["wall_mean"], s["wall_std"]),
                    "_rmse_mean": s["rmse_mean"],
                    "_csi_mean": s["csi_mean"],
                    "_wall_mean": s["wall_mean"],
                }
            )

        for item in case_results:
            write_setting_delta(
                setting_dir=item["setting_dir"],
                setting_name=item["case"]["name"],
                current_summary=current_summary,
                this_summary=item["summary"],
            )

        write_bundle_summary(bundle_dir, rows)

    best_name, best_rmse, best_cfg = pick_best_case(case_results)
    write_best_from_ablation(bundle_dir, best_cfg)
    print(f"[ABLATION] bundle done: {bundle_dir} best_setting={best_name} rmse={fmt(best_rmse)}")
    return best_cfg


def run_test_mode(project_root: Path, output_root: Path, base_cfg: dict, case_name: str, seeds):
    bundle_dir = output_root / "ablation" / case_name / "test"
    compare_cases = [
        {
            "name": "current_config",
            "updates": {},
            "changed_param": "-",
            "changed_from": "-",
            "changed_to": "-",
        }
    ]
    run_bundle(
        project_root=project_root,
        bundle_dir=bundle_dir,
        base_cfg=base_cfg,
        compare_cases=compare_cases,
        seeds=seeds,
    )


def run_phase_chain(project_root: Path, output_root: Path, base_cfg: dict, case_name: str, phases, seeds, ablation_cfg=None):
    current_cfg = deepcopy(base_cfg)
    ablation_root = output_root / "ablation" / case_name

    for phase in phases:
        bundle_dir = ablation_root / phase
        mode = load_ablation_phase_mode(phase, ablation_cfg=ablation_cfg)

        if mode == "grid":
            compare_cases = make_grid_cases(current_cfg, phase, ablation_cfg=ablation_cfg)
        else:
            compare_cases = make_compare_cases(current_cfg, phase, ablation_cfg=ablation_cfg)

        if not compare_cases:
            print(f"[ABLATION] skip phase={phase}: no compare cases")
            continue

        print("=" * 80)
        print(f"[ABLATION] phase : {phase}")
        print(f"[ABLATION] mode  : {mode}")
        print(f"[ABLATION] cases : {len(compare_cases)}")
        print("=" * 80)

        current_cfg = run_bundle(
            project_root=project_root,
            bundle_dir=bundle_dir,
            base_cfg=current_cfg,
            compare_cases=compare_cases,
            seeds=seeds,
        ) or current_cfg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--case_name", required=True)

    parser.add_argument("--phase", choices=ALL_PHASES, default=None)
    parser.add_argument("--from_phase", "--from", dest="from_phase", choices=PHASE_ORDER, default=None)
    parser.add_argument("--until", choices=PHASE_ORDER, default=None)

    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if args.test and (args.phase is not None or args.until is not None or args.from_phase is not None):
        raise ValueError("--test mode does not take --phase/--from_phase/--until")

    if (not args.test) and args.phase is None and args.until is None:
        raise ValueError("Provide either --phase or --until in normal mode")

    if args.phase is not None and args.until is not None:
        raise ValueError("Use either --phase or --until, not both")

    if args.phase is not None and args.from_phase is not None:
        raise ValueError("--from_phase cannot be used with --phase")

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    output_root = resolve_output_root(config_path, args.root_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.rerun:
        rerun_root = output_root / "ablation" / args.case_name
        if rerun_root.exists():
            shutil.rmtree(rerun_root)

    base_cfg = load_yaml(config_path)
    ablation_cfg = base_cfg.get("ablation", {})
    project_root = Path(__file__).resolve().parent.parent

    print("=" * 80)
    print(f"[ABLATION] config    : {config_path}")
    print(f"[ABLATION] root_dir  : {output_root}")
    print(f"[ABLATION] case_name : {args.case_name}")
    print(f"[ABLATION] seeds     : {args.seeds}")
    print(f"[ABLATION] test      : {args.test}")
    print("=" * 80)

    if args.test:
        run_test_mode(
            project_root=project_root,
            output_root=output_root,
            base_cfg=base_cfg,
            case_name=args.case_name,
            seeds=args.seeds,
        )
        return

    if args.phase is not None:
        phases_to_run = [args.phase]
    else:
        start_idx = 0 if args.from_phase is None else PHASE_ORDER.index(args.from_phase)
        end_idx = PHASE_ORDER.index(args.until)
        if start_idx > end_idx:
            raise ValueError(
                f"--from_phase ({args.from_phase}) must be earlier than or equal to --until ({args.until})"
            )
        phases_to_run = PHASE_ORDER[start_idx:end_idx + 1]

    run_phase_chain(
        project_root=project_root,
        output_root=output_root,
        base_cfg=base_cfg,
        case_name=args.case_name,
        phases=phases_to_run,
        seeds=args.seeds,
        ablation_cfg=ablation_cfg,
    )


if __name__ == "__main__":
    main()

"""

python -m swap.run_ablation_pipeline \
  --config /data3/dwkim/masf/configs/dynamics/kolmogorov_256.yaml \
  --root_dir /data3/dwkim/masf/0415/ours/kolmogorov_128/grid_mask/stride_10 \
  --case_name moving_0.2_0418 \
  --until finetuning \
  --seeds 42 43 44 45 46

python -m swap.run_ablation_pipeline \
  --config /data3/dwkim/masf/configs/dynamics/kolmogorov_256.yaml \
  --root_dir /data3/dwkim/masf/0415/ours/kolmogorov_128/grid_mask/stride_10 \
  --case_name moving_0.2_0418 \
  --until finetuning \
  --seeds 42 43 44 45 46

python -m swap.run_ablation_pipeline \
  --config /data3/dwkim/masf/0415/enkf/kolmogorov_128/grid_mask/stride_10/baseline_tuning/num_samples_300/best_enkf.yaml \
  --root_dir /data3/dwkim/masf/0415/enkf/kolmogorov_128/grid_mask/stride_10 \
  --case_name num300_ablation \
  --phase baseline_tuning \
  --seeds 42 43 44 45 46

python -m swap.run_ablation_pipeline \
  --config /data3/dwkim/masf/0415/enkf/kolmogorov_128/grid_mask/stride_10/baseline_tuning/num_samples_200/best_enkf.yaml \
  --root_dir /data3/dwkim/masf/0415/enkf/kolmogorov_128/grid_mask/stride_10 \
  --case_name num300_ablation \
  --phase baseline_tuning \
  --seeds 42 43 44 45 46
"""