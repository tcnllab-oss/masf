import json
from pathlib import Path

# text input/ouput helper

def init_text_file(path, header): ##
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header + "\n")


def append_text(path, text): ##
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(text + "\n")


def write_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

import math


def parse_log_metrics(log_file):
    rmse = float("nan")
    csi = float("nan")
    real = float("nan")
    user = float("nan")
    sys_t = float("nan")

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("FINAL_RMSE"):
                try:
                    rmse = float(line.split()[-1])
                except Exception:
                    pass

            elif line.startswith("FINAL_CSI"):
                try:
                    csi = float(line.split()[-1])
                except Exception:
                    pass

            elif line.startswith("real "):
                try:
                    real = float(line.split()[-1])
                except Exception:
                    pass

            elif line.startswith("user "):
                try:
                    user = float(line.split()[-1])
                except Exception:
                    pass

            elif line.startswith("sys "):
                try:
                    sys_t = float(line.split()[-1])
                except Exception:
                    pass

    return rmse, csi, real, user, sys_t