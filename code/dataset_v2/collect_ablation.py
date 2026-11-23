#!/usr/bin/env python3
"""汇总消融套件结果（服务器端运行，输出 JSON 到 stdout）。

注意：suite ablation_20261121T061720Z 中的输入消融 run 因 monkey-patch 无法进入
spawn worker 而失效，这里只保留其中**有效的 λ 扫描**（A1_*）。
输入消融取 worker 安全版套件 ablation_input_v2_*。
"""
import glob
import json
import os

PHYSICS_SUITE = "/g3240101584lt/srtp-ocean/runs/ablation_20261121T061720Z"
INPUT_SUITE = sorted(glob.glob("/g3240101584lt/srtp-ocean/runs/ablation_input_v2_*"))[-1]
BASELINE = "/g3240101584lt/srtp-ocean/runs/four_models_200ep_seed20261120_20261120T115114Z/swin"


def best_of(path):
    rows = [json.loads(line) for line in open(path)]
    if not rows:
        return None
    b = min(rows, key=lambda r: r["validation_score"])
    return {
        "epochs": len(rows),
        "best_epoch": b["epoch"],
        "score": b["validation_score"],
        "ct": b["validation_CT_RMSE_C"],
        "sa": b["validation_SA_RMSE_g_kg"],
    }


def collect(suite, group, prefix_filter=None):
    out = []
    for d in sorted(glob.glob(os.path.join(suite, "*/"))):
        name = os.path.basename(d.rstrip("/"))
        if prefix_filter and not name.startswith(prefix_filter):
            continue
        mp = os.path.join(d, "metrics.jsonl")
        if not os.path.exists(mp):
            continue
        info = best_of(mp)
        if info is None:
            continue
        out.append({"group": group, "name": name, **info})
    return out


def main():
    runs = []
    runs += collect(PHYSICS_SUITE, "physics", "A1_lambda")
    runs += collect(INPUT_SUITE, "input")
    info = best_of(os.path.join(BASELINE, "metrics.jsonl"))
    runs.append({"group": "baseline", "name": "BASELINE_lambda0p01", **info})

    physics = {}
    for f in sorted(glob.glob(os.path.join(PHYSICS_SUITE, "*.physics.json"))):
        try:
            physics[os.path.basename(f).replace(".physics.json", "")] = json.load(open(f))
        except Exception:
            pass

    print(json.dumps({"runs": runs, "physics": physics, "input_suite": INPUT_SUITE},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
