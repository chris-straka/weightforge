#!/usr/bin/env python3
"""P3 gate: do the P3 methods (transfer + inpaint, optimize) beat P2
(clean-up + geodesic) on the 10 worst regions of a real test set?

    bench/p3.py [models_dir] [--out dir]

Runs `weights fix` twice per asset (--method smooth,geodesic and --method
auto), takes the 10 regions with the lowest input score across the whole
set, and compares their after-fix scores. Prints a table and writes
p3.json. Exit 0 when auto >= P2 on every one of them and better on most.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "..", "rust", "target", "release", "weights")


def fix(src, out_dir, stem, method):
    out = os.path.join(out_dir, f"{stem}_{method.replace(',', '+')}.glb")
    rep = out[:-4] + ".report.json"
    subprocess.run([CLI, "fix", src, "--out", out, "--report", rep, "--method", method], capture_output=True, check=False)
    with open(rep) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="?", default=os.path.join(HERE, "models"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "p3"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    regions = []
    for f in sorted(x for x in os.listdir(a.models) if x.lower().endswith(".glb")):
        stem = os.path.splitext(f)[0]
        src = os.path.join(a.models, f)
        p2 = {r["name"]: r for r in fix(src, a.out, stem, "smooth,geodesic")["regions"]}
        p3 = {r["name"]: r for r in fix(src, a.out, stem, "auto")["regions"]}
        for name, r in p3.items():
            if name in p2:
                regions.append({"asset": stem, "region": name, "before": r["before"], "p2": p2[name]["after"], "p3": r["after"], "p3_method": r["chosen"]})
    worst = sorted(regions, key=lambda r: (r["before"], r["asset"], r["region"]))[:10]
    print(f"{'asset':<26} {'region':<16} {'input':>6} {'P2':>6} {'P3':>6}  P3 via")
    for r in worst:
        print(f"{r['asset']:<26} {r['region']:<16} {r['before']:>6.1f} {r['p2']:>6.1f} {r['p3']:>6.1f}  {r['p3_method']}")
    better = sum(r["p3"] > r["p2"] + 0.05 for r in worst)
    not_worse = all(r["p3"] >= r["p2"] - 0.05 for r in worst)
    print(f"\nP3 better on {better}/10, never worse: {not_worse}")
    with open(os.path.join(a.out, "p3.json"), "w") as f:
        json.dump({"worst": worst, "better": better, "never_worse": not_worse}, f, indent=2)
    sys.exit(0 if not_worse and better >= 5 else 1)


if __name__ == "__main__":
    main()
