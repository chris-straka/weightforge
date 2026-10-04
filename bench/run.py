#!/usr/bin/env python3
"""P4 harness: check + fix every GLB in a folder, collect before/after
numbers and A/B sheets for owner review.

    bench/run.py [models_dir] [--out results_dir] [--candidates dir]

models_dir defaults to bench/models/ and results to bench/results/ (both
gitignored: real assets and their outputs never get committed). With
--candidates, a file of the same stem there (e.g. UniRig output) is passed
as an extra candidate. Writes results/summary.json and results/index.md
(one row per asset: verdict, score before -> after, failing findings,
sheet path). Exit 1 if any fix made an asset worse (must never happen).
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "..", "rust", "target", "release", "weights")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="?", default=os.path.join(HERE, "models"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--candidates")
    ap.add_argument("--cli", default=CLI)
    a = ap.parse_args()
    if not os.path.exists(a.cli):
        sys.exit(f"build the CLI first: cd rust && cargo build --release ({a.cli})")
    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(a.models) if f.lower().endswith(".glb"))
    rows, worse = [], 0
    for f in files:
        stem = os.path.splitext(f)[0]
        src = os.path.join(a.models, f)
        fixed = os.path.join(a.out, f"{stem}_fixed.glb")
        sheet = os.path.join(a.out, f"{stem}_ab.png")
        report = os.path.join(a.out, f"{stem}_fixed.report.json")
        args = [a.cli, "fix", src, "--out", fixed, "--sheet", sheet, "--report", report]
        if a.candidates:
            for ext in (".glb", "_rigged.glb"):
                c = os.path.join(a.candidates, stem + ext)
                if os.path.exists(c):
                    args += ["--candidate", c]
        t0 = time.time()
        proc = subprocess.run(args, capture_output=True, text=True)
        dt = time.time() - t0
        if proc.returncode not in (0, 1):
            rows.append({"asset": stem, "error": (proc.stderr or proc.stdout).strip()})
            print(f"ERROR {stem}: {rows[-1]['error']}")
            continue
        with open(report) as fh:
            r = json.load(fh)
        b, af = r["before"], r["after"]
        verdict = "FIXED" if af["pass"] else ("IMPROVED" if r["improved"] else "UNCHANGED")
        if af["score"] < b["score"]:
            worse += 1
            verdict = "WORSE"
        rows.append({
            "asset": stem, "verdict": verdict, "score_before": b["score"], "score_after": af["score"],
            "fails_before": b["fails"], "fails_after": af["fails"], "verts_changed": r["verts_changed"],
            "seconds": round(dt, 2), "sheet": os.path.relpath(sheet, a.out),
            "methods_used": sorted({x["chosen"] for x in r["regions"] if x["chosen"] != "original"}),
        })
        print(f"{verdict:<9} {stem}: {b['score']:.1f} -> {af['score']:.1f}, fails {b['fails']} -> {af['fails']} ({dt:.1f}s)")
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    with open(os.path.join(a.out, "index.md"), "w") as fh:
        fh.write("# weightforge bench\n\n| asset | verdict | score | fails | sheet |\n|---|---|---|---|---|\n")
        for r in rows:
            if "error" in r:
                fh.write(f"| {r['asset']} | ERROR | | | |\n")
            else:
                fh.write(f"| {r['asset']} | {r['verdict']} | {r['score_before']:.0f}→{r['score_after']:.0f} | {r['fails_before']}→{r['fails_after']} | [A/B]({r['sheet']}) |\n")
    ok = [r for r in rows if "error" not in r]
    print(f"\n{len(ok)} assets: {sum(r['verdict'] == 'FIXED' for r in ok)} fixed, "
          f"{sum(r['verdict'] == 'IMPROVED' for r in ok)} improved, {worse} worse; {a.out}/index.md")
    sys.exit(1 if worse else 0)


if __name__ == "__main__":
    main()
