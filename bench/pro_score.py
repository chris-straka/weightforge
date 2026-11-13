"""Closeness to professional rigs: weightforge's main score (plain Python 3).

    python3 bench/pro_score.py --out DIR --label NAME --weights WEIGHTS --skintokens SKINTOKENS \
        --hero hero.glb [--fix-args "..."] RIG=pro.glb [RIG=pro.glb ...]
    python3 bench/pro_score.py --out DIR --table LABEL [LABEL ...]

Per professionally rigged character (benchmark use only; licences in the
shelf's INDEX.md): its weights are replaced the way the pipeline's inputs
come, by SkinTokens' prediction for that same mesh and skeleton (cached in
DIR/_inputs/RIG.st.glb), then `weights fix` runs on that. `weights
closeness` measures each result against the pro weights: the skin's
distance from where the pro rig puts it on the hero's clips and on the
poses (1.75 m body), and the weights' distance. `weights check` adds the
headline's tears. DIR/LABEL/RIG/ holds every JSON; --table prints the
labels side by side.
"""

import argparse
import json
import os
import statistics
import subprocess


def run(cmd, out=None):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if out:
        with open(out, "w") as f:
            f.write(p.stdout)
    return p.returncode, p.stderr


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def summary(d, rig):
    c, k = load(os.path.join(d, "close.json")), load(os.path.join(d, "check.json"))
    if not c:
        return None
    row = {"rig": rig, "weight_err": c["weight_err"]}
    for m in c["motion"]:
        key = "clips" if m["group"] == "game clips" else "poses"
        row[key] = {"mean_cm": m["mean_cm"], "p95_cm": m["p95_cm"], "off_pct": m["off_pct"]}
    for sec in (k or {}).get("headline", []):
        key = "clips" if sec["name"] == "game clips" else "poses"
        row.setdefault(key, {})["tears"] = sec["tears_pct"]
    return row


def measure(a, pro, test, d):
    run([a.weights, "closeness", pro, test, "--motion", a.hero, "--json"], os.path.join(d, "close.json"))
    run([a.weights, "check", test, "--motion", a.hero, "--json"], os.path.join(d, "check.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label")
    ap.add_argument("--weights")
    ap.add_argument("--skintokens")
    ap.add_argument("--hero")
    ap.add_argument("--fix-args", default="")
    ap.add_argument("--table", nargs="+")
    ap.add_argument("rigs", nargs="*")
    a = ap.parse_args()
    if a.table:
        return table(a.out, a.table)
    inputs = os.path.join(a.out, "_inputs")
    os.makedirs(inputs, exist_ok=True)
    for spec in a.rigs:
        rig, pro = spec.split("=", 1)
        st = os.path.join(inputs, f"{rig}.st.glb")
        if not os.path.exists(st):
            code, err = run([a.skintokens, "skin", pro, st, "--report", st + ".json"])
            if code != 0:
                print(f"{rig}: skintokens failed: {err[-300:]}", flush=True)
                continue
        # The input itself, once.
        d0 = os.path.join(a.out, "input", rig)
        if not os.path.exists(os.path.join(d0, "close.json")):
            os.makedirs(d0, exist_ok=True)
            measure(a, pro, st, d0)
        d = os.path.join(a.out, a.label, rig)
        os.makedirs(d, exist_ok=True)
        fixed = os.path.join(d, "fixed.glb")
        args = [x.replace("HERO", a.hero) for x in a.fix_args.split()]
        run([a.weights, "fix", st, "--out", fixed] + args, os.path.join(d, "fix.txt"))
        if os.path.exists(fixed):
            measure(a, pro, fixed, d)
            os.remove(fixed)
        print(f"{rig}: input {summary(d0, rig)}\n  {a.label}: {summary(d, rig)}", flush=True)


def table(out, labels):
    labels = ["input"] + [l for l in labels if l != "input"]
    rigs = sorted({r for l in labels if os.path.isdir(os.path.join(out, l)) for r in os.listdir(os.path.join(out, l))})
    print("Skin ever over 2 cm off the pro rig's motion on the hero's clips, % (95th percentile cm); weights off (0-1)\n")
    print("| rig | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    agg = {l: [] for l in labels}
    for r in rigs:
        cells = []
        for l in labels:
            s = summary(os.path.join(out, l, r), r)
            if not s or "clips" not in s:
                cells.append("-")
                continue
            c = s["clips"]
            agg[l].append((c["off_pct"], c["p95_cm"], s["weight_err"], c.get("tears")))
            cells.append(f"{c['off_pct']:.1f}% ({c['p95_cm']:.1f}) w {s['weight_err']:.3f}")
        print(f"| {r} | " + " | ".join(cells) + " |")
    for l in labels:
        v = agg[l]
        if v:
            print(f"{l}: median off {statistics.median(x[0] for x in v):.2f}%, median p95 {statistics.median(x[1] for x in v):.2f} cm, "
                  f"median weights off {statistics.median(x[2] for x in v):.3f}, median tears "
                  f"{statistics.median(x[3] for x in v if x[3] is not None):.2f}% ({len(v)} rigs)")


main()
