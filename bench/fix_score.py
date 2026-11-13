"""Does `weights fix` make the cast look better? (plain Python 3)

    python3 bench/fix_score.py --out DIR --label NAME --weights WEIGHTS --motionforge MF \
        --hero hero.glb  RIG=raw.glb [RIG=raw.glb ...]
    python3 bench/fix_score.py --out DIR --compare LABEL_A LABEL_B

Per rig, into DIR/LABEL/RIG/: `weights fix` of the raw rig (the route's
input: the standardized skeleton and its predicted weights), the fixed rig
given the hero's walk, run, attack_1, dodge, hit and death by `motionforge
animate` (the game's path), and `weights check` of it: the headline's tears
(share of the surface in failing findings) on the clips and on the poses.
The raw rig is measured the same way once per DIR (DIR/raw/RIG/).

--compare prints both labels side by side and the cast's mean and median;
a change is kept only when the cast is better as a whole (README of the
round's review doc).
"""

import argparse
import json
import os
import statistics
import subprocess

CLIPS = ["walk", "run", "attack_1", "dodge", "hit", "death"]


def run(cmd, out=None):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if out:
        with open(out, "w") as f:
            f.write(p.stdout)
    return p.returncode, p.stderr


def headline(path):
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None
    out = {"score": d.get("score")}
    for sec in d.get("headline", []):
        k = "clips" if sec["name"] == "game clips" else "poses"
        out[k] = sec["tears_pct"]
        out[k + "_through"] = sec["through_cm2"]
        kinds = {}
        for f in sec["findings"]:
            if f["severity"] == "fail" and f["kind"] != "limb_through":
                kinds[f["kind"]] = round(kinds.get(f["kind"], 0) + f["area_cm2"])
        out[k + "_kinds"] = kinds
    return out


def measure(d, glb, a, pick, clipdir):
    anim = os.path.join(d, "anim.glb")
    code, err = run([a.motionforge, "animate", "--input", glb, "--clips", clipdir, "--pick", pick, "--output", anim])
    if code != 0:
        with open(os.path.join(d, "animate.err"), "w") as f:
            f.write(err)
        return None
    run([a.weights, "check", anim, "--json"], os.path.join(d, "check.json"))
    os.remove(anim)
    return headline(os.path.join(d, "check.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label")
    ap.add_argument("--weights")
    ap.add_argument("--motionforge")
    ap.add_argument("--hero")
    ap.add_argument("--fix-args", default="", help="extra flags for weights fix, space-separated")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("rigs", nargs="*")
    a = ap.parse_args()
    if a.compare:
        return compare(a.out, *a.compare)
    clipdir = os.path.join(a.out, "_clips")
    os.makedirs(clipdir, exist_ok=True)
    hero = os.path.join(clipdir, "hero.glb")
    if not os.path.exists(hero):
        os.symlink(os.path.abspath(a.hero), hero)
    pick = ",".join(f"hero.glb:{c}={c}" for c in CLIPS)
    rows = {}
    for spec in a.rigs:
        name, raw = spec.split("=", 1)
        rd = os.path.join(a.out, "raw", name)
        if not os.path.exists(os.path.join(rd, "check.json")):
            os.makedirs(rd, exist_ok=True)
            measure(rd, raw, a, pick, clipdir)
        d = os.path.join(a.out, a.label, name)
        os.makedirs(d, exist_ok=True)
        fixed = os.path.join(d, "fixed.glb")
        code, err = run([a.weights, "fix", raw, "--out", fixed] + a.fix_args.split(), os.path.join(d, "fix.txt"))
        with open(os.path.join(d, "fix.err"), "w") as f:
            f.write(err)
        row = {"raw": headline(os.path.join(rd, "check.json")), "fixed": measure(d, fixed, a, pick, clipdir) if os.path.exists(fixed) else None}
        rows[name] = row
        r, x = row["raw"] or {}, row["fixed"] or {}
        print(f"{name:<24} raw {r.get('clips')} / {r.get('poses')}   fixed {x.get('clips')} / {x.get('poses')}", flush=True)
    with open(os.path.join(a.out, a.label, "rows.json"), "w") as f:
        json.dump(rows, f, indent=1)


def compare(out, la, lb):
    def load(label):
        try:
            return json.load(open(os.path.join(out, label, "rows.json")))
        except OSError:
            return {}
    ra, rb = load(la), load(lb)
    names = sorted(set(ra) | set(rb))
    cols = {}
    print(f"| rig | raw clips / poses | {la} clips / poses | {lb} clips / poses |")
    print("|---|---|---|---|")
    for n in names:
        raw = (ra.get(n) or rb.get(n) or {}).get("raw") or {}
        a = (ra.get(n) or {}).get("fixed") or {}
        b = (rb.get(n) or {}).get("fixed") or {}
        for k, v in (("raw", raw), (la, a), (lb, b)):
            for s in ("clips", "poses"):
                if v.get(s) is not None:
                    cols.setdefault((k, s), []).append(v[s])
        print(f"| {n} | {raw.get('clips')} / {raw.get('poses')} | {a.get('clips')} / {a.get('poses')} | {b.get('clips')} / {b.get('poses')} |")
    for k in ("raw", la, lb):
        c, p = cols.get((k, "clips"), []), cols.get((k, "poses"), [])
        if c:
            print(f"{k}: clips mean {statistics.mean(c):.2f} median {statistics.median(c):.2f}; poses mean {statistics.mean(p):.2f} median {statistics.median(p):.2f} ({len(c)} rigs)")


main()
