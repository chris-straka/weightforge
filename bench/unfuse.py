#!/usr/bin/env python3
"""Fused-limb bench: fix and score a folder of rigs, then lay two runs side
by side (stdlib only; the rigs are the owner's corpus and stay outside the
repo).

    bench/unfuse.py run   --in IN --out OUT --tag base --weights BIN --probe BIN
                          [--suffix .std.glb] [--no-fix] [--fix-args "..."]
                          [--pre "CMD {src} {dst}"]... [--jobs 2] [--only a,b]
    bench/unfuse.py table --out OUT base new [--md table.md] [--pick]

`run` takes every `IN/<id><suffix>`, runs each `--pre` on it in turn (a
mesh edit or a re-weight that writes `{dst}`), then `weights fix`, then
the probe (`rust/core/examples/fused_probe.rs`: score, failing findings,
failing vertices per pose, joins between limbs) and writes
`OUT/<tag>/<id>.glb`, `<id>.probe.json`, `<id>.fused.png`, `rows.json`.
`table` prints the runs per rig and the headline: the set's median score
and its total failing vertices in the hip-forward poses (crotch) and the
arm poses (armpit); with --pick also what the rig route would keep, the
better of the first and last run per rig.
"""

import argparse
import concurrent.futures
import json
import os
import shlex
import statistics
import subprocess
import sys


def sh(cmd, env=None, timeout=3600):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return r.returncode, r.stdout, r.stderr


def one(a, rid):
    src = os.path.join(a.inp, rid + a.suffix)
    d = os.path.join(a.out, a.tag)
    env = dict(os.environ, RAYON_NUM_THREADS=str(a.threads))
    row = {"id": rid}
    cur = src
    for k, pre in enumerate(a.pre):
        dst = os.path.join(d, f"{rid}.pre{k + 1}.glb")
        cmd = [x.format(src=cur, dst=dst, id=rid) for x in shlex.split(pre)]
        code, out, err = sh(cmd, env)
        row.setdefault("pre", []).append({"code": code, "tail": (err or out).strip()[-400:]})
        if os.path.exists(dst):
            cur = dst
    fixed = os.path.join(d, rid + ".glb")
    if a.no_fix:
        if os.path.abspath(cur) != os.path.abspath(fixed):
            subprocess.run(["cp", cur, fixed], check=True)
    else:
        code, out, err = sh([a.weights, "fix", cur, "--out", fixed] + shlex.split(a.fix_args), env)
        row["fix"] = {"code": code}
        if not os.path.exists(fixed):
            row["fix"]["error"] = (err or out).strip()[-400:]
            subprocess.run(["cp", cur, fixed], check=True)
    code, out, err = sh([a.probe, fixed, "--out", os.path.join(d, rid + ".fused.png")], env)
    try:
        row.update(json.loads(out.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        row["error"] = (err or out).strip()[-400:]
    with open(os.path.join(d, rid + ".probe.json"), "w") as f:
        json.dump(row, f, indent=1)
    return row


def pose_sum(row, prefix, side=None):
    return sum(n for p, n in row.get("poses", {}).items()
               if p.startswith(prefix) and (side is None or p.endswith("." + side)))


def run(a):
    d = os.path.join(a.out, a.tag)
    os.makedirs(d, exist_ok=True)
    ids = sorted(f[: -len(a.suffix)] for f in os.listdir(a.inp) if f.endswith(a.suffix) and not f.startswith("."))
    if a.only:
        ids = [i for i in ids if i in a.only.split(",")]
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda i: one(a, i), ids))
    with open(os.path.join(d, "rows.json"), "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        print(f"{r['id']}: {r.get('score', '?')}/{r.get('fails', '?')} crotch {pose_sum(r, 'hip_forward', 'L')}/"
              f"{pose_sum(r, 'hip_forward', 'R')} armpit {pose_sum(r, 'arm_')} {r.get('error', '')}")
    print(headline(rows))
    return 0


def headline(rows):
    ok = [r for r in rows if "score" in r]
    if not ok:
        return "no rows"
    return (f"median {statistics.median(r['score'] for r in ok):.1f}, mean {statistics.mean(r['score'] for r in ok):.1f}, "
            f"crotch {sum(pose_sum(r, 'hip_forward') for r in ok)}, armpit {sum(pose_sum(r, 'arm_') for r in ok)}, "
            f"failing findings {sum(r['fails'] for r in ok)}, passing {sum(r['fails'] == 0 for r in ok)}/{len(ok)}")


def table(a):
    runs = []
    for tag in a.tags:
        with open(os.path.join(a.out, tag, "rows.json")) as f:
            runs.append({r["id"]: r for r in json.load(f)})
    ids = sorted(set.intersection(*[set(r) for r in runs]))
    lines = ["| rig | " + " | ".join(f"{t}: score/fails, crotch L/R, armpit" for t in a.tags) + " |",
             "|---|" + "---|" * len(a.tags)]
    for i in ids:
        cells = []
        for r in runs:
            x = r[i]
            cells.append(f"{x.get('score', '?')}/{x.get('fails', '?')}, {pose_sum(x, 'hip_forward', 'L')}/"
                         f"{pose_sum(x, 'hip_forward', 'R')}, {pose_sum(x, 'arm_')}")
        lines.append(f"| {i} | " + " | ".join(cells) + " |")
    lines.append("")
    for t, r in zip(a.tags, runs):
        lines.append(f"{t}: {headline([r[i] for i in ids])}")
    if a.pick and len(runs) >= 2:
        # What the rig route keeps: per rig, the better of the first and
        # the last run, as it ranks tries (fewer failing findings, then
        # score; the first run on a tie).
        key = lambda x: (x.get("fails", 99), -x.get("score", 0.0))
        picked = [min((runs[0][i], runs[-1][i]), key=key) for i in ids]
        took = sum(p is runs[-1][i] for p, i in zip(picked, ids))
        lines.append(f"pick of {a.tags[0]} and {a.tags[-1]} ({took}/{len(ids)} take {a.tags[-1]}): {headline(picked)}")
    text = "\n".join(lines) + "\n"
    if a.md:
        with open(a.md, "w") as f:
            f.write(text)
    print(text)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--in", dest="inp", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--weights", required=True)
    r.add_argument("--probe", required=True)
    r.add_argument("--suffix", default=".std.glb")
    r.add_argument("--no-fix", action="store_true")
    r.add_argument("--fix-args", default="")
    r.add_argument("--pre", action="append", default=[])
    r.add_argument("--jobs", type=int, default=2)
    r.add_argument("--threads", type=int, default=3)
    r.add_argument("--only", default="")
    t = sub.add_parser("table")
    t.add_argument("--out", required=True)
    t.add_argument("tags", nargs="+")
    t.add_argument("--md", default="")
    t.add_argument("--pick", action="store_true", help="also the per-rig better of the first and last run")
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else table(a)


if __name__ == "__main__":
    sys.exit(main())
