"""Before/after of the rig checker on a set of rigs (plain Python 3).

    python3 bench/rig_score.py --out DIR --hero hero.glb --old OLD_WEIGHTS --new NEW_WEIGHTS \
        --motionforge MOTIONFORGE  NAME=path.glb [NAME=path.glb ...]

Per rig, into DIR/NAME/:
- old.poses.json: the old checker on its own poses (`--no-clips`);
- anim.glb: the hero's walk, run, attack_1, dodge, hit and death put on the
  rig by `motionforge animate` (the game's own path, contact pass and all),
  when motionforge can map the rig;
- old.clips.json: the old checker on those clips (a pose set of rest only,
  so its findings are the clips');
- new.json: the new checker: on anim.glb's own clips when motionforge
  animated it, else with `--motion hero.glb` (its role transfer), and its
  poses either way.
Then DIR/rows.json and DIR/table.md.

Tears (old) = vertices in failing findings / vertices, as the bench shelf
and wrap-vs-graft reports measured it. Tears (new) = the share of the
surface inside failing findings (the new headline), limbs inside other
parts apart.
"""

import argparse
import json
import os
import subprocess
import sys

CLIPS = ["walk", "run", "attack_1", "dodge", "hit", "death"]
REST_ONLY = 'PoseSet(name: "rest", poses: [PoseSpec(name: "rest", moves: [])])\n'


def run(cmd, out=None, timeout=1800):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, ""
    if out:
        with open(out, "w") as f:
            f.write(p.stdout)
    return p.returncode, p.stderr


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def old_tears(d):
    if not d:
        return None
    v = max(d.get("verts", 0), 1)
    t = sum(f["verts"] for f in d.get("findings", []) if f.get("severity") == "fail")
    return round(100.0 * t / v, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hero", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--motionforge", required=True)
    ap.add_argument("--only-table", action="store_true")
    ap.add_argument("--new-only", action="store_true", help="keep the old checker's and motionforge's outputs, rerun the new checker")
    ap.add_argument("rigs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clipdir = os.path.join(a.out, "_clips")
    os.makedirs(clipdir, exist_ok=True)
    hero = os.path.join(clipdir, "hero.glb")
    if not os.path.exists(hero):
        os.symlink(os.path.abspath(a.hero), hero)
    rest = os.path.join(a.out, "_rest_only.ron")
    with open(rest, "w") as f:
        f.write(REST_ONLY)
    pick = ",".join(f"hero.glb:{c}={c}" for c in CLIPS)
    rows = []
    for spec in a.rigs:
        name, path = spec.split("=", 1)
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        anim = os.path.join(d, "anim.glb")
        if not a.only_table:
            print(f"== {name}", flush=True)
            if not a.new_only:
                run([a.old, "check", path, "--no-clips", "--json"], os.path.join(d, "old.poses.json"))
                code, err = run([a.motionforge, "animate", "--input", path, "--clips", clipdir, "--pick", pick, "--output", anim],
                                os.path.join(d, "animate.txt"))
                if code != 0 and os.path.exists(anim):
                    os.remove(anim)
                with open(os.path.join(d, "animate.err"), "w") as f:
                    f.write(err)
                if os.path.exists(anim):
                    run([a.old, "check", anim, "--poses", rest, "--json"], os.path.join(d, "old.clips.json"))
            if os.path.exists(anim):
                run([a.new, "check", anim, "--json"], os.path.join(d, "new.json"))
            else:
                run([a.new, "check", path, "--json", "--motion", a.hero], os.path.join(d, "new.json"))
        op, oc, nw = load(os.path.join(d, "old.poses.json")), load(os.path.join(d, "old.clips.json")), load(os.path.join(d, "new.json"))
        row = {"name": name, "file": path, "animated_by": "motionforge" if os.path.exists(anim) else "weightforge roles"}
        if op:
            row.update(old_score=op["score"], old_class=op["class"], old_poses=op["poses"], old_tears_poses=old_tears(op),
                       old_fails=sum(1 for f in op["findings"] if f["severity"] == "fail"))
        if oc:
            row["old_tears_clips"] = old_tears(oc)
        if nw:
            row.update(new_score=nw["score"], new_class=nw["class"], new_poses=nw["poses"], anatomy=nw.get("anatomy", []))
            for sec in nw.get("headline", []):
                key = "clips" if sec["name"] == "game clips" else "poses"
                row[f"tears_{key}"] = sec["tears_pct"]
                row[f"tears_verts_{key}"] = sec["tears_verts_pct"]
                row[f"through_{key}"] = sec["through_cm2"]
                row[f"frames_{key}"] = sec["frames"]
                fails = [f for f in sec["findings"] if f["severity"] == "fail" and f["kind"] != "limb_through"]
                kinds = {}
                for f in fails:
                    kinds[f["kind"]] = round(kinds.get(f["kind"], 0) + f["area_cm2"], 1)
                row[f"kinds_{key}"] = kinds
                row[f"top_{key}"] = [f"{f['kind']}: {f['part']} {f['area_cm2']:.0f} cm2 ({f['worst']})" for f in fails[:3]]
        rows.append(row)
    with open(os.path.join(a.out, "rows.json"), "w") as f:
        json.dump(rows, f, indent=1)

    def n(x, fmt="{:.2f}"):
        return "-" if x is None else fmt.format(x)

    lines = ["| rig | body (old / new) | old score | old tears % poses / clips | new tears % clips / poses | limbs inside, cm² clips | biggest finding on clips |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        top = (r.get("top_clips") or r.get("top_poses") or ["none"])[0]
        lines.append(f"| {r['name']} | {r.get('old_class', '-')} / {r.get('new_class', '-')} | {n(r.get('old_score'), '{:.1f}')} "
                     f"| {n(r.get('old_tears_poses'))} / {n(r.get('old_tears_clips'))} "
                     f"| {n(r.get('tears_clips'))} / {n(r.get('tears_poses'))} | {n(r.get('through_clips'), '{:.0f}')} | {top} |")
    with open(os.path.join(a.out, "table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


main()
