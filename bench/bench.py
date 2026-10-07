#!/usr/bin/env python3
"""weightforge benchmark on the CC0 MPFB2 corpus. One command, headless:

    python3 bench/bench.py [--tag NAME] [--only name,name] [--no-sheets]

Builds the CLI and the corpus if missing (bench/corpus/fetch.sh), then for
every character:

  variants   reference  MakeHuman's artist weights (the target)
             heat       Blender automatic weights (free alternative)
             heat+wf    `weights fix` on heat
             ref+wf     `weights fix` on the artist weights (must not damage them)
             broken     artist weights + injected faults (speckle, hand->thigh
                        bleed, binary elbow, head piece on the spine)
             broken+wf  `weights fix` on broken
             messy_*    the same on a cracked, jittered "scan soup" copy,
                        scored against the artist weights carried onto it

measures every variant against the reference in an independent pose set
(bench/eval_weights.py, Blender), records weightforge's own check score and
runtimes, and writes under bench/results/<tag>/:
  summary.json, scorecard.md, <character>_sheet.png (contact sheet: rows =
  variants, columns = poses, red = distance from the artist deformation)
and copies summary.json + scorecard.md to bench/history/<tag>.* (tracked),
so every change is measured against the last run.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLI = os.environ.get("WEIGHTS_BIN", os.path.join(ROOT, "rust", "target", "release", "weights"))
FORGE = os.environ.get("FORGE_BENCH", os.path.expanduser("~/.cache/forge-bench"))
CORPUS = os.path.join(FORGE, "corpus")


def blender_bin():
    for c in (os.environ.get("BLENDER"), shutil.which("blender"),
              "/Applications/Blender.app/Contents/MacOS/Blender"):
        if c and os.path.exists(c):
            return os.path.realpath(c)
    sys.exit("set $BLENDER to the Blender binary")


def run(cmd, **kw):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return p, time.time() - t0


def check_score(glb):
    p, _ = run([CLI, "check", glb, "--json"])
    try:
        r = json.loads(p.stdout)
        return {"score": r.get("score"), "fails": sum(1 for f in r.get("findings", []) if f.get("severity") == "fail")}
    except Exception:
        return {"score": None, "fails": None, "error": (p.stderr or p.stdout)[-300:]}


def fix(src, out):
    p, dt = run([CLI, "fix", src, "--out", out, "--report", out + ".report.json"])
    ok = p.returncode in (0, 1) and os.path.exists(out)
    return {"seconds": round(dt, 2), "exit": p.returncode, "ok": ok, "err": "" if ok else (p.stderr or p.stdout)[-400:]}


def evaluate(blender, ref, variants, out, tiles):
    cmd = [blender, "-b", "--factory-startup", "--python", os.path.join(HERE, "eval_weights.py"), "--",
           "--ref", ref, "--out", out]
    for k, v in variants:
        cmd += ["--variant", f"{k}={v}"]
    if tiles:
        cmd += ["--tiles", tiles]
    p, dt = run(cmd)
    if p.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"eval failed: {p.stdout[-1500:]}{p.stderr[-1500:]}")
    with open(out) as fh:
        return json.load(fh)


def sheet(tiles, rows, out, title):
    from PIL import Image, ImageDraw
    poses = ["rest", "elbow_140.L", "elbow_140.R", "forearm_twist_90.L", "arm_up_150.L", "knee_140.L", "hip_flex_100.L", "spine_twist_60"]
    tw, th, lw, hh = 260, 400, 150, 40
    img = Image.new("RGB", (lw + tw * len(poses), hh + th * len(rows)), (40, 40, 40))
    d = ImageDraw.Draw(img)
    d.text((8, 8), title + "   (red = distance from the artist deformation, full red >= 3% of height)", fill=(255, 255, 255))
    for c, p in enumerate(poses):
        d.text((lw + c * tw + 8, 24), p, fill=(220, 220, 220))
    for r, (label, note) in enumerate(rows):
        y = hh + r * th
        d.text((8, y + 8), label, fill=(255, 255, 255))
        for k, line in enumerate(note.split("\n")):
            d.text((8, y + 28 + 14 * k), line, fill=(200, 200, 200))
        for c, p in enumerate(poses):
            f = os.path.join(tiles, f"{label}__{p}.png")
            if os.path.exists(f):
                img.paste(Image.open(f).convert("RGB"), (lw + c * tw, y))
    img.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=datetime.date.today().isoformat())
    ap.add_argument("--only")
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(CLI):
        subprocess.run(["cargo", "build", "--release", "-j2"], cwd=os.path.join(ROOT, "rust"), check=True)
    if not os.path.isdir(CORPUS) or not os.listdir(CORPUS):
        subprocess.run([os.path.join(HERE, "corpus", "fetch.sh")], check=True)
    blender = blender_bin()
    out = os.path.join(HERE, "results", a.tag)
    os.makedirs(out, exist_ok=True)
    names = sorted(os.listdir(CORPUS))
    if a.only:
        names = [n for n in names if n in a.only.split(",")]
    summary = {"tag": a.tag, "date": datetime.datetime.now().isoformat(timespec="seconds"),
               "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                        text=True).stdout.strip(),
               "characters": {}}
    for n in names:
        src = os.path.join(CORPUS, n)
        meta = json.load(open(os.path.join(src, "meta.json")))
        w = os.path.join(out, n)
        os.makedirs(w, exist_ok=True)
        files = {k: os.path.join(src, k + ".glb") for k in ("ref", "heat", "broken", "messy_ref", "messy_heat")}
        runs = {}
        for k in ("heat", "ref", "broken", "messy_heat"):
            runs[k] = fix(files[k], os.path.join(w, k + "_wf.glb"))
            print(f"{n}: fix {k} {runs[k]['seconds']}s exit {runs[k]['exit']}", flush=True)
        clean = [("heat", files["heat"]), ("heat+wf", os.path.join(w, "heat_wf.glb")),
                 ("ref+wf", os.path.join(w, "ref_wf.glb")), ("broken", files["broken"]),
                 ("broken+wf", os.path.join(w, "broken_wf.glb"))]
        messy = [("messy_heat", files["messy_heat"]), ("messy_heat+wf", os.path.join(w, "messy_heat_wf.glb"))]
        tiles = None if a.no_sheets else os.path.join(w, "tiles")
        e1 = evaluate(blender, files["ref"], [x for x in clean if os.path.exists(x[1])], os.path.join(w, "eval.json"), tiles)
        e2 = evaluate(blender, files["messy_ref"], [x for x in messy if os.path.exists(x[1])],
                      os.path.join(w, "eval_messy.json"), tiles and tiles + "_messy")
        variants = {}
        for k, v in e1["variants"].items():
            variants[k] = {kk: vv for kk, vv in v.items() if kk != "per_pose"}
        for k, v in e2["variants"].items():
            variants["messy_reference" if k == "reference" else k] = {kk: vv for kk, vv in v.items() if kk != "per_pose"}
        glb_of = dict(clean + messy + [("reference", files["ref"]), ("messy_reference", files["messy_ref"])])
        for k in variants:
            if k in glb_of and os.path.exists(glb_of[k]):
                variants[k]["wf_check"] = check_score(glb_of[k])
        summary["characters"][n] = {"verts": meta["verts"], "messy_verts": meta.get("messy_verts"),
                                    "heat_seconds": meta["heat_seconds"], "heat_failed": meta["heat_failed_meshes"],
                                    "messy_heat_failed": meta.get("messy_heat_failed_meshes"),
                                    "fix": runs, "variants": variants}
        if tiles:
            def note(k):
                v = variants.get(k, {})
                if "dev_mean" not in v:
                    return "missing"
                s = v.get("wf_check", {}).get("score")
                return (f"dev {v['dev_mean']:.2f}% / p99 {v['dev_p99']:.2f}%\nbad {v['bad_verts']:.2f}%  bleed {v['bleed_verts']:.1f}%  "
                        f"rough {v['rough']:.3f}\nwf score {s if s is None else round(s, 1)}")
            sheet(tiles, [(k, note(k)) for k in ("reference", "heat", "heat+wf", "ref+wf", "broken", "broken+wf")],
                  os.path.join(out, f"{n}_sheet.png"), f"{n} ({meta['verts']} verts)")
            sheet(tiles + "_messy", [(k, note("messy_reference" if k == "reference" else k))
                                     for k in ("reference", "messy_heat", "messy_heat+wf")],
                  os.path.join(out, f"{n}_messy_sheet.png"), f"{n} messy ({meta.get('messy_verts')} verts)")
        with open(os.path.join(out, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
    write_scorecard(summary, os.path.join(out, "scorecard.md"))
    hist = os.path.join(HERE, "history")
    os.makedirs(hist, exist_ok=True)
    shutil.copy(os.path.join(out, "summary.json"), os.path.join(hist, f"{a.tag}.json"))
    shutil.copy(os.path.join(out, "scorecard.md"), os.path.join(hist, f"{a.tag}.md"))
    print(open(os.path.join(out, "scorecard.md")).read())


ROWS = ["heat", "heat+wf", "ref+wf", "broken", "broken+wf", "messy_heat", "messy_heat+wf"]


def write_scorecard(s, path):
    import statistics as st
    chars = s["characters"]
    L = [f"# weightforge bench `{s['tag']}` ({s['commit']}, {s['date']})", "",
         f"{len(chars)} CC0 MPFB2 characters ({min(c['verts'] for c in chars.values())}-"
         f"{max(c['verts'] for c in chars.values())} verts), 25 extreme poses, scored against "
         "MakeHuman's artist weights. dev = deformed distance to the artist result, % of height "
         "(mean over verts and poses); bleed = % verts with >5% weight on a wrong body part; "
         "rough = weight roughness; stretch = % edges >2x or <0.5x; wf = weightforge's own check score "
         "(0-100). Means over characters.", "",
         "| variant | dev mean | dev p99 | bad % | bleed % | rough | stretch % | wf score | wf fails |",
         "|---|---|---|---|---|---|---|---|---|"]

    def m(k, f, sub=None):
        xs = []
        for c in chars.values():
            v = c["variants"].get(k, {})
            x = v.get(sub, {}).get(f) if sub else v.get(f)
            if x is not None:
                xs.append(x)
        return st.mean(xs) if xs else float("nan")
    for k in ["reference"] + ROWS[:5] + ["messy_reference"] + ROWS[5:]:
        L.append(f"| {k} | {m(k, 'dev_mean'):.3f} | {m(k, 'dev_p99'):.3f} | {m(k, 'bad_verts'):.3f} | {m(k, 'bleed_verts'):.2f} | "
                 f"{m(k, 'rough'):.4f} | {m(k, 'stretch_bad'):.3f} | {m(k, 'score', 'wf_check'):.1f} | "
                 f"{m(k, 'fails', 'wf_check'):.1f} |")
    L += ["", "Per character, bad % = verts more than 1% of height off the artist deformation (lower is better):", "",
          "| character | verts | heat | heat+wf | ref+wf | broken | broken+wf | messy heat | messy heat+wf | fix s (heat) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, c in chars.items():
        v = c["variants"]
        cell = lambda k: f"{v[k]['bad_verts']:.2f}" if k in v and "bad_verts" in v[k] else "-"
        L.append(f"| {n} | {c['verts']} | " + " | ".join(cell(k) for k in ROWS) + f" | {c['fix']['heat']['seconds']} |")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
