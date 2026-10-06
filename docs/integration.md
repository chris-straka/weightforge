# Integrating weightforge

Where it sits in the HLL character chain: after rigging (Tripo auto-rig,
SkinTokens, rigforge, wrapforge) and rfcheck, before animation and Godot
export. It only ever rewrites `JOINTS_0`/`WEIGHTS_0`; topology, shape, bone
names and hierarchy belong to retopoforge, wrapforge, rigforge.

## The loop a pipeline runs

```sh
weights check rig.glb --json > check.json        # exit 0: pass, 1: faults, 2: error
# on exit 1:
weights fix rig.glb --out rig.fixed.glb --sheet rig.ab.png \
    [--skintokens] [--source base.glb]  # writes rig.fixed.report.json
weights check rig.fixed.glb --json > recheck.json  # exit 0 means fixed
```

If `fix` exits 1 the output is still never worse than the input (it may
be byte-identical to it); route the asset to a re-rig (wrapforge for
humanoids, rigforge for monsters) and show the owner `rig.ab.png`.

## Exit codes

| code | check | fix | sheet | compare/dump/fixture |
|---|---|---|---|---|
| 0 | no failing finding | output passes | input passes | success |
| 1 | failing findings | output still fails (but improved or unchanged) | input fails | – |
| 2 | usage / IO / unreadable GLB | same | same | same |

## `check --json`

```json
{
  "tool": "weightforge", "version": "0.1.0", "file": "rig.glb",
  "class": "humanoid | custom", "verts": 4879, "tris": 9600, "bones": 20,
  "poses": 35, "scale": 1.93, "score": 9.7, "pass": false,
  "findings": [
    {"code": "D_BLEED", "severity": "fail | warn", "region": "DEF-hand.L",
     "verts": 161, "detail": "161 verts weighted to DEF-thigh.L (...)"}
  ],
  "regions": [
    {"name": "DEF-hand.L", "label": "left hand", "verts": 165, "bad": 165,
     "score": 5.3, "energy": 1.665, "flags": {"bleed": 161},
     "worst_pose": "hip_out.L"}
  ],
  "worst_poses": [{"pose": "spine_back", "bad": 37}],
  "thresholds": {"stretch": 2.5, "thin": 0.5, "...": "..."}
}
```

Region names are bone names, or `piece:<mesh name>` for detachable pieces
(mesh names containing cape, cloak, hair, ponytail, braid). Codes:
`D_STRETCH`, `D_VOLUME`, `D_BLEED`, `D_NOISE`, `D_PIECE_FOLLOW`,
`D_PIECE_BONES`, `D_UNWEIGHTED` (fail); `D_INTERSECT`, `D_SEAM` (warn).
Meanings and thresholds: [phase-0.md](phase-0.md).

## `fix` report (`<out stem>.report.json`)

```json
{
  "input": "rig.glb", "methods": ["smooth", "geodesic", "transfer", "optimize"],
  "before": {"score": 9.7, "pass": false, "fails": 4},
  "after":  {"score": 89.6, "pass": true, "fails": 0},
  "improved": true, "pick_exact": true, "external_kept": false, "verts_changed": 225, "mean_l1_change": 0.0123,
  "max_influences": 4,
  "candidates": {"original": 61.0, "despeckle": 70.2, "geodesic-band": 68.0},
  "regions": [
    {"name": "DEF-hand.L", "label": "left hand", "before": 5.3, "after": 92.8,
     "chosen": "despeckle", "candidates": {"original": 5.3, "despeckle": 92.8}}
  ],
  "findings_before": [], "findings_after": []
}
```

`pick_exact` (since 2026-10-05): the region pick searched every plan.
False only when the search hit its node budget (rigs with a dozen or more
failing regions); the fix is then the best plan found, still never worse
than the input. `external_kept`: with `--candidate`/`--skintokens`, false
when the run without them scored better and was written.

## Custom range of motion

`--poses my.ron` replaces the built-in set (`poses/humanoid.ron` is the
template). Bones are roles (`forearm.L`, `spine*`, `fingers.R`) or exact
bone names; `Bend(Forward, 140)` turns a bone toward a character direction,
`Twist(80)` about its own axis; `mirror: true` emits `.L` and `.R`.
