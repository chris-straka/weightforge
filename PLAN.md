# weightforge — build plan

Automatic skin-weight QA and repair for game-bound characters. Written
2026-10-04 for the HLL asset pipeline (`~/SWE/games/_tools/asset-pipeline.md`)
and genforge's `gen character` chain (`~/SWE/genforge/PLAN.md` P7).

## Why

Tripo's auto-rig and other ML riggers give weights that pass structural
checks (rfcheck: ≤4 influences, weights sum to 1) but still deform badly:
elbows and knees collapse ("candy wrapper"), a hand vertex follows the
thigh, a cape tears off the back, the face pulls with the neck. rfcheck can't
see any of this because it doesn't move the rig. weightforge moves the rig,
measures what breaks, fixes it, and proves the fix with numbers and pictures.

## What it does

```
weights check  hero.glb [--poses rom.ron] [--json]   # score deformation, list bad regions
weights sheet  hero.glb --out sheet.png              # range-of-motion pose sheet, bad verts in red
weights fix    hero.glb --out fixed.glb [--method auto|smooth|geodesic|transfer|optimize]
weights compare a.glb b.glb --out ab.png             # same poses side by side, for the inbox
```

`fix --method auto` builds several weight candidates, measures each on sets
of failing regions, picks the best plan exactly (gate first, then score; a
better candidate never picks a worse plan), blends at region seams with
widths in limb radii (not edge rings), and writes a report
(`fixed.report.json`) with before/after scores. Since 2026-10-05:
[`docs/pick-and-bands.md`](docs/pick-and-bands.md).

## Detection (moving the rig)

Range-of-motion (ROM) pose set per skeleton class (humanoid, quadruped,
custom): shoulders up/forward/back, elbows and knees to 140°, wrist and
ankle twist, spine bend and twist, head turn, finger curls, plus the asset's
own animation clips when present. For each pose, per vertex:

- **Volume loss:** local volume vs rest (catches candy wrapper at twisting
  joints).
- **Edge stretch:** edge length ratio vs rest, beyond a threshold.
- **Wrong-bone bleed:** a vertex weighted to a bone far away along the
  surface (geodesic distance from the bone's segment).
- **Weight noise:** high weight Laplacian (jagged, speckled weights).
- **Self-intersection:** triangles passing through other parts of the body
  (arm into torso, cape into legs).
- **Pieces:** capes/hair/skirts follow the right bones (shares rfcheck's
  `P_PIECE_BONES` rule).

Output: per-region scores (left forearm, right knee, cape...), a JSON
report, and a pose sheet with bad vertices painted red.

## Repair (candidate methods)

1. **Clean-up:** prune to 4 influences, renormalize, remove tiny weights,
   localized Laplacian smoothing on flagged regions only.
2. **Geodesic voxel binding:** voxelize the mesh and compute weights from
   geodesic distance through the voxels (Dionne & de Lasa 2013, the method
   behind Maya's geodesic voxel bind). Robust on non-manifold, multi-shell,
   self-intersecting AI meshes, where Blender's bone heat fails.
3. **Transfer + inpaint:** copy weights from a known-good rigged base where
   the surfaces match closely, and inpaint the rest (robust skin weight
   transfer via weight inpainting, Abdrashitov et al., SIGGRAPH Asia 2023).
   wrapforge's fitted base body is the source for humanoids.
4. **Optimize:** minimize deformation error over the ROM pose set with
   smoothness and locality terms, starting from the best candidate
   (bounded-biharmonic-style constraints: weights in [0,1], partition of
   unity, sparse).
5. **ML candidate:** SkinTokens weights for the rig's own skeleton
   (`weights fix --skintokens` runs `skintokens skin` from
   `~/SWE/blender/skintokens`) as one more candidate, never trusted
   blindly; it gets scored like the rest. SkinTokens replaced UniRig
   (`unirig-mac`, retired) on 2026-10-05: on the genforge rehearsal Andras
   its own rig scores 47.9 raw / 61.6 fixed against UniRig's 10.1 / 33.6
   and the game rig's 22.0 / 37.2 (`skintokens/docs/evaluation.md`).

The original Tripo weights are always candidate 0, so a fix never scores
worse than what came in.

## Stack

- `rust/`: core (MIT): mesh, skeleton, linear blend skinning, metrics,
  voxel geodesics, solvers. Reads/writes GLB with `gltf`. Byte-deterministic.
- `weights` CLI over the core.
- `blender/weightforge/` (GPL): panel with check, sheet, fix, and a heatmap
  view; talks to the CLI over files, like retopoforge.
- Pose sheets rendered headless with Blender, or in-process (wgpu), whichever
  the P0 spike shows is simpler.
- Licences: check every paper's reference code and every crate before use.
  No GPL code in the Rust core.

## Phases and gates

| Phase | Builds | Gate (numbers in `docs/phase-N.md`) |
|---|---|---|
| P0 | Core: GLB read/write, skinning, ROM pose sets (RON), `check` with volume/stretch/bleed/noise metrics, JSON report | Synthetic fixtures with injected faults (noise, hand→thigh bleed, missing elbow falloff, cape on wrong bone): every fault found, 0 false alarms on the clean twin |
| P1 | `sheet`: ROM pose sheet with red bad-vertex overlay; `compare` A/B sheet | Golden images on the fixture mannequin; sheet readable on a phone (checked by eye) |
| P2 | `fix` methods 1-2 (clean-up, geodesic voxel binding) + per-region picking and seam blending | On fixtures: weight error vs ground truth drops ≥80%; deformation score better on every faulted region; output passes rfcheck |
| P3 | Methods 3-4 (transfer + inpaint, optimization) | Beats P2 on the 10 worst regions of the real test set |
| P4 | Real assets: 10+ Tripo-rigged characters (humanoid + creature) from genforge, SkinTokens candidate | Report per asset; the owner judges before/after sheets in the inbox; ≥8/10 rated better or equal |
| P5 | Blender extension + `gen character` hook (genforge P7 calls `weights check`, then `fix` on failure) | One end-to-end genforge run: Tripo rig fails check → fixed → passes → inbox |

## Rules

- Never ship a fix without before/after numbers and a sheet.
- Never move vertices; weightforge only changes weights (topology and shape
  belong to retopoforge and wrapforge).
- Keep the bone names and hierarchy the GLB came with; renaming and
  retargeting belong to rigforge/motionforge.
