# weightforge

Finds and fixes bad skin weights on rigged game characters, especially
AI-generated ones (Tripo, SkinTokens, UniRig). Rust core + `weights` CLI + Blender
extension, same split as retopoforge. GLB in, GLB out, rfcheck-clean.

rfcheck checks structure (≤4 influences, sums to 1). weightforge moves the
rig through a range of motion, measures what breaks (tears, collapsed
elbows, a hand that follows the thigh, a cape on the wrong bone, speckled
weights), repairs it, and proves the repair with numbers and pictures.

Status: P0–P3 and P5's Blender extension built and gated on fixtures; P4
(owner-rated Tripo set) and the genforge hook are open. Gate numbers:
[`docs/phase-0.md`](docs/phase-0.md) … [`docs/phase-5.md`](docs/phase-5.md).
Spec: [PLAN.md](PLAN.md).

## Build

    cd rust && cargo build --release      # -> rust/target/release/weights

Pure Rust, permissive deps only (serde, serde_json, ron, png, rayon).

## Use

    weights check   hero.glb [--poses rom.ron] [--json] [--vertex-json f.json]
    weights sheet   hero.glb --out sheet.png        # ROM pose sheet, bad verts red
    weights fix     hero.glb --out fixed.glb --sheet ab.png
                    [--method auto|smooth,geodesic,transfer,optimize]
                    [--source base.glb] [--candidate other.glb] [--skintokens]
                    [--all-regions]
    weights compare a.glb b.glb --out ab.png        # same poses side by side
    weights dump    fixed.glb --out weights.json    # per-vertex weights for DCCs
    weights fixture all --out fixtures/             # test mannequins

Exit codes: 0 clean (for fix: the output passes), 1 faults found, 2
usage/IO error. `weights --help` has every flag.

**check** poses the rig with a role-based humanoid set (Rigify `DEF-*`,
Mixamo, Unreal, and plain bone names; elbows/knees to 140°), a generic
per-bone set for anything else, or your own `.ron` (see
`poses/humanoid.ron`), plus samples of the asset's own clips. It reports
per-region scores (0–100) and findings: `D_STRETCH`, `D_VOLUME`,
`D_BLEED`, `D_NOISE`, `D_PIECE_FOLLOW`, `D_PIECE_BONES`, `D_UNWEIGHTED`
(fail), `D_INTERSECT`, `D_SEAM` (warn).

**fix** builds candidates: the input (always candidate 0), despeckle,
smooth, geodesic voxel binding, geodesic joint bands, transfer + inpaint,
optimize (LBS fit to dual-quaternion targets), and any `--candidate`. Each
is scored on the same poses; the best one replaces only the flagged area of
each failing region, blended at seams. No region ends worse than the input
and no region gains a failing finding, or the input comes back unchanged.
`--skintokens` adds the ML candidate: SkinTokens weights for the input's
own skeleton (`skintokens skin`, sibling repo `~/SWE/blender/skintokens`,
~1 min on the M4; opt-in, so default runs stay byte-deterministic).
It writes `fixed.glb` (in place: only JOINTS_0/WEIGHTS_0 bytes change,
never positions or bones), `fixed.report.json` (before/after per region,
method chosen, candidate scores), and with `--sheet` the A/B picture.

## Twist/helper bones

The HLL humanoid has helper bones at the upper arms and thighs
(`DEF-upper_arm_twist.L` ..., added by motionforge's standardize). Each
sits on its driver's joint and turns by a share (0.5) of the driver's
rotation; clips bake that, and weightforge poses them the same way
(`core/src/helpers.rs`: node `extras.hll_helper`, else the `_twist`
name). Helpers are not pose roles and not regions of their own, and
noise is judged on what a vertex does (a helper weight counts as half
driver, half parent), so unweighted helpers leave `check` unchanged.

`fix` weights them: `helper-band` widens each limb/body split around
the joint (6 rings and 20 smoothing passes for arms; 4 and 30 for legs,
hips chain only so one thigh never spreads into the other) and re-blends
it onto the helper, and `helper-band+optimize` refines that with the
LBS-to-DQS fit. Generic candidates leave helpers unweighted (re-blending
them too scored worse). Why: SkinTokens' armpit/groin transition is one
or two edges wide, so a 90 deg swing tears those edges; widening it
alone collapses the joint, and the half-turn helper holds the volume.

Andras game mesh, fresh SkinTokens rig (seed 0), standardized
(2026-10-05; sheets in `~/Downloads/twist-bones/`):

| | raw | after `weights fix` |
|---|---|---|
| without helpers | 47.3 (4 failing) | 62.8 (4) |
| with helpers | 47.6 (4) | 68.8 (2) |

Arms up and right arm forward are clean. Still failing: right thigh
stretch in `hip_forward.R` (9 verts, the crotch strip between the legs)
and left upper arm volume in `arm_forward.L` (7 verts, front fold).
Gate unchanged.

## Blender

`blender/weightforge/` (GPL-3.0-or-later): sidebar tab with Check (heat
map in the `weightforge_bad` color attribute), Pose Sheet, and Fix
(writes vertex groups, one undo). See [blender/README.md](blender/README.md).

## Checks

    cd rust && cargo fmt --all --check && cargo build --release && cargo test --release
    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
        --python blender/tests/test_headless.py
    bench/run.py [models_dir]      # real assets (gitignored), A/B sheets + index
    bench/p3.py  [models_dir]      # P3 vs P2 on the 10 worst regions

## Layout

- `rust/core/` — scene (GLB IO, welding), poses, skinning (LBS/DQS),
  voxel geodesics, metrics, report, render, transfer/inpaint, fix,
  fixtures. MIT.
- `rust/cli/` — the `weights` binary + CLI contract test.
- `blender/` — extension + headless test.
- `poses/` — range-of-motion sets (RON).
- `bench/` — real-asset harnesses; `bench/models/`, `bench/results/` are
  gitignored.
- `tests/fixtures/golden/` — golden sheets.

## References

- Dionne & de Lasa, *Geodesic Voxel Binding for Production Character
  Meshes*, SCA 2013 (geodesic distances, binding falloff). Implemented from
  the paper; no reference code used.
- Abdrashitov et al., *Robust Skin Weights Transfer via Weight
  Inpainting*, SIGGRAPH Asia 2023 (match thresholds, harmonic inpainting).
  Implemented from the paper; no reference code used.
- Kavan et al., *Geometric Skinning with Approximate Dual Quaternion
  Blending*, 2008 (DQS targets for optimize).

License: MIT (Rust core, CLI); GPL-3.0-or-later (Blender extension).
