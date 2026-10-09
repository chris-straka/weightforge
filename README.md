# weightforge

Finds and fixes bad skin weights on rigged game characters, especially
AI-generated ones (Tripo, SkinTokens, UniRig). Rust core + `weights` CLI + Blender
extension, same split as retopoforge. GLB in, GLB out, rfcheck-clean.

rfcheck checks structure (≤4 influences, sums to 1). weightforge moves the
rig through a range of motion, measures what breaks (tears, collapsed
elbows, a hand that follows the thigh, a cape on the wrong bone, speckled
weights), repairs it, and proves the repair with numbers and pictures.

Status: P0–P3 and P5's Blender extension built and gated on fixtures; P4
(owner-rated Tripo set) is open; the genforge hook is built, its live run pending. Gate numbers:
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
optimize (LBS fit to dual-quaternion targets), and any `--candidate`.
Each candidate is measured on sets of failing regions (only their flagged
area changes, blended into the input at the seams), and an exact search
picks the plan that passes the gate if any can, then scores best
(`core/src/pick.rs`): a better candidate never picks a worse plan, and an
external candidate never leaves the fix worse than no candidate.
Refinement rounds re-measure against the current plan; every stage is
scored on the mesh and the best one that keeps the rules is written. No
region ends above 1.1x its input energy and no region gains a failing
finding, or the input comes back unchanged. Band widths (repair margin,
seam blend, smoothing, armpit/groin bands) are in limb radii, not edge
rings, so the fix works the same at any mesh density. Details and
evidence: [`docs/pick-and-bands.md`](docs/pick-and-bands.md).
`--skintokens` adds the ML candidate: SkinTokens weights for the input's
own skeleton (`skintokens skin`, sibling repo `~/Games/_blender/skintokens`,
Rust, ~20 s on the M4; opt-in, so default runs stay byte-deterministic).
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
the joint (a zone of 2.55 limb radii and smoothing 0.95 radii wide for
arms; 1.6 and 1.1 for legs, hips chain only so one thigh never spreads
into the other; on the 4.5k-vertex Andras mesh that is the old 6 rings /
20 passes and 4 rings / 30 passes) and re-blends it onto the helper, and
`helper-band+optimize` refines that with the LBS-to-DQS fit. Each is
tried at 0.7x, 1x and 1.4x that width (`-narrow`, `-wide`): wider trades
stretch for collapse, and the right width depends on the body. Generic candidates leave helpers unweighted (re-blending
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

Later the same day (exact region pick, bands in limb radii; gate
unchanged): the genforge rehearsal of Andras (repair-topology with
`--skeleton`, 4,459 welded verts) fixes 62.7 -> 76.0 and **passes**,
where it stopped at 75.7 with one left-arm finding. The same character
remeshed at 2.5k / 4.5k / 7k verts now fixes to 77.2 / 76.2 / 76.3 (mean
of 3 SkinTokens seeds; was 73.3 / 75.4 / 77.8), passing in 5 of 9 runs
(was 2). After animation, the game's `attack_3` overhead swing still
fails the recheck. See [`docs/pick-and-bands.md`](docs/pick-and-bands.md);
sheets in `~/Downloads/weightforge-fix/`.

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
