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
    weights compare a.glb b.glb --out ab.png --zoom --label-a THEIRS --label-b OURS --clips clips/
                                                    # which rig is better: worst poses first, close-ups in
                                                    # clay with the failing skin red, a loop for each pose
    weights probe   hero.glb --out cases/           # unmarked loops for a person to judge
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
(fail), `D_INTERSECT`, `D_SEAM`, `D_JOINT_RANGE` (weights on a joint index
past the skin's joints: dropped on load), `D_OPEN_MESH` (the surface is
too open to tell inside from outside) (warn).
The standard humanoid skeleton's neck and head are Rigify's numbered spine
(`DEF-spine.004`/`.005` neck, `DEF-spine.006` head): they get the head
poses, and the spine poses bend `DEF-spine.001`-`.003` only (until
2026-11-12 the numbered neck and head counted as spine, so the head poses
never ran on it and the spine bend was spread over the neck and head). A
pose that cannot run on the skeleton is a `pose not tested:` line (JSON
`skipped_poses`), never left out silently. Distances through the body are
measured in a voxel fill that small openings in the surface do not empty
(see `core/src/voxel.rs`); the report's `solid` says how many openings
there are and whether the fill can be trusted. Unknown flags are a usage
error (exit 2).

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
On a body whose fused legs were cut apart (meshkit `tools/unfuse.py`
records the cut in the file; see "Legs that share one shell" below) the
fix starts from each half following its own leg, and one more candidate
is built: `limbs-apart`.
It writes `fixed.glb` (in place: only JOINTS_0/WEIGHTS_0 bytes change,
never positions or bones), `fixed.report.json` (before/after per region,
method chosen, candidate scores), and with `--sheet` the A/B picture.

## Twist/helper bones

The standard humanoid skeleton has helper bones at the upper arms and thighs
(`DEF-upper_arm_twist.L` ..., added by motionforge's standardize). Each
sits on its driver's joint and turns by a share (0.5) of the driver's
rotation; clips bake that, and weightforge poses them the same way
(`core/src/helpers.rs`: node `extras.helper_bone`, else the `_twist`
name). Helpers are not pose roles and not regions of their own, and
noise is judged on what a vertex does (a helper weight counts as half
driver, half parent), so unweighted helpers leave `check` unchanged.

`fix` weights them: `helper-band` widens each limb/body split around
the joint (a zone of 2.55 limb radii and smoothing 0.95 radii wide for
arms; 1.6 and 1.1 for legs, hips chain only so one thigh never spreads
into the other; on the 4.5k-vertex hero mesh that is the old 6 rings /
20 passes and 4 rings / 30 passes) and re-blends it onto the helper, and
`helper-band+optimize` refines that with the LBS-to-DQS fit. Each is
tried at 0.7x, 1x and 1.4x that width (`-narrow`, `-wide`): wider trades
stretch for collapse, and the right width depends on the body. Generic candidates leave helpers unweighted (re-blending
them too scored worse). A vertex on two limbs (the crotch, between the
thighs) re-blends both at once, sharing the body weight between them, so
it comes out the same on both sides whatever order the helpers are in
(until 2026-11-12 the first thigh's helper counted the other thigh as
body: hips 0.4, thighs 0.3 / 0.3 came out moving as 0.3 / 0.171). Why: SkinTokens' armpit/groin transition is one
or two edges wide, so a 90 deg swing tears those edges; widening it
alone collapses the joint, and the half-turn helper holds the volume.

A stylized hero game mesh, fresh SkinTokens rig (seed 0), standardized
(2026-10-05; sheets in `~/Downloads/twist-bones/`):

|                 | raw              | after `weights fix` |
| --------------- | ---------------- | ------------------- |
| without helpers | 47.3 (4 failing) | 62.8 (4)            |
| with helpers    | 47.6 (4)         | 68.8 (2)            |

Arms up and right arm forward are clean. Still failing: right thigh
stretch in `hip_forward.R` (9 verts, the crotch strip between the legs)
and left upper arm volume in `arm_forward.L` (7 verts, front fold).
Gate unchanged.

Later the same day (exact region pick, bands in limb radii; gate
unchanged): the genforge rehearsal of the hero (repair-topology with
`--skeleton`, 4,459 welded verts) fixes 62.7 -> 76.0 and **passes**,
where it stopped at 75.7 with one left-arm finding. The same character
remeshed at 2.5k / 4.5k / 7k verts now fixes to 77.2 / 76.2 / 76.3 (mean
of 3 SkinTokens seeds; was 73.3 / 75.4 / 77.8), passing in 5 of 9 runs
(was 2). After animation, the game's `attack_3` overhead swing still
fails the recheck. See [`docs/pick-and-bands.md`](docs/pick-and-bands.md);
sheets in `~/Downloads/weightforge-fix/`.

## Legs that share one shell

A generated human is one shell. Where it wears a tunic, a coat or wide
trousers, or stands with its thighs together, the shell below the hips is
one lump holding both thighs; the rig splits the lump down the middle, one
half on each thigh, and it tears there when a leg lifts (`hip_forward`).
No weights repair that: spread the split wider and the skin stops
tearing and collapses instead (measured, `docs/unfuse-log.md`). The
repair is a cut, and it is a mesh edit, so it is not made here:
`meshkit/tools/unfuse.py` slits the lump along the plane between the
legs from the hip joints down and walls each half, and writes the cut
into the mesh's `extras.unfuse`. On such a file `fix`
(`core/src/unfuse.rs`):

- starts from `own_legs`, not from the file's weights (a cut leaves them
  interpolated across it): skin whose outline (a horizontal cut through
  the vertex) lies wholly on one leg's side of the middle plane gives up
  its weight on the other leg's bones to their counterparts on its own
  (thigh to thigh, shin to shin, helper to helper);
- builds one more candidate, `limbs-apart`: that, and a falloff at the
  crotch: skin near the middle plane follows the hips at and above the
  cut's top and takes up its leg over 0.3 thigh lengths below it.

A file that was not cut gets neither, and `fix` runs on it as before
(`check` always reports a file as it is).
`motionforge/tools/tripo_rig.py --unfuse` runs the cut and keeps it only
when the cut copy ranks better. Bench and numbers: `bench/unfuse.py`,
`docs/unfuse-log.md`. Probes (`rust/core/examples`): `fused_probe` (what
joins two limbs, cross-sections, failing vertices per pose),
`worst_edges` (the most stretched edges of a pose), `limbs_apart_try`
(the weights step alone).

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
    bench/unfuse.py run|table ...  # fused legs: fix and score a folder of rigs, lay runs side by side
    blender -b --python bench/rest_diff.py -- A.glb B.glb --out DIR    # rest pose unchanged by a mesh edit?
    blender -b --python bench/pose_paint.py -- A.glb B.glb --out DIR   # two rigs posed with their paint

## Layout

- `rust/core/` — scene (GLB IO, welding), poses, skinning (LBS/DQS),
  voxel geodesics, metrics, report, render, transfer/inpaint, fix,
  unfuse (legs apart), fixtures. MIT.
- `rust/cli/` — the `weights` binary + CLI contract test.
- `blender/` — extension + headless test.
- `poses/` — range-of-motion sets (RON).
- `bench/` — real-asset harnesses; `bench/models/`, `bench/results/` are
  gitignored.
- `tests/fixtures/golden/` — golden sheets.

## References

- Dionne & de Lasa, _Geodesic Voxel Binding for Production Character
  Meshes_, SCA 2013 (geodesic distances, binding falloff). Implemented from
  the paper; no reference code used.
- Abdrashitov et al., _Robust Skin Weights Transfer via Weight
  Inpainting_, SIGGRAPH Asia 2023 (match thresholds, harmonic inpainting).
  Implemented from the paper; no reference code used.
- Kavan et al., _Geometric Skinning with Approximate Dual Quaternion
  Blending_, 2008 (DQS targets for optimize).

License: MIT (Rust core, CLI); GPL-3.0-or-later (Blender extension).
