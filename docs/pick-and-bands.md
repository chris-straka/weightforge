# Region pick and metric bands (2026-10-05)

Two fixes to `weights fix`, prompted by the SkinTokens Rust port and the
groin-topology work (genforge D62):

1. A better candidate could give a worse fix.
2. Repair bands were counted in edge rings, so they narrowed on denser
   meshes.

Scoring (`weights check`) did not change: old and new binaries give
byte-identical `check --json` on all 39 files measured here (the 7 bench
assets, 27 density meshes and 5 fixtures). No bench score shifts come from
scoring; every change below is in what `fix` writes.

## 1. Region pick

### Root cause

The old pick was greedy, worst failing region first. For each region it
tried candidates on three region sets (alone, with failing neighbours,
with all neighbours), in "smallest edit among those within 25% of the
best" order, and took the first acceptable trial. Every later move had to
beat the current state by 20%. A strong candidate (a Rust SkinTokens
skin) won early on a neighbour set: the right upper arm and right thigh
went to the external candidate at 33.9 and 40.0. Better options (`optimize`
43.0, `helper-band+optimize` 52.1) then could not clear the 20% bar from
there, and `optimize` itself was seeded from that locked mix. A weaker
Python skin did not win those early sets, so the better options got them.
Noise made a skin worse, it lost the early sets, and the fix went up
(46.8 → 49.5).

### What it does now

- **Moves.** One candidate on one region set: alone, with failing
  neighbours, with all neighbours, the group of failing regions within two
  steps of each other (both thighs across the hips), and that group with
  all its neighbours. Each move is measured once on the mesh against the
  input: the change of every touched region's mean energy and flagged
  counts. A candidate's numbers depend only on itself.
- **Exact pick** (`core/src/pick.rs`). Branch and bound over plans
  (disjoint moves). Plans compare on passing the gate first, then score
  energy (`0.5 x vertex mean + 0.5 x worst region`, the formula `check`
  reports), then a small edit cost (0.01 per unit of mean weight change).
  The rules are the old ones: no region above 1.1x its input energy +
  0.01, and no region gains a failing finding. A region a move covers
  takes that move's own numbers; an uncovered region takes the sum of the
  moves' spills (summing onto covered regions counted one improvement
  twice).
- **Proof** (`pick.rs` tests). The search equals brute force on 300 random
  problems. In 2,000 random problems, lowering one candidate's measured
  energies and flag counts never gives a worse picked plan. The pick
  maximises over a fixed set of plans, and every plan's value can only
  improve when a candidate does.
- **Optimize** is seeded from the picked plan (and from the score-only
  plan when the gate pick passes), and everything is picked again.
- **Refinement rounds.** Moves are re-measured against the current plan
  (including "back to the input"), because repairs measured apart miss
  how they meet. The exact pick runs again, up to 4 rounds, while the
  mesh score improves or a broken rule gets repaired.
- **Every stage is judged on the mesh.** The fix is the best stage that
  keeps the rules. If the best-scoring plan breaks a rule, repairs are
  dropped one at a time until it keeps them. The input is still the
  floor.
- **An external candidate never hurts.** With `--candidate` or
  `--skintokens`, fix also runs without the externals and keeps the
  better result (`external_kept` in the report).
- The search has a 2M-node budget per pick. `pick_exact` in the report is
  false when it ran out; that happened only on rigs with 14+ failing
  regions (the 17.0 game rig, generic-named UniRig rigs). There the
  proof above does not apply, but the floors do.

### Evidence

The case from the SkinTokens port (`skintokens/docs/evaluation.md`,
"Rust port"). The input is the standardized 17.0-score game rig of the
Andras rehearsal mesh, and one `skintokens skin` output is the
`--candidate`. Candidates are 8 Python and 8 Rust seeds; Rust seed 4 is
also degraded with multiplicative weight noise (amplitude 0.25 to 2.0,
plus the noisy copy the evaluation used). Candidate quality is the skin's
own `check` score.

| | old (greedy) | new |
|---|---|---|
| Python skins (own 43.5-45.8) | 44.8-56.2, mean 52.2 | 46.4-58.3, mean 49.9 |
| Rust skins (own 44.4-47.1) | 44.6-47.2, mean 45.9 | 52.3-52.5, mean 52.5 |
| Rust s4 clean (own 46.6) | 46.8 | 52.5 |
| + noise 0.25 (43.7) | 45.8 | 58.5 |
| + noise 0.5 (35.3) | 36.6 | 47.3 |
| + noise 1.0 (11.1) / 2.0 (0.7) | 36.6 / 36.6 | 46.4 / 46.4 |
| evaluation's noisy copy (38.3) | 49.5 | 47.3 |
| no candidate | – | 46.4 |

Sheet: `~/Downloads/weightforge-fix/3-candidate-quality-vs-fix.png`.

What this shows:

- Better skins now fix better on average: Rust skins beat Python by 2.6
  points. The old greedy had Python ahead by 6.3.
- Noise no longer helps: 0.5 and above, and the evaluation's noisy copy,
  all fix below the clean skin.
- No candidate drags the fix below no candidate at all (46.4). The old
  greedy fell to 36.6 with a useless candidate.
- Not fixed on this rig: the 0.25-noise skin (58.5) beat the clean one
  (52.5), and two Python seeds beat every Rust seed. This rig has 16
  failing regions, and the search runs out of its node budget on every
  pick (`pick_exact: false`), so the exact-pick proof does not apply and
  noise can land a luckier plan. On rigs with few failing regions (the
  SkinTokens humanoids below: 4-7) the search is exact.

## 2. Metric bands

### Root cause

Every repair width was a ring or pass count: the repair margin (6 rings),
seam blend (3 passes), clean-up margin and smoothing (3 rings, 12
passes), geodesic smoothing (4), piece smoothing (2), optimize margin
(4), and the armpit/groin helper bands (6 rings / 20 passes for arms,
4 / 30 for legs). On a mesh with 1.6x the edges per metre, each band was
1/1.6 as wide in metres. SkinTokens makes the same ring-wide transitions,
so on denser meshes neither the input nor the repair had room, and the
creases tore.

### What it does now

Widths are in feature sizes: `Ctx::feature` is the median distance of
body vertices to their region's bone, a typical limb radius. The helper
bands use the driver's own limb radius. Distances are rest-pose edge
paths (`fix::grow`). Smoothing runs as many passes as the width needs
(`fix::passes`: n passes of lambda 0.5 spread `h·sqrt(n)/2`). The seam
blend is computed from distances, not passes (`fix::blend`): a vertex
keeps `Phi(s/sigma)` of its own candidate. That costs one bounded
Dijkstra per candidate in use, at any density. Diffusion would need
1,616 passes on a 145k-vertex prop.

| band | was | now |
|---|---|---|
| repair margin | 6 rings | 2.8 feature sizes |
| seam blend | 3 passes | Gaussian, 0.4 |
| clean-up margin, smoothing | 3 rings, 12 passes | 1.4, 0.8 |
| geodesic smoothing | 4 passes | 0.5 |
| piece smoothing | 2 passes | 0.57 |
| optimize margin | 4 rings | 1.9 |
| arm helper band (zone, smoothing) | 6 rings, 20 passes | 2.55, 0.95 limb radii |
| leg helper band | 4 rings, 30 passes | 1.6, 1.1 limb radii |

Calibration: each width equals the old ring count on the mesh it was
tuned on. The fix bands match the test mannequin from P2 (mean edge 0.47
feature sizes, cape edge 0.8). The helper bands match the 4.5k-vertex
Andras game mesh (the README tuning; its armpit and groin edges are
0.031-0.043 m against 0.075-0.10 m limb radii). The helper band is also
tried at 0.7x and 1.4x (`helper-band-narrow`, `-wide`), because the right
width depends on the body: wider trades stretch for collapse.

### Evidence

**Fixture mannequin** (`core/tests/density.rs`), built at 0.8x to 2x the
edges per metre. The clean twin is the reference at each density: the
checker's own verdict on ground-truth weights moves with tessellation
(its armpit sits at the 2.5x stretch limit; `*` = fails).

| fault | 0.8x | 1x | 1.3x | 1.6x | 2x |
|---|---|---|---|---|---|
| clean twin | 84.8* | 89.6 | 90.0 | 90.1* | 90.4* |
| elbow_collapse, old | 84.8* | 89.6 | 90.0 | 90.1* | 86.5* |
| elbow_collapse, new | 84.8* | 89.6 | 90.0 | 90.1* | 90.4* |
| knee_wide_falloff, old | 84.8* | 89.4 | 90.0 | 90.1* | 89.2* |
| knee_wide_falloff, new | 84.8* | 89.4 | 90.0 | 90.1* | 90.4* |
| ring_bands, old | 81.1 | 82.2 | 75.3 | 76.4 | 69.9* |
| ring_bands, new | 81.1 | 82.2 | 75.9 | 77.7 | 72.2* |

With metric inputs the fix now lands on the clean twin at every density.
`ring_bands` (every blend one ring wide, as ML riggers make them) is
repaired to no failing finding the clean twin lacks, at every density.
Its score still falls with density, because `fix` only changes flagged
areas, and the narrow blends it leaves alone cost energy without failing.

**Real characters**, three densities each through the genforge chain:
repair-topology (with `--skeleton`), a fresh SkinTokens `rig` (seeds 0,
1, 2), motionforge standardize, then `weights fix` with defaults as the
chain runs it. Score after fix, mean of 3 seeds (seed range), runs that
pass the gate:

| | low | mid | high |
|---|---|---|---|
| Andras, verts | 2,503 | 4,459 | 6,971 |
| old | 73.3 (66.5-77.7), 2/3 | 75.4 (75.2-75.7), 0/3 | 77.8 (77.2-78.6), 0/3 |
| new | 77.2 (73.2-79.3), 3/3 | 76.2 (76.0-76.4), 2/3 | 76.3 (73.2-78.2), 0/3 |
| mannequin, verts | 2,662 | 4,455 | 6,683 |
| old | 78.4 (77.1-81.0), 0/3 | 83.0 (79.4-86.6), 0/3 | 82.7 (78.4-86.3), 0/3 |
| new | 86.1 (84.9-87.2), 2/3 | 83.2 (79.4-87.0), 1/3 | 86.2 (82.5-90.5), 0/3 |
| giraffe, verts | 3,721 | 5,879 | 8,430 |
| old | 41.4 (40.6-41.9) | 35.6 (29.2-42.3) | 55.4 (41.4-81.2) |
| new | 43.0 (42.9-43.1) | 38.1 (34.6-43.2) | 56.0 (42.2-81.6) |

What this shows:

- **Andras:** the spread of the means across densities went from 4.5
  points to 1.0, well inside seed noise (up to 6 points within one
  density). Runs passing the gate: 2/9 → 5/9. High density still keeps
  one left upper arm stretch (7 verts, arm_up) in every seed.
- **Mannequin:** spread 4.6 → 3.0 (seed noise up to 8); passing 0/9 → 3/9.
- **Giraffe:** SkinTokens builds a different quadruped skeleton every run
  (raw 5.4-41.4, 21-27 bones), so its spread is the rig, not the weights.
  The fix is level or better at every density.

Sheets: `~/Downloads/weightforge-fix/2-density-scores-old-vs-new.png`, and
`4-andras-low-density-old-vs-new.png` (seed 0, 2.5k verts: old 66.5 fails
ARM_UP.L, new 73.2 passes).

**Bench (P4 set, 7 UniRig rigs with generic bone names, harsh generic
ROM; old → new):** andras 50.9 → 51.1, bird 42.5 → 35.4, cree 64.6 →
65.0, giraffe 32.5 → 30.9, stalker 77.6 → 66.5 (failing 3 → 1), tira 39.8
→ 60.3, carrot 27.3 → 40.0. None worse than its input, none passes.
Three improved and three dropped. These rigs have 14 to 88 failing
findings, so the pick is not exact. On stalker, gate-first traded 11
points for two fewer failing findings. These rigs are re-rig cases
(rigforge, wrapforge), not weight cases.

## Andras through the genforge chain

genforge `--mode rehearsal` (free mock providers, the real local
adapters), Andras fixture model, SkinTokens rig, same as D62; only the
weightforge binary changed.

| step | before | after |
|---|---|---|
| D62 (old weightforge) | 62.7, 4 failing | 75.7, 1 failing: left upper arm stretch, arm_up, 4 verts → stops at the gate |
| now | 62.7, 4 failing | **76.0, passes** |

Sheet: `~/Downloads/weightforge-fix/1-andras-gate-before-vs-after.png`
(every ROM pose clean, ARM_UP.L 8 → 0).

The chain then goes past the gate for the first time: pose-test (gate),
animate (the game's 9 clips), recheck. The recheck adds 54 clip samples
to the 31 ROM poses and fails at 45.7. The worst frame is the both-arms
overhead swing of `attack_3` at 0.17 s, with 52 bad verts across both
upper arms. Fix 45.7 → 50.1 still leaves upper-arm stretch and volume on
both sides, and the chain stops at its repair limit. Sheet:
`5-after-animate-attack3-fails.png`. The rehearsal mannequin fails the
same swing (genforge phase-7, run B). A shoulder raised past the head
collapses under linear blend skinning whatever the weights. Next lever:
corrective blend shapes for arm_up/arm_forward baked into the clips
(Bevy plays morph weights), or a limit on that swing in the clip.

## Cost

`weights fix` time on the M4, old → new: Andras and the mannequin
0.3 → 1-2 s; the 17.0 game rig 0.9 → about 30 s (16 failing regions:
each pick runs to its node budget, and refinement re-measures about 600
moves per round); bench tira (54k verts) 16 → 106 s and carrot (145k)
6 → 106 s. Moves are measured on the mesh one region set at a time,
against the old greedy's first-acceptable trial. With `--skintokens` or
`--candidate` the fix runs twice (with and without externals).
