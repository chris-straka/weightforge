# P2 — fix: clean-up, geodesic, picking, seams

Gate: on fixtures, weight error vs ground truth drops ≥80%; deformation
score better on every faulted region; output passes rfcheck. **Met**
(`rust/core/tests/p2_gate.rs`).

| fixture | weight error before → after | drop | score | verdict |
|---|---|---|---|---|
| noise | 0.00781 → 0.00113 | 85% | 30.6 → 89.6 | FIXED |
| bleed | 0.01320 → 0.00000 | 100% | 9.7 → 89.6 | FIXED |
| no_elbow_falloff | 0.00433 → 0.00004 | 99% | 58.8 → 89.6 | FIXED |
| cape_wrong_bone | 0.06128 → 0.00545 | 91% | 18.1 → 89.6 | FIXED |
| elbow_collapse | 0.02190 → 0.00005 | 100% | 28.6 → 89.6 | FIXED |
| knee_wide_falloff | 0.01483 → 0.00186 | 87% | 77.4 → 89.4 | FIXED |
| clean | unchanged (byte-identical file) | – | 89.6 | – |

Weight error = mean over vertices of half the L1 distance to ground truth.
Every output: ≤4 influences, stored sums exactly 1 in the accessor's own
type, rfcheck exit 0, vertex positions and bone names identical, same
file size (in-place write).

## Methods (P2)

- **despeckle**: drop influences on geodesically far bones (bleed), then
  replace speckled vertices by the componentwise median of their neighbors,
  worst first and in sequence.
- **smooth**: same bleed pruning, Laplacian smoothing only on torn or
  collapsed areas.
- **geodesic**: geodesic voxel binding, `w = (1/((1-a)d + a d²))²`, a=0.5.
- **geodesic-band**: each vertex belongs to its geodesically nearest bone
  and blends with the connected parent/child over one limb radius each
  side of the joint (smoothstep). Restores a standard falloff where the
  input's is missing, too hard, or far too wide.

## Picking (why it is shaped like this)

Greedy, worst failing region first. Each candidate is scored as it would
be applied: only on the flagged area (clusters of ≥3 flagged verts plus a
6-ring margin), seam-blended over 3 rings, input elsewhere. For each
candidate it tries the region alone, with its failing neighbors, and with
all neighbors (a fault across a joint spans two regions), keeping the best
valid set. Preference: the smallest edit among candidates within 25% of
the best score. A move must improve the region by ≥20% (no polishing), the
total must drop, and no region may get worse than 1.1× its input energy +
0.01; if any region then gains a failing finding it did not have, the pass
reruns with no slack, and failing that the input is returned. Candidate 0
is always the input.
