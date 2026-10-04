# P0 — check (detection)

Gate: synthetic fixtures with injected faults: every fault found, 0 false
alarms on the clean twin. **Met** (`rust/core/tests/p0_gate.rs`).

## Fixtures

`weights fixture all --out dir` builds a procedural mannequin (4,879
welded verts, 20 Rigify-style `DEF-*` bones, A-pose, multi-shell like AI
meshes, plus a `Cape` piece) with known-good weights, and six faulted
twins:

| fixture | injected fault | score | failing findings |
|---|---|---|---|
| clean | none | 89.6 | none (PASS) |
| noise | speckle on 50% of torso verts | 30.6 | D_NOISE ×5, D_STRETCH ×3 (torso, upper arms) |
| bleed | left hand 40% on left thigh | 9.7 | D_BLEED hand.L, D_STRETCH/D_VOLUME hand.L, forearm.L |
| no_elbow_falloff | hard 0/1 step at left elbow | 58.8 | D_STRETCH forearm.L |
| cape_wrong_bone | cape 100% on right upper arm | 18.1 | D_PIECE_FOLLOW Cape |
| elbow_collapse | 50/50 band ±12 cm at right elbow | 28.6 | D_STRETCH + D_VOLUME forearm.R, upper_arm.R |
| knee_wide_falloff | smooth but 30 cm-wide right knee blend | 77.4 | D_VOLUME shin.R, thigh.R |

The gate also requires each faulted twin to fail only in regions the fault
touches, and to score more than 5 points below the clean twin.

## Metrics and thresholds (why these numbers)

All thresholds are fractions of the rest-pose bounding-box diagonal or
ratios, so they are scale-free. Poses: built-in humanoid ROM (35 poses;
elbows/knees to 140°), generic ±45° per bone for unknown skeletons, plus
6 samples per animation clip.

- **Stretch** (D_STRETCH): edge length ratio > 2.5, ≥3 verts in a region.
  Clean twin max 2.78 (fewer than 3 verts in any region);
  hard elbow step 6.26; collapse band 8.97.
- **Volume** (D_VOLUME): distance to own bones < 50% of rest, *and* the
  collapsed zone in one pose spans > 1.5 limb radii along the bone. Plain
  linear blend skinning always crushes the joint ring of a 140° elbow to
  ~34% (clean twin min 0.10), so thinness alone false-alarms; zone length
  separates them: clean max 0.71 radii, faults 1.78–2.51.
- **Bleed** (D_BLEED): weight ≥ 0.05 on a bone whose geodesic distance
  (through a 128³ voxelization, Dionne & de Lasa 2013) exceeds the nearest
  bone's by > 15% of body size. Hand-on-thigh: 42%.
- **Noise** (D_NOISE): min(Laplacian residual, lower-median neighbor
  distance) > 0.5. A gradient has ~0 Laplacian, a clean step has ~0 median,
  speckle has both high. Coarse 3-ring falloffs in the clean twin reach
  0.82 median but ~0 Laplacian.
- **Piece follow** (D_PIECE_FOLLOW): attached piece verts (≤3% of size
  from the body, facing within 35°) drift > 8% of size from where Data
  Transfer from the body would carry them; fail when ≥ half of the
  attached verts drift (fewer = warn: an artistic choice).
- **Self-intersection** (D_INTERSECT, warn): triangles intersecting in a
  pose but not at rest, rest centroids > 12% of size apart.
- Also: D_UNWEIGHTED, D_PIECE_BONES (rfcheck's rule), D_SEAM (warn: seam
  copies with different weights).

Region = nearest bone by geodesic distance (bad weights cannot move a
vertex's region), or the piece mesh. Score = 100 / (1 + 10·E), E = half
the vertex-mean energy + half the worst region's; a vertex's energy is
half its pose mean + half its worst pose.

## Speed

Fixture check 0.1–0.3 s; a 54k-vertex, 44-bone real character with 252
generic poses: 1.4 s (M-series Mac, release build).
