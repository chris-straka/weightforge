# P3 — transfer + inpaint, optimize

Gate: beats P2 on the 10 worst regions of the real test set. **Met on the
local real set, which is not yet the intended one** (see below).

## Methods

- **transfer**: with `--source base.glb`, body vertices take the source's
  weights where the surfaces match (≤5% of size, facing within 30°;
  Abdrashitov et al. 2023 defaults), the rest is inpainted harmonically
  (graph Laplace, per bone, CG). Joints map by name, else by nearest head.
  Pieces always: follow the bones at their attached vertices and those
  bones' parent chain, never sibling limbs (a cape follows the spine down
  to the hips, not the arms or thighs; generic nearest-surface transfer
  bound the fixture cape's edges to the arms and tore it).
- **optimize**: per vertex, fit linear-blend weights to the dual-quaternion
  deformation of the best mix so far over the whole pose set (DQS does not
  collapse at bends or twists), plus smoothness and stay-close terms,
  projected gradient on the simplex, bones limited to the start set plus
  geodesically near ones, flagged areas only.
- **external**: `--candidate other.glb` (e.g. UniRig output via
  `~/SWE/blender/unirig-mac`) is mapped by robust transfer and scored like the rest,
  never trusted blindly.

## Result (bench/p3.py)

Local set: 7 rigs auto-rigged by UniRig from the owner's corpus (assets A–G;
names withheld). 10 worst input regions across the set:

| region (asset) | input | P2 only | auto (P3) | P3 via |
|---|---|---|---|---|
| D/r1 | 1.3 | 99.4 | 99.4 | geodesic |
| E/r1 | 1.5 | 60.9 | 68.3 | optimize |
| E/r2 | 1.5 | 98.0 | 98.0 | smooth |
| E/r3 | 5.2 | 100.0 | 100.0 | optimize |
| E/r4 | 5.2 | 69.3 | 81.6 | optimize |
| E/r5 | 5.2 | 100.0 | 100.0 | geodesic-band |
| F/r1 | 5.3 | 26.4 | 33.4 | optimize |
| F/r2 | 6.2 | 94.5 | 99.7 | geodesic-band |
| F/r3 | 6.7 | 56.4 | 63.9 | optimize |
| F/r4 | 6.9 | 49.1 | 51.6 | optimize |

P3 better on 6/10, never worse (ties are regions P2 already brings to
~100). Caveat: the plan's real test set is Tripo-rigged characters from
genforge; none exist on this machine yet, so this is UniRig output.
