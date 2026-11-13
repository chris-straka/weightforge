# Rivals and sources of ideas, 2023-2026 (surveyed 2026-11-13)

Licences were read from each repo's LICENSE on 2026-11-13. Nothing from a
non-commercial or unlicensed repo is copied; ideas only.

## Auto-skinning and auto-rigging

| work | code, licence | idea worth borrowing |
|---|---|---|
| Make-It-Animatable (CVPR 2025) | jasongzy/Make-It-Animatable, MIT | surface normals keep points close in space but far in meaning (inner thigh, armpit) from sharing weights: a normal-agreement term in binding |
| UniRig (SIGGRAPH 2025, 2504.12451) | VAST-AI-Research/UniRig, MIT | learned correction on top of a voxel-binding prior; spring-bone stiffness for hair and cloth (not released) |
| SkinTokens / TokenRig (2602.04805) | VAST-AI-Research/SkinTokens, MIT | RL rewards usable as checks: no vertex above 0.1, more than 4 influences, edge distortion in sampled poses, active-zone IoU; names bleeding onto disconnected parts as the main fault of earlier riggers |
| Puppeteer (NeurIPS 2025) | Seed3D/Puppeteer, Apache-2.0 | skeleton-distance-aware attention: penalise one vertex weighted to bones far apart on the skeleton (both thighs, no pelvis) |
| MagicArticulate (CVPR 2025) | Seed3D/MagicArticulate, Apache-2.0 | diffusion skinning conditioned on volumetric geodesic distance: geodesic stays the core |
| HumanRig (CVPR 2025) | c8241998/HumanRig, MIT | don't trust mesh edges of generated meshes across fused seams |
| RigAnything (SIGGRAPH 2025) | Adobe Research Licence, **non-commercial** | nothing copied |
| Anymate (2025) | **no licence** (all rights reserved) | nothing copied |
| Rigel3D (2605.13129) | no code | - |

## Weight repair and priors

- **A Geodesic Cut-Cell Prior for Neural Skinning** (2608.11272, no code):
  grid rays cut exactly at the surface, inside segments kept by winding
  number; graph geodesics on that don't bridge narrow gaps the way voxel
  geodesics do (hand to torso). Candidate replacement for the voxel graph
  under `voxel.rs`.
- **SkinCells** (2506.14714, Meta, no code): Voronoi-cell sparse weights,
  DeltaMush residual as the smoothness term, tested on skirts.
- **Robust Biharmonic Skinning Using Geometric Fields** (TOG 2024): mesh-free
  biharmonic weights that survive broken topology; code not found.

## Garments and loose cloth

- **TailorRig** (SIGGRAPH Asia 2026, 2611.02902; github.com/Tencent/TailorRig
  announced, 404 today): extra chains for garments, two-level skinning (a
  vertex picks a chain, then weights within it); fewer bleeds between near
  unrelated parts.
- **FreeCloth** (CVPR 2025): split unclothed, tight (skinned) and loose
  cloth (free); skinning snaps a skirt to the nearest leg.
- TailorNet's skirt rule (2020): skirt weights = inverse-distance average of
  the nearest body vertices, no hard switch between the legs.
- HOOD (MIT) and ContourCraft (MIT) depend on SMPL (restricted licence):
  reference only for a later cloth path.

## Separating fused parts

- **HoloPart** (ICLR 2026, VAST-AI-Research/HoloPart, MIT): part completion
  after segmentation; could rebuild inner-thigh surfaces after the unfuse cut.
  Heavy ML.
- **SAMPart3D** (Pointcept/SAMPart3D, MIT): part labels for skirt and coat
  panels (Python, GPU).
- PartField (NVIDIA, **non-commercial**): nothing copied.
- No 2023-2026 paper found that separates self-contacting limbs on a
  generated humanoid; meshkit's unfuse is ahead of published work.

## Measuring deformation

No agreed perceptual metric. The Cut-Cell paper's error: each joint turned
30 degrees about each axis, largest squared gap per vertex, averaged, cm^2
on a 1 m mesh. SkinTokens' bleed and IoU checks as above.

## Ranked for the fused crotch and cloth on the wrong limb (~7k triangles)

1. Cut-cell geodesic graph instead of voxels (stops near-touching thighs and
   armpits bridging; pure geometry, cheap at 7k triangles).
2. Chain first, then bones (TailorRig): label left leg / right leg / torso /
   arm / cloth with a smoothed graph cut, then bind or inpaint within each.
3. Normal-agreement term in binding (Make-It-Animatable).
4. Loose cloth gets a wide blend of pelvis and both thighs (FreeCloth,
   TailorNet) instead of riding one leg.
5. Far-apart-bones and per-joint 30-degree error as checks (SkinTokens,
   Puppeteer, Cut-Cell).

What was tried from this list, and kept or dropped, is logged in the round
reviews (`docs/review-*.md`).
