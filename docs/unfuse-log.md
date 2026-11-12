# Fused legs, arms and coats: what was tried

The fault: on generated humans the strip between the thighs fails
`hip_forward.L/.R` on the same vertices every seed, armpits fail the arm
poses, and long coats drag with the legs. This is the log of the work on
it (2026-11-12): what the fault is, every route tried, the numbers, and
what was kept.

Test set: 20 rigs of the owner's corpus (16 `c01`-`c16`, one seed each;
4 `h01`-`h04`, best of three seeds), about 7,500 triangles and 3,800
welded vertices each, a little under a metre tall. Each starts as the
rig the route makes before `weights fix` (`<id>.std.glb`). Bench:
`bench/unfuse.py` (fix, then `fused_probe`: score, failing findings,
failing vertices per pose). Headline: the set's median score, and its
failing vertices in the two `hip_forward` poses ("crotch") and the six
arm poses ("armpit"). Checker and fixer at e4754bb unless said.

| run | median | mean | crotch | armpit | failing findings | passing |
|---|---|---|---|---|---|---|
| the rigs as made (`raw`) | 55.0 | 52.5 | 873 | 339 | 62 | 1/20 |
| `weights fix` (`base`, the route today) | 58.5 | 56.2 | 623 | 147 | 37 | 6/20 |

## What is fused

`fused_probe --slices` draws the body's outline at eight heights from
the hip joints down. On every rig of the set the outline just below the
hips is one loop round both leg axes: one lump holding both thighs,
almost always with nothing inside it (no thighs, no inner surface). How
far down the lump goes decides how badly the rig fails (failing crotch
vertices on the rig as made):

- about 3 cm below the hip joints (trousers, legs apart): 0 to 40.
- 8 to 15 cm (a tunic or coat skirt over the thighs, thighs standing
  together): 26 to 130.
- 28 cm (wide trousers gathered at the ankle): 266.

So it is not a thin web of skin between two thighs. It is the garment:
one solid body below the hips, which SkinTokens splits down the middle,
left half on the left thigh, right half on the right, switching over one
or two edges. In `hip_forward` one half swings 90 degrees and the edges
across the middle stretch 5 to 70 times. The hardest four (crotch
vertices left/right after `weights fix` 50/80, 22/26, 116/87, 16/14) are
a tunic, a long coat with wide sleeves, wide trousers and a long open
coat. The texture has paint for the lump's outside only; there is no
inner surface to paint.

Armpits: on the long-coat rigs a wide sleeve is joined to the coat's
flank or skirt at the forearm and cuff (not at the armpit), and the join
stretches into a web when the arm lifts. Most rigs' arms are free.

## Tried

1. **Weights only: spread the split across a wedge** (`drape`, dropped).
   The share between a left leg bone and its mirror went from 0 to 1
   across a wedge between the legs that widens with distance from the
   hip axis (0.45 of it), so no edge should stretch past 2.6. On the
   hardest four, score / failing findings / crotch vertices, before any
   fix: 36.3/3/130 -> 29.7/4/228, 30.7/6/60 -> 25.8/8/130, 18.4/8/266 ->
   14.3/8/309, 51.7/5/39 -> 39.0/7/132. The tear goes and a collapse
   comes: a vertex half on a thigh that swung 90 degrees and half on one
   that did not lands inside the lifted thigh (`D_VOLUME`). Linear blend
   skinning cannot make a skirt out of two thigh bones; the lump has to
   be two pieces.
2. **Cut in Blender, before rigging, then rig again** (dropped). The cut
   itself worked, but SkinTokens on the build machine's CPU takes over
   ten minutes a rig and holds a machine-wide lock, and a Blender import
   and export of a rigged model would rewrite the skeleton. The cut
   became a direct edit of the GLB instead (3).
3. **Cut the lump in two in the GLB** (kept): `meshkit/tools/unfuse.py`.
   The plane through the midpoint of the hip joints; the plane's curves
   on the surface below the crotch height are cut where the outline
   goes round both legs; each half is closed with a wall; the cut is
   written into `extras.unfuse`. Then weights (`core/src/unfuse.rs`):
   `own_legs` (a half gives the other leg's bones' weight to its own).
   Points that mattered:
   - *How high to cut.* Sweep of the crotch height, in thigh lengths
     below the hip joints, cut + `own_legs` + crotch falloff, no fixer,
     all 20 (median / mean / crotch / failing findings / rigs worse than
     raw): 0.45: 54.7 / 49.6 / 1288 / 69 / 6; 0.35: 52.5 / 50.8 / 1063 /
     71 / 9; 0.22: 51.6 / 50.8 / 565 / 62 / 12; 0.1: 56.7 / 55.5 / 193 /
     44 / 6; **0.0: 57.5 / 60.1 / 32 / 34 / 1**; -0.1: 57.5 / 60.6 / 52
     / 37 / 1. A cut that stops below the hip joints leaves a stub of
     lump that still splits left/right and tears at the cut's top; at
     the joints the top sits on the axis the thighs turn about.
   - *A falloff at the crotch* (`crotch_band`: skin near the plane
     follows the hips at and above the cut's top, and takes up its leg
     over 0.3 thigh lengths below). With the cut at the hip joints, all
     20, no fixer: without it 58.8 / 60.9 / 53 / 37 / 0 worse; with it
     57.5 / 60.1 / 32 / 34 / 1 worse. Widths 0.2 / 0.45 down and 1.0 /
     2.2 out were no better. Both are fixer candidates (`own-legs`,
     `limbs-apart`); the pick takes whichever does better per region.
   - *The falloff without the cut* is worse than nothing (the four
     hardest: 19.3, 20.7, 9.3, 30.3), so both candidates exist only on a
     file that records a cut.
   - A cut must run through whole stretches of the plane's curve, and
     through large triangles that reach above the crotch height, or it
     stops half way and the wall closes across the gap.
   - Vertices on the cut are told apart by the side of the middle plane
     their outline lies on (with a centimetre of slack, and a vote of
     their neighbours for wall corners), not by which leg axis the
     outline goes round: a tunic's hem is an open wrap that goes round
     neither.
4. **Sleeves** (`unfuse.py --parts legs,arms`, off; not solved). Three
   tries. (a) The plane beside the arm with the smallest section: it
   cut arms that were free (a rig with no armpit fault went 46.7 ->
   18.1). (b) Outlines across the arm, a join where the outline runs on
   round the trunk with a waist: this finds the joins and leaves free
   arms alone (4 of 6 rigs tried: not cut, unchanged). (c) The cut only
   where joined, on a plane through the waists. Where (b) and (c) cut,
   the rigs score far worse after the weights step (40.8 -> 19.0, 56.1
   -> 13.6, 55.9 -> 15.9): skin just across the cut ends up on the
   shoulder while its neighbour is on the forearm. The side test that
   works for legs (outlines across the limb, wholly on one side of the
   plane) mislabels sleeve skin; a wide sleeve's outline across the arm
   is not a clean loop. Left in the tool, off by default.

Not tried: a through-the-volume distance prior as a fixer candidate
(route 2) and transfer from a clean base body (route 4). After the cut
the crotch is ordinary geometry, and what is left there is small.

## Result

All 20, `bench/unfuse.py` (score / failing findings, crotch vertices
left / right, armpit vertices). `base`: `weights fix` on the rig as made.
`own`: cut, each half to its own leg, no fixer. `final`: cut, then
`weights fix` (which starts from `own` and may add the crotch falloff).

| rig | base: score/fails, crotch L/R, armpit | own: score/fails, crotch L/R, armpit | final: score/fails, crotch L/R, armpit |
|---|---|---|---|
| c01 | 63.9/2, 6/7, 0 | 57.0/1, 0/0, 16 | 65.0/0, 0/0, 0 |
| c02 | 53.7/3, 16/14, 14 | 56.1/4, 2/2, 8 | 64.8/2, 2/2, 6 |
| c03 | 54.0/1, 8/10, 0 | 65.7/0, 0/0, 0 | 65.7/0, 0/0, 0 |
| c04 | 50.3/4, 24/20, 6 | 51.5/2, 0/0, 39 | 69.1/1, 0/0, 6 |
| c05 | 78.4/0, 0/0, 0 | 79.1/0, 0/0, 0 | 79.1/0, 0/0, 0 |
| c06 | 49.1/3, 21/25, 0 | 65.5/0, 0/0, 0 | 65.5/0, 0/0, 0 |
| c07 | 66.4/0, 0/0, 0 | 61.7/1, 0/0, 8 | 67.4/0, 0/0, 0 |
| c08 | 62.4/1, 4/4, 0 | 56.9/3, 4/2, 10 | 65.0/1, 4/2, 0 |
| c09 | 36.3/3, 50/80, 0 | 58.1/0, 0/0, 0 | 58.1/0, 0/0, 0 |
| c10 | 64.8/3, 3/5, 8 | 56.2/5, 3/17, 18 | 65.0/2, 2/4, 8 |
| c11 | 33.4/3, 22/26, 83 | 40.8/6, 4/4, 162 | 42.7/4, 4/5, 117 |
| c12 | 61.6/0, 0/0, 0 | 77.3/2, 2/2, 0 | 77.5/1, 2/2, 0 |
| c13 | 57.2/1, 5/7, 0 | 63.1/0, 0/0, 0 | 63.1/0, 0/0, 0 |
| c14 | 49.0/3, 3/6, 26 | 52.3/3, 0/0, 41 | 55.8/1, 0/0, 21 |
| c15 | 20.2/7, 116/87, 0 | 50.9/2, 0/0, 10 | 51.1/1, 0/0, 0 |
| c16 | 65.5/0, 0/0, 0 | 59.5/0, 0/0, 0 | 59.5/0, 0/0, 0 |
| h01 | 55.3/2, 16/20, 10 | 55.9/3, 2/2, 21 | 56.8/3, 2/2, 16 |
| h02 | 74.0/0, 0/0, 0 | 70.0/3, 2/5, 6 | 73.8/0, 0/0, 0 |
| h03 | 59.7/1, 7/11, 0 | 78.6/1, 0/0, 0 | 79.3/0, 0/0, 0 |
| h04 | 69.5/0, 0/0, 0 | 62.2/1, 0/0, 0 | 70.2/0, 0/0, 0 |

| run | median | mean | crotch | armpit | failing findings | passing |
|---|---|---|---|---|---|---|
| `base` | 58.5 | 56.2 | 623 | 147 | 37 | 6/20 |
| `own` | 58.8 | 60.9 | 53 | 339 | 37 | 6/20 |
| cut, fixer with `own-legs` only as a candidate (dropped) | 63.4 | 62.5 | 87 | 169 | 19 | 10/20 |
| **`final`** | **65.0** | **64.7** | **33** | 174 | **16** | **11/20** |
| what the route keeps (better of `base` and `final` per rig; 15 of 20 take the cut) | 65.0 | 63.7 | 100 | 134 | 13 | 12/20 |

- The fixer has to start from `own`, not pick it: offered as one
  candidate among the others, the region pick sometimes left a half on
  the cut's interpolated weights (the "dropped" row; four rigs ended
  below `own`).
- Rest pose: `bench/rest_diff.py` renders the uncut and the cut model
  from eight cameras (four of the whole figure, four of the hips at four
  times the scale, two of those from below), unlit, 512 px. Over the 20:
  at most 5 pixels of outline and 20 pixels of colour differ in any
  view. With the walls 0.1 mm apart (the first setting) a close-up
  straight along the cut showed a broken hairline (up to 59 outline
  pixels); at 0.02 mm it does not. The texture and every old vertex are
  the same bytes.
- Contract: `rfcheck --class npc` is clean on all 20 cut and fixed rigs
  (a cut adds 150 to 600 vertices before welding).
- The route: `motionforge/tools/tripo_rig.py --unfuse` on one rig of the
  set, with SkinTokens stood in for by the rig it made earlier: the
  uncut try 36.3 with 3 failing findings, the cut try 58.1 and passing;
  the cut copy is the one kept.
- The website's own auto-rig, at game budget on two of the characters
  (prepared by another run; today's checker): no failing crotch vertex
  as rigged and 0 and 4 after `weights fix`, whole-rig scores 44.9 and
  15.1. The same two characters here after the cut: 0 failing crotch
  vertices on the three-seed rigs (73.8 and 79.3), 6 and 4 on the
  one-seed ones (65.0 and 77.5).

## Tears and pokes

The 0-100 score is half the single worst region, so one small fault caps
it. Two plain shares of a rig's vertices, from `weights check --json`
(`bench/unfuse.py` puts both in every row): **tears**, vertices in
failing findings; **pokes**, vertices flagged `intersect` (skin through
another part: a warning the score leaves out). Hand-made game rigs sit
at 0.3 to 0.7 % tears and 1.3 to 2.5 % pokes; the target here is tears
at or under 0.7 % without buying it with pokes.

| run | tears median | tears worst | at or under 0.7 % | pokes median | pokes worst |
|---|---|---|---|---|---|
| the rigs as made | 0.62 | 5.35 | 11/20 | 11.7 | 21.1 |
| `base` | 0.27 | 4.92 | 15/20 | 10.9 | 19.9 |
| `own` | 0.23 | 2.46 | 18/20 | 11.5 | 23.4 |
| **`final`** | **0.00** | **1.78** | **19/20** | 11.5 | 23.1 |
| what the route keeps | 0.00 | 1.96 | 19/20 | 11.5 | 21.6 |

| rig | tears %: base -> final | pokes %: base -> final | pokes away from the cut %: base -> final |
|---|---|---|---|
| c01 | 0.19 -> 0.00 | 9.2 -> 9.0 | 9.2 -> 9.2 |
| c02 | 0.66 -> 0.21 | 18.8 -> 18.6 | 18.7 -> 18.7 |
| c03 | 0.32 -> 0.00 | 9.7 -> 11.3 | 9.7 -> 11.4 |
| c04 | 0.86 -> 0.10 | 11.9 -> 11.6 | 12.0 -> 12.0 |
| c05 | 0.00 -> 0.00 | 2.8 -> 2.7 | 2.8 -> 2.8 |
| c06 | 1.08 -> 0.00 | 12.1 -> 15.4 | 12.1 -> 15.1 |
| c07 | 0.00 -> 0.00 | 8.1 -> 8.2 | 8.1 -> 8.4 |
| c08 | 0.11 -> 0.16 | 8.1 -> 8.2 | 8.1 -> 8.3 |
| c09 | 2.56 -> 0.00 | 13.1 -> 13.5 | 13.1 -> 13.2 |
| c10 | 0.32 -> 0.21 | 9.8 -> 9.6 | 9.9 -> 9.9 |
| c11 | 1.96 -> 1.78 | 19.9 -> 23.1 | 19.8 -> 20.9 |
| c12 | 0.00 -> 0.10 | 10.0 -> 9.7 | 10.1 -> 10.1 |
| c13 | 0.21 -> 0.00 | 12.4 -> 12.1 | 12.5 -> 12.5 |
| c14 | 0.64 -> 0.33 | 19.9 -> 21.6 | 19.9 -> 20.6 |
| c15 | 4.92 -> 0.08 | 18.2 -> 21.3 | 18.0 -> 19.8 |
| c16 | 0.00 -> 0.00 | 17.4 -> 21.1 | 17.5 -> 20.8 |
| h01 | 0.36 -> 0.17 | 12.4 -> 12.5 | 12.4 -> 12.7 |
| h02 | 0.00 -> 0.00 | 3.9 -> 3.9 | 3.9 -> 3.9 |
| h03 | 0.16 -> 0.00 | 7.2 -> 7.1 | 7.3 -> 7.3 |
| h04 | 0.00 -> 0.00 | 5.0 -> 5.1 | 5.0 -> 5.2 |
| **median** | 0.27 -> 0.00 | 10.9 -> 11.5 | 11.0 -> 11.7 |

Tears: the hard four go 2.56, 1.96, 4.92, 0.66 to 0.00, 1.78, 0.08,
0.21. The one left above 0.7 % is the long coat whose sleeves are joined
to it (its tears are the sleeves').

Pokes: the cut does trade a little. The set's median goes 10.9 to 11.5,
and six rigs gain 1.7 to 3.6 points: three robe or skirt figures (+1.7,
+3.3, +3.6), the wide trousers (+3.1) and the two long coats with wide
sleeves (+3.2, +1.8). `pokes_probe` counts them per pose and away from
the cut (more than 5 mm from its plane): the rise is not the new inner
walls, it is the garment. Before the cut a lifted leg could not take
its half of a robe with it (the other leg held it, and it tore); now
the half lifts whole and its top folds into the belt and belly at the
hip, as a bare thigh does at 90 degrees. On the wide trousers the gain
is in the knee poses (242 flagged vertices to 362): each trouser leg now
folds at its own knee. The other fourteen rigs move by 0.3 points or
less. Nothing was seen passing through the body that did not before:
`bench/pose_paint.py` pictures of the worst gains show the lifted half
against the waist, and the other half's inner wall where the garment
used to stretch. Every rig of the set, cut or not, is far above the
hand-made rigs' 1.3 to 2.5 % pokes; that is the garments and the 140
degree elbow and knee poses, and this work did not go after it.

## What is left

- Crotch: 33 failing vertices on 7 rigs, none above 9 on one rig.
- Pokes: up 1.7 to 3.6 points on six garment rigs (Tears and pokes). A
  lifted robe half folds into the waist; nothing limits that yet.
- Armpits and sleeves: not solved (Tried, 4). The set's armpit count is
  174 after the cut against 147 before: one long-coat rig whose sleeves
  are joined to the coat's skirt goes 83 -> 117 once the skirt's halves
  follow the legs. The route keeps that rig uncut (it ranks by failing
  findings first).
- Five rigs rank no better cut than uncut and the route keeps them as
  they were: the sleeve rig above, two that were already passing and
  score a little lower cut, one passing rig that gains a finding (four
  crotch vertices), and the other seed of the long open coat.
- The walls are flat and take single texels from their rim. On the set
  they read as the garment's dark inside; a wall has no paint of its own
  and no vertices but its rim, so a long one moves as a few large
  triangles.
- A bare mesh (no skeleton) is supported by the cut but was not measured:
  the set is rigged.
- Routes 2 and 4 were not tried.
