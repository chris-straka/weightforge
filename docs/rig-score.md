# The rig checker's headline (2026-11-13)

What `weights check` now leads with, why, what was fixed in the checker
on the way, and the numbers the lines were set from. The owner-facing
write-up with pictures is outside this repo (the game's `tmp/rig-score/`).

## Why

The 0-100 score is half the mean and half the single worst region, so it
counts faults rather than measuring quality: hand-made professional rigs
scored 3 to 70, the owner's own generated cast 20 to 78. The owner cannot
judge grey extreme poses and said what they do notice (2026-11-12): cloth
moving with the wrong bone, cloth not following its limb, stiff things
bending, limbs passing through each other, big stretch. The headline
measures those, on the game's own clips as well as on the poses.

## The headline

Per pose group (the game's clips, then the range-of-motion poses):

- **tears**: the share of the surface (area) inside failing findings.
  The vertex share is kept beside it (`tears_verts_pct`; the bench shelf
  and wrap-vs-graft reports used it).
- **findings as areas**, each named by part and side ("left thigh, outer
  side"), with the frame where it is worst and the amount, all on a body
  scaled to 1.75 m:
  - `wrong_bone`: a vertex is pulled more than 6 cm off where its own
    bones would put it by weight on a bone it does not belong to. It
    belongs to its own part's bones; the torso and head blend; torso skin
    may follow a limb's first bones (shoulder, upper arm, thigh: armpit,
    groin, a dog's flank) but not its forearm, hand, shin or foot; a limb
    never follows another limb.
  - `not_following`: skin round the middle of a limb (30-80% along the
    bone, within two limb radii) more than 5 cm and half a limb radius off
    the motion of its own bone and its children, held back by a bone off
    the limb.
  - `stiff_bends`: a part that should move as one piece is more than
    1.5 cm and 8% of its size off its best rigid motion. Stiff parts are
    pieces named for one (belt, buckle, pouch, plate, armour, bag,
    scabbard, bracer...) and small solid loose shells off the head.
  - `limb_through`: skin of one part inside another (arm into torso, leg
    into leg), per pair of parts at its worst frame. Reported beside the
    tears (`through_cm2`), not in them: the clips cause it as often as the
    weights (a hand swung into the chest, a body lying on its arm). A
    failing finding on the clips, a warning on the poses (full bends poke
    by nature).
  - `stretch` (past 3.5x, the owner's line of 2026-11-11), `volume`,
    `bleed`, `noise` (speckle that shows), `unweighted`, `piece_follow`,
    as areas. `piece_bones` (rfcheck's contract rule) stays in the
    findings list, not the headline.

The score, `pass` and the old findings list are unchanged in the JSON
(genforge reads them; `fix` gates on them).

## The clips

`--motion hero.glb` plays the source's clips (default walk, run,
attack_1, dodge, hit, death) on the checked rig, put on by bone role:
per frame, each source bone's world rotation change from rest is applied
to the target bone of the same role after aligning their rest
directions, with a yaw when the rigs face different ways (motionforge
`animate`'s transfer). Measured against motionforge's own output for the
same rig without its contact pass: within 1.8 cm at every joint. The
contact pass (which eases hands out of the body) moves wrists up to
12 cm more, so a rig the game uses is best checked as motionforge
animated it (its own clips): that is what the game plays. The role
transfer serves rigs motionforge cannot map (CesiumMan, Victoria, the
fox). Clips are sampled at 15 frames a second.

## What was wrong in the checker, and the fix

- **Only the first skin was read.** Every skin's joints now merge into
  one list; each primitive's rest shape comes from its own skin; writing
  maps back per skin and appends a joint a skin lacks.
- **Bone roles came from substrings** (`ring` in `Eye_Ring` made it a
  finger; `bone_N` skeletons had no roles). Roles now come from structure
  and the mesh (`core/src/anatomy.rs`): feet are chains ending at the
  floor, clustered by distance or by meeting low in the tree; legs climb
  from them to the joint holding no other foot; the heaviest legs win
  (skirt chains weigh less). Two legs and an upright spine is a humanoid:
  the spine runs from the hips (where the legs meet) to the joint owning
  the crown; arms are the subtrees reaching farthest to each side; the
  head is the joint above the chest with the most vertices. In a leg the
  thigh is the last joint at hip height and the knee the joint halfway
  down; in an arm the hand is the first joint past 72% of the arm's
  reach, the upper arm starts after a clavicle (a whole-word name hint
  inside the chain structure found) or at the first joint clear of the
  midline, and the elbow is the joint nearest the middle of the arc.
  Four legs: a quadruped (`poses/quadruped.ron`); other counts: legged
  (leg swings and bends). Sides come from position. On the owner's cast
  and graft the roles equal the old name roles exactly; on the shelf they
  fix CesiumMan and RiggedFigure (generic before), the fox, horse and pug
  (quadrupeds now), and the VRM arms (roll bones were taken for arms).
- **Bone tails by one global vote**: 30 grafted finger bones following
  Blender's +Y convention flipped it for the whole body and swung the
  thighs ~55 degrees (a false shin bleed). Each joint is now decided on
  its own: it ends at the child on its +Y axis, else at the child
  continuing the chain.
- **Rigid-parented parts were untested.** An unskinned mesh under a joint
  is loaded riding it (`Model::rigid`), checked (it can pass through the
  body), never written, never changed by `fix`.
- **Noise tripped on 24 of 25 shelf models.** A speckle now counts only
  where it shows (some pose kinks it off its neighbours by a third of an
  edge) and only when it stands out of its neighbours (a coarse mesh's
  joint vertex halfway between them does not). Low-poly rigs: 5-14
  findings each before, 0 now; Snow 65 before, 2.
- **Flat skeletons** (deform bones exported without the bones joining
  them: 309 of Victoria's 312 joints were roots) bent one bone at a time,
  so a forearm bend left the hand behind. Such joints are now linked by
  position (`Skeleton::link_flat`) and posed through the links.

## Calibration (`bench/rig_score.py`, 2026-11-13)

Tears on the hero's six clips, % of the surface:

- hand-made game rigs (Quaternius UBC male/female, knight, two robots;
  Khronos CesiumMan, RiggedFigure; both VRM samples; the MPFB base):
  median 0.36 (0 to 1.0; the knight 7.1, its armour collapses lying
  down in the death clip). The old score's median for them was 15.
- the owner's ten generated cast rigs: median 4.1 (1.0 to 10.6).
  Old score median 61.
- the game's hero file: 8.8.

The lines were set between the two groups: on these clips hand-made
rigs reach 4-8 cm (wrong bone) and 2-4 cm (not following), the cast
7-70 cm. Tried and left out, with numbers:

- the belt line (the ring of torso skin between the hips and the first
  spine joint) as a stiff part on the clips: hand-made waists bend as
  much in a run (3.3 cm) as the game hero's (2.5 cm);
- painted parts on a fused body (a band of one colour round the waist,
  a patch unlike its surroundings): over 36 rigs they found one patch,
  the tip of a beard; the cast hero's belt is the colour of the vest it
  sits on.

So a belt on a fused generated body is not found yet; a modelled-apart
or named one is.

## Known limits

- The Blender Studio rigs on the shelf are film rigs: their deform bones
  follow controls through constraints, IK stretch bones deform, and
  correctives are shape keys. Neither their deform-only export (flat)
  nor a game-skeleton rebuild in Blender poses like the rig does in
  Blender; five of ten read as humanoids after the rebuild. Their numbers
  measure those files as a game would play them, not the rigs.
- Four-legged and other bodies get no clips (the hero's are for
  humanoids); their headline is on the poses.
- Quaternius animals weight their hooves to IK foot bones under the root:
  bending a leg leaves the hoof behind. That is reported (wrong bone);
  their own clips would key those bones.
