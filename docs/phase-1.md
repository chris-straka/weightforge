# P1 — sheet and compare

Gate: golden images on the fixture mannequin; sheet readable on a phone
(checked by eye). **Goldens met** (`rust/core/tests/p1_golden.rs`,
`tests/fixtures/golden/`); **phone readability: needs the owner**.

- Renderer: own deterministic software rasterizer (orthographic 3/4
  view, two-sided Lambert, 2× supersampling). Chosen over headless
  Blender/wgpu: no external process or GPU, byte-stable goldens, 0.2 s.
- `weights sheet`: rest + the 11 worst poses (max 2 per bone), reported
  bad verts red, self-intersections orange, a piece that tears off is
  painted whole. 1200 px wide, header text 21 px, labels 14 px.
- `weights compare`: two GLBs on the same poses, A | B per row (600 px
  wide, phone-friendly vertical scroll). `weights fix --sheet` writes the
  input vs fixed compare.
- `weights compare --zoom`: the sheet for judging two rigs at a glance
  (added 2026-11-11, after the whole-body compare of two generated rigs
  could not be read: every figure small, mild poses first, red specks).
  - A headline says which rig is better (the gate, then the score;
    "mixed" when the higher score has more failing findings), with the
    findings, bad-vertex totals and each rig's worst part under it.
  - One pose a row, ranked by the larger of the two rigs' failing-vertex
    counts, `--rows` of them (default 5); the row's label gives both
    counts, the larger in red.
  - Both rigs are drawn from the same camera, cropped to the same part
    of the body: the box around both rigs' failing vertices, in body
    heights, never tighter than 0.3 of one. The camera is the one of
    twelve (front or back, left or right, level, from below, from above)
    that shows the most of both rigs' failing vertices unhidden. A small
    whole figure in each cell's corner boxes the crop.
  - The body is grey clay: smooth normals under one strong side light,
    so stretched and collapsed skin shows as creases and dents. A face
    with a failing vertex is filled solid red.
  - The last row shows where each rig loses its score: every vertex's
    deformation energy over all poses, grey through yellow to red on one
    fixed scale, so faults no single pose paints (speckled or bleeding
    weights) are seen too.
  - `--clips <dir>`: a two-second loop for each row, both rigs moving
    from rest into the pose and back on the row's camera and crop, the
    red arriving with the pose; `<rank>_<pose>.mp4` each and `reel.mp4`
    with every loop twice (ffmpeg; without it the PNG frames stay).
  - `--poses-b` gives the second rig its own pose set when its bones are
    named differently; poses are matched by name, and a pose only one rig
    has is shown and said.
  - Tests: `rust/core/tests/compare_zoom.rs` and `render.rs`'s own.
- `weights probe`: loops for a person to judge, so the pass line can be
  set to their eye (added 2026-11-11: the gate's limits were tuned on the
  test mannequin only, and no real rig has ever passed, the shipped hero
  included). Up to five poses of one rig, the ones whose stretch is
  nearest 1.8, 2.2, 2.6, 3.2 and 4.5 (the third-largest vertex stretch in
  the pose: the limit at which it starts to fail; a pose and its left or
  right twin are used once). Each is a two-second loop from rest into the
  pose and back, clay, framed on the stretched area but never tighter
  than half a body height, with only "CASE NN" on it: nothing marked, no
  verdict. One JSON line a case (rig, pose, stretch, failing vertices) is
  the key the person's "fine" or "broken" is laid against.
  `--number-from` continues the numbers across rigs.
- The skeleton is drawn through the body in every close-up (compare
  rows, their clips, probe cases): a line along each bone and a dot at
  each joint, so the skin can be laid against where it is meant to bend
  (the owner, on a probe case: "idk if the elbow is right there").
- Font: own 5×7 bitmap glyphs (uppercase), no third-party font data.
