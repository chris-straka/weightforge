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
  could not be read: every figure small, mild poses first). A headline
  says which rig is better (the gate, then the score; "mixed" when the
  higher score has more failing findings) with the findings and bad-vertex
  totals under it. Then one pose a row, ranked by the larger of the two
  rigs' failing-vertex counts, `--rows` of them (default 5); the row's
  label gives both counts, the larger in red. Both rigs are drawn from the
  same camera, cropped to the same part of the body: the box around both
  rigs' failing vertices, in body heights, never tighter than 0.3 of one.
  The camera is the one of twelve (front or back, left or right, level,
  from below, from above) that shows the most of both rigs' failing
  vertices unhidden. Failing vertices are red on plain grey; a small whole
  figure in each cell's corner boxes the crop. `--poses-b` gives the
  second rig its own pose set when its bones are named differently; poses
  are matched by name, and a pose only one rig has is shown and said.
  Tests: `rust/core/tests/compare_zoom.rs` and `render.rs`'s own.
- Font: own 5×7 bitmap glyphs (uppercase), no third-party font data.
