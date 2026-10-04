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
- Font: own 5×7 bitmap glyphs (uppercase), no third-party font data.
