# P5 — Blender extension and genforge hook

Gate: one end-to-end genforge run: Tripo rig fails check → fixed → passes →
inbox. **Extension met; genforge hook built; live run pending.**
genforge main 494b82c adds `genforge.weights_adapter check|fix` (check
--json; fix --out --sheet --report; fix exit 1 keeps the output as
best-so-far and routes to the re-rig step). Its
`tests/test_weights_adapter.py` runs this repo's binary on the bleed
fixture: check fails → fix 9.7 → 89.6 → recheck passes (5 tests, re-run
here 2026-10-04: green). Still open: one live `gen character` run on a
real Tripo rig through to the inbox (owner schedules it).

- `blender/weightforge/` (GPL, subprocess only): Check Weights (paints
  the `weightforge_bad` color attribute), Pose Sheet (image datablock),
  Fix Weights (writes bone vertex groups, one undo step, loads the A/B
  sheet). Results map back by exact vertex position.
- Headless test `blender/tests/test_headless.py` (Blender 5.2): imports
  the bleed fixture through Blender's glTF importer, check finds the bleed
  and paints 177 verts, fix gives FIXED 9.7 → 89.6 with 0 hand verts left
  on the thigh and 0 vertices moved, the re-exported rig passes the CLI
  check, the sheet renders.

The hook genforge needs (`gen character` step 8/9): run
`weights check in.glb --json`; on exit 1 run `weights fix in.glb --out
fixed.glb --sheet ab.png [--candidate unirig.glb]`, then `weights check
fixed.glb`; exit 0 means pass; send `ab.png` + `fixed.report.json` to
the inbox.
