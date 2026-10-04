# P4 — real assets

Gate: 10+ Tripo-rigged characters (humanoid + creature) from genforge,
UniRig candidate; the owner rates before/after sheets; ≥8/10 better or
equal. **Not met: blocked on assets and on the owner.**

What exists:

- `bench/run.py [dir]` fixes every GLB in `bench/models/` (gitignored),
  writes A/B sheets, reports, `summary.json`, and a review index; exits 1
  if any asset got worse. `--candidates dir` adds same-stem UniRig outputs.
- Dry run on the 7 local UniRig rigs (assets A–G, owner's corpus):

| asset | verdict | score | failing findings | time |
|---|---|---|---|---|
| A | IMPROVED | 49.7 → 50.9 | 2 → 1 | 0.1 s |
| B | IMPROVED | 21.6 → 42.5 | 47 → 16 | 2.3 s |
| C | IMPROVED | 11.9 → 64.6 | 4 → 2 | 0.2 s |
| D | IMPROVED | 1.9 → 32.5 | 51 → 11 | 3.9 s |
| E | IMPROVED | 1.6 → 77.6 | 17 → 3 | 0.4 s |
| F | IMPROVED | 6.8 → 39.8 | 88 → 17 | 33.8 s |
| G | IMPROVED | 11.9 → 27.3 | 9 → 5 | 7.1 s |

0 worse. None reach PASS: these skeletons have generic `bone_N` names, so
they get the harsh generic ROM (every bone ±45° on two axes plus twist),
and UniRig weights are broadly wrong; the right fix for several is a
re-rig (rigforge/wrapforge), which weightforge does not do by design.
F (54k verts, 44 bones, 252 poses) took 34 s; 12 s since incremental trial scoring.

To finish P4: generate 10+ Tripo-rigged characters with genforge, drop
them in `bench/models/` (UniRig outputs in a candidates folder), run
`bench/run.py --candidates <dir>`, send the A/B sheets to the inbox.
