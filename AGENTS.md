# weightforge — agent notes

Skin-weight QA and repair for rigged game GLBs. Rust core
(`rust/core`, MIT) + `weights` CLI (`rust/cli`) + Blender extension
(`blender/weightforge`, GPL, subprocess only). Spec and gates: PLAN.md;
measured gate numbers: `docs/phase-N.md`.

## Standing rules

- Commit and push after each completed chunk; never force-push or
  rewrite history. No remote exists yet: ask before creating one.
- Never commit the owner's game assets, nor their names or paths in
  tracked files ("the owner's corpus"). `bench/models/` and
  `bench/results/` are gitignored.
- Never move vertices or rename/reparent bones: weightforge only writes
  JOINTS_0/WEIGHTS_0.
- No GPL code in `rust/`. Check a crate's licence before adding it.
- Headless Blender only (`--background --factory-startup`), and use the
  app binary, not the `~/.local/bin/blender` shim.

## Checks

- `cd rust && cargo fmt --all --check && cargo build --release && cargo test --release`
  must be green with zero rustc warnings.
- Golden images (`tests/fixtures/golden/`): after an intended render
  change, regenerate with `UPDATE_GOLDENS=1 cargo test --release` and
  look at the PNGs before committing.
- Threshold changes: rerun the P0 gate (`core/tests/p0_gate.rs`) and update
  the numbers in `docs/phase-0.md`.
- New CLI flags go in `--help` and the README.
