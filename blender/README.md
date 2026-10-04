# WeightForge Blender extension

Skin-weight QA and repair inside Blender, driven by the `weights` CLI from
this repo. Select the armature (or any mesh it deforms), open the
*WeightForge* tab in the 3D Viewport sidebar.

- **Check Weights** poses the rig through its range of motion and paints
  the `weightforge_bad` color attribute on every bound mesh: red = a
  reported fault (tear, collapse, bleed, speckle, piece drift), tan =
  passes through the body (warning), white→pink = soft deformation cost. View it with
  Solid shading, Color: Attribute. The panel lists score and findings.
- **Pose Sheet** renders the range-of-motion sheet into an image datablock
  (shown in any open Image Editor).
- **Fix Weights** runs `weights fix` and writes the result into the bone
  vertex groups in one undo step, and loads the before/after sheet. The fix
  never makes any region worse than the input and never moves a vertex.
  Options: method, a known-good **Source Rig** to transfer from, an extra
  **Candidate** (e.g. UniRig output) to score, **All Regions**, **Use
  Clips**, a custom **Poses** `.ron`, voxel resolution.

How it works: the armature and every mesh bound to it are exported to a
temp GLB (deform bones only, modifiers not applied, so positions are the
meshes' own), the CLI runs on it, and results come back by exact vertex
position. Non-bone vertex groups are left alone.

The CLI is found via the add-on preference, then `PATH`, then
`rust/target/release/weights` in this checkout.

## Test

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --factory-startup --python blender/tests/test_headless.py

Imports the bleed fixture, checks (bleed found, vertices painted), fixes
(no hand vertex left on the thigh, no vertex moved), re-exports, and
confirms the CLI now passes it.
