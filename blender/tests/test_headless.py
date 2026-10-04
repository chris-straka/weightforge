# Headless end-to-end test for the WeightForge Blender extension.
#
# Run with the Blender app binary (not the ~/.local/bin/blender shim):
#
#   /Applications/Blender.app/Contents/MacOS/Blender --background \
#       --factory-startup --python blender/tests/test_headless.py
#
# Needs a built CLI (rust/target/release/weights or on PATH).
# Exits 0 on pass or SKIP (binary missing), 1 on failure.

import os
import subprocess
import sys
import tempfile

import bpy

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "blender"))

import weightforge  # noqa: E402


def check(cond, msg):
    print(("PASS" if cond else "FAIL") + ": " + msg)
    if not cond:
        raise SystemExit(1)


def main():
    cli = weightforge.find_cli()
    if not cli:
        print("SKIP: weights CLI not built")
        return
    tmp = tempfile.mkdtemp(prefix="wf_blender_test_")
    subprocess.run([cli, "fixture", "bleed", "--out", tmp], check=True, capture_output=True)
    glb = os.path.join(tmp, "mannequin_bleed.glb")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    weightforge.register()
    bpy.ops.import_scene.gltf(filepath=glb)
    arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    body = next(o for o in bpy.context.scene.objects if o.type == "MESH" and o.data.name == "Body")
    for o in bpy.context.scene.objects:
        o.select_set(False)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    s = bpy.context.scene.weightforge

    # Negative path: nothing rigged active.
    bpy.context.view_layer.objects.active = None
    try:
        res = bpy.ops.weightforge.check()
        check("CANCELLED" in res, "check cancels without a rig")
    except RuntimeError:
        print("PASS: check cancels without a rig (raised)")
    bpy.context.view_layer.objects.active = arm

    res = bpy.ops.weightforge.check()
    check("FINISHED" in res, "check runs")
    check(s.summary.startswith("FAIL"), f"check reports the bleed: {s.summary.splitlines()[0]}")
    check("D_BLEED" in s.summary, "bleed finding listed")
    attr = body.data.color_attributes.get("weightforge_bad")
    check(attr is not None, "weightforge_bad color attribute created")
    red = sum(1 for c in attr.data if c.color[0] > 0.99 and c.color[1] < 0.1)
    check(red > 50, f"bad vertices painted red ({red})")

    before = [v.co.copy() for v in body.data.vertices]
    thigh = body.vertex_groups["DEF-thigh.L"].index
    hand = body.vertex_groups["DEF-hand.L"].index
    def bleeding():
        n = 0
        for v in body.data.vertices:
            gs = {g.group: g.weight for g in v.groups}
            if gs.get(hand, 0) > 0.3 and gs.get(thigh, 0) > 0.05:
                n += 1
        return n
    b0 = bleeding()
    check(b0 > 50, f"hand vertices weighted to the thigh before fix ({b0})")

    res = bpy.ops.weightforge.fix()
    check("FINISHED" in res, "fix runs")
    check(s.summary.startswith("FIXED"), f"fix verdict: {s.summary.splitlines()[0]}")
    b1 = bleeding()
    check(b1 == 0, f"no hand vertex weighted to the thigh after fix ({b1})")
    moved = max((a - v.co).length for a, v in zip(before, body.data.vertices))
    check(moved == 0.0, "no vertex moved")
    check("not matched" not in s.summary, "every vertex matched back")
    check(bpy.data.images.get("before_after.png") is not None, "before/after sheet loaded")

    # End to end: export the fixed rig and check it with the CLI.
    out = os.path.join(tmp, "after_blender.glb")
    weightforge.export_glb(bpy.context, out, arm, [o for o in bpy.context.scene.objects if o.type == "MESH"], False)
    proc = subprocess.run([cli, "check", out], capture_output=True, text=True)
    check(proc.returncode == 0, f"re-exported rig passes check: {proc.stdout.splitlines()[0]}")

    res = bpy.ops.weightforge.sheet()
    check("FINISHED" in res and bpy.data.images.get("sheet.png") is not None, "sheet renders into an image")
    weightforge.unregister()
    print("ALL PASS")


try:
    main()
except SystemExit as e:
    sys.exit(e.code)
except Exception:
    import traceback

    traceback.print_exc()
    sys.exit(1)
