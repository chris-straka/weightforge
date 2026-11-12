"""Two rigs side by side with their own paint, one leg (or arm) swung
through a pose: stills, or the frames of a loop.

    blender -b --factory-startup --python bench/pose_paint.py -- A.glb B.glb --out DIR
        [--pose hip_forward.L] [--angle 90] [--frames 1] [--size 720]
        [--view front_left_below] [--label-a BEFORE] [--label-b AFTER]

Poses: hip_forward, hip_out, hip_back, knee, arm_up, arm_forward (.L or .R),
as weightforge's humanoid set names them; the bone turns about a world
axis through its head, and a `_twist` helper of the same limb turns half
as far, as the game does. With --frames N > 1 the angle swings 0 -> max
-> 0 over N frames (`DIR/frame_0001.png` ...; join them with ffmpeg).
Each frame is A on the left and B on the right from one camera, lit
flat-ish with the model's texture. Prints `POSE_PAINT {json}`.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

POSES = {  # bone word, world axis, sign of the angle for the left side (Blender: +X left, -Y front, +Z up)
    "hip_forward": ("thigh", "X", -1.0, False),
    "hip_back": ("thigh", "X", 1.0, False),
    "hip_out": ("thigh", "Y", -1.0, True),
    "knee": ("shin", "X", 1.0, False),
    "arm_up": ("upper_arm", "Y", -1.0, True),
    "arm_forward": ("upper_arm", "Z", -1.0, True),
}
VIEWS = {
    "front_left_below": (0.55, -1.0, -0.45),
    "front_right_below": (-0.55, -1.0, -0.45),
    "front": (0.0, -1.0, 0.0),
    "back_left_below": (0.55, 1.0, -0.45),
    "below": (0.0, -0.35, -1.0),
    "back_right": (-0.7, 1.0, 0.1),
    "right": (-1.0, 0.0, 0.0),
}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:]
    o = {"out": "", "pose": "hip_forward.L", "angle": 90.0, "frames": 1, "size": 720, "view": "front_left_below",
         "label_a": "BEFORE", "label_b": "AFTER", "zoom": 2.2}
    pos, i = [], 0
    while i < len(argv):
        k = argv[i][2:].replace("-", "_") if argv[i].startswith("--") else None
        if k in o:
            o[k] = type(o[k])(argv[i + 1])
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    o["a"], o["b"] = pos
    return o


def load(path, name):
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [ob for ob in bpy.context.scene.objects if ob not in before]
    arm = next(ob for ob in new if ob.type == "ARMATURE")
    # The importer adds a sphere it draws bones with: not part of the model.
    for ob in [ob for ob in new if ob.type == "MESH" and not ob.material_slots]:
        new.remove(ob)
        bpy.data.objects.remove(ob)
    for ob in new:
        ob.name = f"{name}:{ob.name}"
    return arm, new


def set_pose(arm, pose, angle):
    base, side = pose.rsplit(".", 1)
    word, axis, sign, flips = POSES[base]
    if side == "R" and flips:
        sign = -sign
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    for pb in arm.pose.bones:
        low = pb.name.lower()
        if word not in low or not low.endswith("." + side.lower()):
            continue
        share = 0.5 if "twist" in low else 1.0
        head = arm.matrix_world @ pb.bone.head_local
        turn = Matrix.Translation(head) @ Matrix.Rotation(math.radians(angle * sign * share), 4, axis) @ Matrix.Translation(-head)
        pb.matrix = arm.matrix_world.inverted() @ turn @ arm.matrix_world @ pb.bone.matrix_local
        bpy.context.view_layer.update()


def main():
    o = args()
    os.makedirs(o["out"], exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    a_arm, a_obs = load(o["a"], "a")
    b_arm, b_obs = load(o["b"], "b")
    for arm in (a_arm, b_arm):
        arm.hide_render = True
    lo, hi = Vector((1e9, 1e9, 1e9)), Vector((-1e9, -1e9, -1e9))
    for ob in a_obs:
        if ob.type == "MESH":
            for v in ob.data.vertices:
                p = ob.matrix_world @ v.co
                lo, hi = Vector(map(min, lo, p)), Vector(map(max, hi, p))
    H = hi.z - lo.z
    base = o["pose"].rsplit(".", 1)[0]
    aim = 0.46 if base.startswith("hip") else 0.3 if base == "knee" else 0.74
    target = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z + aim * H))
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "TEXTURE"
    sc.display.shading.show_cavity = False
    sc.display.shading.show_backface_culling = False
    sc.render.film_transparent = False
    sc.world = bpy.data.worlds.new("w")
    sc.world.color = (0.02, 0.022, 0.028)
    sc.render.resolution_x = sc.render.resolution_y = o["size"]
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = "Standard"
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = H / o["zoom"]
    d = Vector(VIEWS[o["view"]]).normalized()  # from the figure toward the camera
    cam.location = target + d * 10.0
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    n = max(1, o["frames"])
    for f in range(n):
        phase = 1.0 if n == 1 else 0.5 - 0.5 * math.cos(2 * math.pi * f / n)
        halves = []
        for arm, obs, other in ((a_arm, a_obs, b_obs), (b_arm, b_obs, a_obs)):
            set_pose(arm, o["pose"], o["angle"] * phase)
            for ob in obs:
                ob.hide_render = ob.type != "MESH"
            for ob in other:
                ob.hide_render = True
            sc.render.filepath = os.path.join(o["out"], "_half.png")
            bpy.ops.render.render(write_still=True)
            img = bpy.data.images.load(sc.render.filepath)
            w, h = img.size
            halves.append(np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4))
            bpy.data.images.remove(img)
        gap = np.zeros((halves[0].shape[0], 6, 4), np.float32)
        gap[..., 3] = 1.0
        sheet = np.concatenate([halves[0], gap, halves[1]], 1)
        out = bpy.data.images.new("sheet", sheet.shape[1], sheet.shape[0], alpha=True)
        out.pixels = sheet.ravel()
        out.filepath_raw = os.path.join(o["out"], f"frame_{f + 1:04d}.png")
        out.file_format = "PNG"
        out.save()
        bpy.data.images.remove(out)
    os.remove(os.path.join(o["out"], "_half.png"))
    print("POSE_PAINT " + json.dumps({"frames": n, "pose": o["pose"], "left": o["label_a"], "right": o["label_b"]}))


main()
