"""Rest-pose check for a mesh edit: render two models from the same
cameras with their own paint, unlit, and count what differs.

    blender -b --factory-startup --python bench/rest_diff.py -- BEFORE.glb AFTER.glb --out DIR [--size 640]

Views: front, back, left, right over the whole figure, and the hips from
the front, the back and below at four times the scale (where a cut
between the legs would show). Each view is rendered for both models with
flat shading and a transparent background, so a pixel is the paint's own
colour or nothing. Writes `DIR/diff.json` (per view: pixels whose outline
changed, pixels whose colour moved more than 8/255, the mean and largest
colour change inside the shared outline) and `DIR/diff.png` (before,
after, and the difference times ten, a row per view). Prints
`REST_DIFF {json}`: the worst view's numbers.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1:]
    o = {"size": 640, "out": ""}
    pos, i = [], 0
    while i < len(argv):
        if argv[i] == "--out":
            o["out"] = argv[i + 1]
            i += 2
        elif argv[i] == "--size":
            o["size"] = int(argv[i + 1])
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    o["before"], o["after"] = pos
    return o


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    # The importer adds a sphere it draws bones with: not part of the model.
    for ob in [ob for ob in bpy.context.scene.objects if ob.type == "MESH" and not ob.material_slots]:
        bpy.data.objects.remove(ob)
    lo, hi = Vector((1e9, 1e9, 1e9)), Vector((-1e9, -1e9, -1e9))
    for ob in bpy.context.scene.objects:
        if ob.type == "MESH":
            for v in ob.data.vertices:
                p = ob.matrix_world @ v.co
                lo = Vector(map(min, lo, p))
                hi = Vector(map(max, hi, p))
        if ob.type == "ARMATURE":
            ob.data.pose_position = "REST"
            ob.hide_render = True
            ob.hide_viewport = True
    return lo, hi


def render(views, size, tag, out):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "FLAT"
    sc.display.shading.color_type = "TEXTURE"
    sc.display.render_aa = "OFF"
    sc.render.film_transparent = True
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    cam.data.clip_start, cam.data.clip_end = 0.01, 100.0
    paths = []
    for name, target, direction, scale in views:
        cam.data.ortho_scale = scale
        d = Vector(direction).normalized()
        cam.location = Vector(target) + d * 10.0
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(out, f"{tag}_{name}.png")
        bpy.ops.render.render(write_still=True)
        paths.append(sc.render.filepath)
    return paths


def pixels(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    a = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1]
    bpy.data.images.remove(img)
    return a


def main():
    o = args()
    os.makedirs(o["out"], exist_ok=True)
    lo, hi = load(o["before"])
    c = (lo + hi) / 2
    H = hi.z - lo.z
    whole = 1.08 * max(H, hi.x - lo.x)
    hips = Vector((c.x, c.y, lo.z + 0.45 * H))
    close = whole / 4
    views = [
        ("front", c, (0, -1, 0), whole), ("back", c, (0, 1, 0), whole),
        ("left", c, (1, 0, 0), whole), ("right", c, (-1, 0, 0), whole),
        ("hips_front", hips, (0, -1, 0), close), ("hips_back", hips, (0, 1, 0), close),
        ("hips_below_front", hips, (0, -0.6, -0.8), close), ("hips_below_back", hips, (0, 0.6, -0.8), close),
    ]
    a_paths = render(views, o["size"], "before", o["out"])
    load(o["after"])
    b_paths = render(views, o["size"], "after", o["out"])
    rows, report = [], {}
    for (name, *_), pa, pb in zip(views, a_paths, b_paths):
        a, b = pixels(pa), pixels(pb)
        ma, mb = a[..., 3] > 0.5, b[..., 3] > 0.5
        both = ma & mb
        d = np.abs(a[..., :3] - b[..., :3]).max(-1) * 255.0
        d_in = d[both] if both.any() else np.zeros(1)
        report[name] = {
            "outline_changed_px": int((ma != mb).sum()),
            "colour_changed_px": int((d_in > 8).sum()),
            "mean_colour_change": round(float(d_in.mean()), 3),
            "max_colour_change": round(float(d_in.max()), 1),
            "figure_px": int(ma.sum()),
        }
        heat = np.zeros_like(a)
        heat[..., 0] = np.clip(d / 25.5, 0, 1)
        heat[..., 1] = (ma != mb).astype(np.float32)
        heat[..., 3] = 1.0
        bg = np.array([0.09, 0.10, 0.12])
        flat = lambda x: np.concatenate([x[..., :3] * x[..., 3:] + bg * (1 - x[..., 3:]), np.ones_like(x[..., 3:])], -1)
        rows.append(np.concatenate([flat(a), flat(b), heat], 1))
    sheet = np.concatenate(rows, 0)[::-1]
    img = bpy.data.images.new("sheet", sheet.shape[1], sheet.shape[0], alpha=True)
    img.pixels = sheet.astype(np.float32).ravel()
    img.filepath_raw = os.path.join(o["out"], "diff.png")
    img.file_format = "PNG"
    img.save()
    with open(os.path.join(o["out"], "diff.json"), "w") as f:
        json.dump(report, f, indent=1)
    worst = {k: max(v[k] for v in report.values()) for k in ("outline_changed_px", "colour_changed_px", "mean_colour_change", "max_colour_change")}
    print("REST_DIFF " + json.dumps(worst))


main()
