"""Measure skin weights against a reference, in Blender (headless).

    blender -b --factory-startup --python bench/eval_weights.py -- \
        --ref ref.glb --variant heat=heat.glb --variant fixed=fixed.glb ... \
        --out metrics.json [--tiles tiles_dir]

Every variant must be the same meshes and skeleton as the reference with
different JOINTS_0/WEIGHTS_0 (weightforge only rewrites those; the corpus
exports its heat variant from the same objects), so vertices correspond
one to one.

The pose set is independent of weightforge's own ROM poses on purpose
(no grading our own homework): 25 extreme poses defined by swinging a
bone toward a world direction or twisting it about its own axis, so they
work on any humanoid whose bones carry the usual names.

Metrics (per variant, means over poses; lengths in % of character height):
  dev_mean / dev_p95 / dev_p99 / dev_max  deformed vertex distance to the
                 reference deformation (same pose)
  bad_verts      % verts more than 1% of height away from the reference
  stretch_bad    % edges stretched > 2x or squashed < 0.5x
  area_logp99    99th pct |log(triangle area ratio)| (collapse / bulge)
  flips          % triangles whose normal flips > 120 deg vs rest
  rough          weight roughness: mean |w_v - mean(w_neighbours)| (L1 over
                 bones, rest mesh; lower = smoother)
  bleed_verts    % verts with > 5% weight on a bone the reference does not
                 use within one edge ring of that vertex, nor its parent or
                 children (weight on the wrong body part)
With --tiles, renders each variant in a few poses, coloured by deviation
from the reference (white 0 .. red >= 3% of height), for the contact sheet.
"""

import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
REF, OUT, TILES, VARIANTS = None, None, None, []
i = 0
while i < len(argv):
    a = argv[i]
    if a == "--ref":
        REF = argv[i + 1]; i += 2
    elif a == "--variant":
        k, v = argv[i + 1].split("=", 1); VARIANTS.append((k, v)); i += 2
    elif a == "--out":
        OUT = argv[i + 1]; i += 2
    elif a == "--tiles":
        TILES = argv[i + 1]; i += 2
    else:
        sys.exit(f"unknown arg {a}")

FWD, BACK, UP, DOWN = Vector((0, -1, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)), Vector((0, 0, -1))


def side(x, s):
    return Vector((x.x * (1 if s == "l" else -1), x.y, x.z))


def out_(s):
    return Vector((1 if s == "l" else -1, 0, 0))


# (name, [(bone pattern, kind, direction-or-None, degrees)]); {s} = l/r.
def pose_set():
    poses = []
    for s in "lr":
        S = s.upper()
        poses += [
            (f"elbow_140.{S}", [(f"lowerarm_{s}", "swing", FWD, 140)]),
            (f"forearm_twist_90.{S}", [(f"lowerarm_{s}", "twist", None, 90)]),
            (f"arm_up_150.{S}", [(f"upperarm_{s}", "swing", UP, 110)]),
            (f"arm_forward_90.{S}", [(f"upperarm_{s}", "swing", FWD, 90)]),
            (f"arm_back_50.{S}", [(f"upperarm_{s}", "swing", BACK, 50)]),
            (f"wrist_80.{S}", [(f"hand_{s}", "swing", DOWN, 80)]),
            (f"fist.{S}", [(f"{f}_0{k}_{s}", "swing", DOWN, 80) for f in ("index", "middle", "ring", "pinky") for k in (1, 2, 3)]),
            (f"knee_140.{S}", [(f"calf_{s}", "swing", BACK, 140)]),
            (f"hip_flex_100.{S}", [(f"thigh_{s}", "swing", FWD, 100)]),
            (f"hip_out_45.{S}", [(f"thigh_{s}", "swing", out_(s), 45)]),
            (f"run_stride.{S}", [(f"thigh_{s}", "swing", FWD, 60), (f"calf_{s}", "swing", BACK, 90),
                                 (f"thigh_{'r' if s == 'l' else 'l'}", "swing", BACK, 30)]),
        ]
    poses += [
        ("spine_bend_60", [(b, "swing", FWD, 20) for b in ("spine_01", "spine_02", "spine_03")]),
        ("spine_twist_60", [(b, "twist", None, 20) for b in ("spine_01", "spine_02", "spine_03")]),
        ("head_turn_70", [("neck_01", "twist", None, 35), ("head", "twist", None, 35)]),
    ]
    return poses


POSES = pose_set()
TILE_POSES = ["rest", "elbow_140.L", "elbow_140.R", "forearm_twist_90.L", "arm_up_150.L", "knee_140.L", "hip_flex_100.L", "spine_twist_60"]


def load(path):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.actions):
        for x in list(coll):
            coll.remove(x)
    bpy.ops.import_scene.gltf(filepath=path, bone_heuristic="TEMPERANCE")
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    meshes = sorted((o for o in bpy.data.objects if o.type == 'MESH'), key=lambda o: o.name)
    for o in meshes:
        o.data.shape_keys and o.shape_key_clear()
    if arm.animation_data:
        arm.animation_data.action = None
    return arm, meshes


def rest_pose(arm):
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def apply_pose(arm, ops):
    rest_pose(arm)
    for pat, kind, d, deg in ops:
        pb = arm.pose.bones.get(pat)
        if pb is None:
            continue
        mw = arm.matrix_world
        head = mw @ pb.head
        tail = mw @ pb.tail
        bdir = (tail - head).normalized()
        if kind == "twist":
            axis = bdir
        else:
            axis = bdir.cross(d)
            if axis.length < 1e-6:
                continue
            axis.normalize()
        R = Matrix.Rotation(math.radians(deg), 4, axis)
        world = Matrix.Translation(head) @ R @ Matrix.Translation(-head) @ (mw @ pb.matrix)
        pb.matrix = mw.inverted() @ world
        bpy.context.view_layer.update()


def deformed(meshes):
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for o in meshes:
        e = o.evaluated_get(dg)
        m = e.to_mesh()
        co = np.empty(len(m.vertices) * 3)
        m.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        mw = np.array(o.matrix_world)
        co = co @ mw[:3, :3].T + mw[:3, 3]
        out.append(co)
        e.to_mesh_clear()
    return np.concatenate(out)


def topology(meshes):
    edges, tris, off = [], [], 0
    for o in meshes:
        m = o.data
        e = np.empty(len(m.edges) * 2, dtype=np.int64)
        m.edges.foreach_get("vertices", e)
        edges.append(e.reshape(-1, 2) + off)
        m.calc_loop_triangles()
        t = np.empty(len(m.loop_triangles) * 3, dtype=np.int64)
        m.loop_triangles.foreach_get("vertices", t)
        tris.append(t.reshape(-1, 3) + off)
        off += len(m.vertices)
    return np.concatenate(edges), np.concatenate(tris), off


def weights(arm, meshes):
    names = [b.name for b in arm.data.bones]
    idx = {n: i for i, n in enumerate(names)}
    rows = []
    for o in meshes:
        W = np.zeros((len(o.data.vertices), len(names)))
        gmap = {g.index: idx.get(g.name) for g in o.vertex_groups}
        for v in o.data.vertices:
            for g in v.groups:
                j = gmap.get(g.group)
                if j is not None:
                    W[v.index, j] = g.weight
        s = W.sum(1, keepdims=True)
        W = np.where(s > 0, W / np.maximum(s, 1e-12), 0)
        rows.append(W)
    parent = np.array([idx[b.parent.name] if b.parent else -1 for b in arm.data.bones])
    return np.concatenate(rows), names, parent


def tri_normals_areas(P, T):
    n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a[:, None], 1e-20), a


def roughness(W, E, n):
    acc = np.zeros_like(W)
    cnt = np.zeros(n)
    np.add.at(acc, E[:, 0], W[E[:, 1]])
    np.add.at(acc, E[:, 1], W[E[:, 0]])
    np.add.at(cnt, E[:, 0], 1)
    np.add.at(cnt, E[:, 1], 1)
    ok = cnt > 0
    return float(np.abs(W[ok] - acc[ok] / cnt[ok, None]).sum(1).mean())


def bleed(W, Wref, E, parent):
    used = Wref > 0.01
    ring = used.copy()
    ring[E[:, 0]] |= used[E[:, 1]]
    ring[E[:, 1]] |= used[E[:, 0]]
    allowed = ring.copy()
    for j, p in enumerate(parent):
        if p >= 0:
            allowed[:, p] |= ring[:, j]
            allowed[:, j] |= ring[:, p]
    mass = (W * ~allowed).sum(1)
    return float((mass > 0.05).mean() * 100)


def render_tiles(arm, meshes, P_by_pose, Pref_by_pose, height, label):
    scn = bpy.context.scene
    scn.render.engine = 'BLENDER_WORKBENCH'
    scn.display.shading.light = 'STUDIO'
    scn.display.shading.color_type = 'VERTEX'
    scn.render.resolution_x, scn.render.resolution_y = 260, 400
    scn.render.film_transparent = False
    scn.world = scn.world or bpy.data.worlds.new("w")
    cam = bpy.data.objects.get("bench_cam") or bpy.data.objects.new("bench_cam", bpy.data.cameras.new("bench_cam"))
    if cam.name not in scn.collection.objects:
        scn.collection.objects.link(cam)
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = height * 1.25
    cam.location = (height * 0.55, -height * 2.0, height * 0.55)
    cam.rotation_euler = (math.radians(88), 0, math.radians(15))
    scn.camera = cam
    os.makedirs(TILES, exist_ok=True)
    for pname in TILE_POSES:
        ops = dict(POSES).get(pname, [])
        apply_pose(arm, ops)
        d = np.linalg.norm(P_by_pose[pname] - Pref_by_pose[pname], axis=1) / height * 100
        t = np.clip(d / 3.0, 0, 1)
        col = np.stack([np.ones_like(t), 1 - t, 1 - t, np.ones_like(t)], 1)
        col[:, :3] = col[:, :3] * 0.85 + 0.1
        off = 0
        for o in meshes:
            n = len(o.data.vertices)
            attr = o.data.color_attributes.get("dev") or o.data.color_attributes.new("dev", 'FLOAT_COLOR', 'POINT')
            attr.data.foreach_set("color", col[off:off + n].ravel())
            o.data.color_attributes.active_color = attr
            off += n
        scn.render.filepath = os.path.join(TILES, f"{label}__{pname}.png")
        bpy.ops.render.render(write_still=True)


def main():
    arm, meshes = load(REF)
    E, T, n = topology(meshes)
    rest_pose(arm)
    P0 = deformed(meshes)
    height = float(P0[:, 2].max() - P0[:, 2].min())
    Wref, bone_names, parent = weights(arm, meshes)
    Pref = {}
    for name, ops in [("rest", [])] + POSES:
        apply_pose(arm, ops)
        Pref[name] = deformed(meshes)
    n0, a0 = tri_normals_areas(P0, T)
    l0 = np.linalg.norm(P0[E[:, 0]] - P0[E[:, 1]], axis=1)
    results = {"height_m": height, "verts": int(n), "poses": [p[0] for p in POSES], "variants": {}}
    for label, path in [("reference", REF)] + VARIANTS:
        arm, meshes = load(path)
        if sum(len(o.data.vertices) for o in meshes) != n:
            results["variants"][label] = {"error": "vertex count differs from reference"}
            continue
        W, names, _ = weights(arm, meshes)
        if names != bone_names:
            W = W[:, [names.index(b) for b in bone_names]]
        per, P_by = [], {}
        for name, ops in [("rest", [])] + POSES:
            apply_pose(arm, ops)
            P = deformed(meshes)
            P_by[name] = P
            if name == "rest":
                continue
            d = np.linalg.norm(P - Pref[name], axis=1) / height * 100
            l = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
            r = l / np.maximum(l0, 1e-12)
            nn, a = tri_normals_areas(P, T)
            ok = a0 > 1e-14
            la = np.abs(np.log(np.maximum(a[ok], 1e-20) / a0[ok]))
            flip = (nn[ok] * n0[ok]).sum(1) < -0.5
            per.append({"pose": name, "dev_mean": float(d.mean()), "dev_p95": float(np.percentile(d, 95)), "dev_p99": float(np.percentile(d, 99)),
                        "bad_verts": float((d > 1.0).mean() * 100),
                        "dev_max": float(d.max()), "stretch_bad": float(((r > 2) | (r < 0.5)).mean() * 100),
                        "area_logp99": float(np.percentile(la, 99)), "flips": float(flip.mean() * 100)})
        agg = {k: float(np.mean([p[k] for p in per])) for k in per[0] if k != "pose"}
        agg["dev_worst_pose"] = max(per, key=lambda p: p["dev_p95"])["pose"]
        agg["rough"] = roughness(W, E, n)
        agg["bleed_verts"] = bleed(W, Wref, E, parent)
        agg["per_pose"] = per
        results["variants"][label] = agg
        print(f"EVAL {label}: dev_mean {agg['dev_mean']:.3f}% p95 {agg['dev_p95']:.3f}% "
              f"bad {agg['bad_verts']:.3f}% stretch {agg['stretch_bad']:.3f}% rough {agg['rough']:.4f} bleed {agg['bleed_verts']:.2f}%")
        if TILES:
            render_tiles(arm, meshes, P_by, Pref, height, label)
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=1)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
