"""Build the CC0 benchmark corpus: MPFB2 humans, rigged, two weight sets each.

Run headless with the MPFB2 extension installed in an isolated Blender
profile (bench/corpus/fetch.sh sets that up):

    BLENDER_USER_RESOURCES=$FORGE_BENCH/blender-user blender -b --factory-startup \
        --python bench/corpus/build_mpfb.py -- $FORGE_BENCH/corpus [names...]

Per character, under <corpus>/<name>/:
  ref.glb    MPFB2 game_engine rig with MakeHuman's artist-made weights (the
             reference: hand-painted, CC0)
  heat.glb   the same mesh and rig with Blender's automatic weights (bone
             heat), the free alternative everyone starts from
  meta.json  body settings, mesh/asset list, vertex counts, heat failures

Licences: MPFB2 base mesh, targets, rigs and weights and the MakeHuman
system asset pack are CC0 (static.makehumancommunity.org). The extension
code is GPL but nothing of it ends up in the GLBs.
"""

import json
import os
import sys
import time
import zipfile

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = os.path.abspath(argv[0] if argv else "corpus")
ONLY = set(argv[1:])
PACK = os.environ.get("MH_SYSTEM_PACK", "")

import addon_utils  # noqa: E402
addon_utils.enable("bl_ext.user_default.mpfb", default_set=True)
from bl_ext.user_default.mpfb.services import HumanService, LocationService, AssetService  # noqa: E402

# Body settings span the MakeHuman macro space; meshes span densities
# (proxy741 ~0.7k verts .. full base mesh ~13k) and clothing styles.
CHARACTERS = [
    ("m_avg_casual", dict(gender=1.0), None, ["male_casualsuit01", "shoes01"], "short02"),
    ("f_avg_sport", dict(gender=0.0), "female_generic", ["female_sportsuit01", "shoes02"], "ponytail01"),
    ("m_heavy_work", dict(gender=1.0, weight=1.0, muscle=0.3), "male_generic", ["male_worksuit01", "shoes03"], None),
    ("f_slim_elegant", dict(gender=0.0, weight=0.1, height=0.9), None, ["female_elegantsuit01", "shoes04"], "long01"),
    ("m_muscle_bare", dict(gender=1.0, muscle=1.0, weight=0.6), "male_muscle_13290", [], "short04"),
    ("f_old_casual", dict(gender=0.0, age=1.0, weight=0.7), "female1605", ["female_casualsuit02", "shoes05"], "bob01"),
    ("m_child_lowpoly", dict(gender=1.0, age=0.15), "proxy741", ["male_casualsuit03"], None),
    ("m_tall_suit", dict(gender=1.0, height=1.0, proportions=1.0, weight=0.3), "male1591", ["male_elegantsuit01", "shoes06"], "fedora01"),
]


def ensure_pack():
    data = LocationService.get_user_data()
    if os.path.isdir(os.path.join(data, "clothes", "male_casualsuit01")):
        return
    if not PACK or not os.path.exists(PACK):
        sys.exit("MH_SYSTEM_PACK must point at makehuman_system_assets_cc0.zip (run bench/corpus/fetch.sh)")
    with zipfile.ZipFile(PACK) as z:
        z.extractall(data)
    AssetService.update_all_asset_lists()


def reset():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials):
        for x in list(coll):
            coll.remove(x)


def asset(sub, name, ext):
    p = AssetService.find_asset_absolute_path(f"{name}/{name}.{ext}", asset_subdir=sub)
    if not p:
        raise RuntimeError(f"missing asset {sub}/{name}")
    return p


def build(name, macro, proxy, clothes, hair):
    reset()
    m = {"gender": 0.5, "age": 0.5, "muscle": 0.5, "weight": 0.5, "proportions": 0.5,
         "height": 0.5, "cupsize": 0.5, "firmness": 0.5,
         "race": {"asian": 0.33, "caucasian": 0.33, "african": 0.33}}
    m.update(macro)
    base = HumanService.create_human(macro_detail_dict=m)
    base.name = name
    rig = HumanService.add_builtin_rig(base, "game_engine")
    meshes = [base]
    if proxy:
        p = HumanService.add_mhclo_asset(asset("proxymeshes", proxy, "proxy"), base, asset_type="Proxymeshes",
                                         subdiv_levels=0, material_type="NONE")
        meshes = [p]
    for c in clothes:
        meshes.append(HumanService.add_mhclo_asset(asset("clothes", c, "mhclo"), base, asset_type="Clothes",
                                                   subdiv_levels=0, material_type="NONE"))
    if hair:
        sub = "clothes" if hair.startswith("fedora") else "hair"
        meshes.append(HumanService.add_mhclo_asset(asset(sub, hair, "mhclo"), base, asset_type=sub.capitalize(),
                                                   subdiv_levels=0, material_type="NONE"))
    if proxy:
        bpy.data.objects.remove(base)
    else:
        # Apply the masks (helpers, body under clothes) so every variant
        # shares one exported topology.
        bpy.context.view_layer.objects.active = base
        if base.data.shape_keys:
            base.shape_key_add(name="baked", from_mix=True)
            for kb in list(base.data.shape_keys.key_blocks):
                if kb.name != "baked":
                    base.shape_key_remove(kb)
            base.shape_key_remove(base.data.shape_keys.key_blocks["baked"])
        for mod in list(base.modifiers):
            if mod.type == 'MASK':
                bpy.ops.object.modifier_apply(modifier=mod.name)
    for o in meshes:
        for mod in list(o.modifiers):
            if mod.type in ('SUBSURF',):
                o.modifiers.remove(mod)
        o.data.materials.clear()
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    nverts = sum(len(o.data.vertices) for o in meshes)
    export(os.path.join(d, "ref.glb"), rig, meshes)

    # Synthetic faults on the artist weights: ground truth is known.
    copies = duplicate(meshes, rig)
    faults = inject_faults(copies, rig, seed=sum(map(ord, name)))
    export(os.path.join(d, "broken.glb"), rig, copies)
    remove(copies)

    # Scan/AI-style soup: same shape, triangulated, cracked into shells,
    # jittered. messy_ref carries the artist weights through the cracks.
    copies = duplicate(meshes, rig)
    height = max(v.co.z for o in meshes for v in o.data.vertices)
    for k, o in enumerate(copies):
        make_messy(o, height, seed=k + sum(map(ord, name)))
    export(os.path.join(d, "messy_ref.glb"), rig, copies)
    messy_s, messy_fail = heat_bind(rig, copies)
    export(os.path.join(d, "messy_heat.glb"), rig, copies)
    messy_verts = sum(len(o.data.vertices) for o in copies)
    remove(copies)

    # Free alternative: Blender automatic weights (bone heat) on the same meshes.
    heat_s, heat_fail = heat_bind(rig, meshes)
    export(os.path.join(d, "heat.glb"), rig, meshes)
    meta = {"name": name, "macro": m, "proxy": proxy, "clothes": clothes, "hair": hair,
            "meshes": [o.name for o in meshes], "verts": nverts, "heat_seconds": round(heat_s, 2),
            "heat_failed_meshes": heat_fail, "faults": faults, "messy_verts": messy_verts,
            "messy_heat_seconds": round(messy_s, 2), "messy_heat_failed_meshes": messy_fail, "rig": "mpfb game_engine",
            "licence": "CC0 (MPFB2 / MakeHuman system assets)"}
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"BUILT {name}: {nverts} verts, heat {heat_s:.1f}s, heat failed on {heat_fail}")


def heat_bind(rig, meshes):
    """Blender automatic weights, one mesh at a time (as a user would)."""
    fails = []
    bone_names = {b.name for b in rig.data.bones}
    t0 = time.time()
    for o in meshes:
        for vg in list(o.vertex_groups):
            if vg.name in bone_names:
                o.vertex_groups.remove(vg)
        for mod in list(o.modifiers):
            if mod.type == 'ARMATURE':
                o.modifiers.remove(mod)
        bpy.ops.object.select_all(action='DESELECT')
        o.parent = None
        o.select_set(True)
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        # Heat failure ("failed to find solution") leaves verts with no bone
        # weight; a user falls back to envelope-style weights for those, so
        # they get their nearest bone (distance to the bone segment).
        idx = {vg.index for vg in o.vertex_groups if vg.name in bone_names}
        empty = [v.index for v in o.data.vertices if not any(g.group in idx and g.weight > 0 for g in v.groups)]
        if empty:
            fails.append({"mesh": o.name, "unweighted_verts": len(empty), "fallback": "nearest bone"})
            nearest_bone(o, rig, empty)
    return time.time() - t0, fails


def nearest_bone(o, rig, verts):
    from mathutils.geometry import intersect_point_line
    segs = [(b.name, rig.matrix_world @ b.head_local, rig.matrix_world @ b.tail_local)
            for b in rig.data.bones if b.use_deform and b.name != "Root"]
    mw = o.matrix_world
    for i in verts:
        p = mw @ o.data.vertices[i].co
        best = None
        for name, h, t in segs:
            q, f = intersect_point_line(p, h, t)
            q = h if f < 0 else t if f > 1 else q
            d = (p - q).length
            if best is None or d < best[0]:
                best = (d, name)
        g = o.vertex_groups.get(best[1]) or o.vertex_groups.new(name=best[1])
        g.add([i], 1.0, 'REPLACE')


def duplicate(meshes, rig):
    out = []
    for o in meshes:
        c = o.copy()
        c.data = o.data.copy()
        bpy.context.scene.collection.objects.link(c)
        out.append(c)
    return out


def remove(objs):
    for o in objs:
        me = o.data
        bpy.data.objects.remove(o)
        bpy.data.meshes.remove(me)


def inject_faults(meshes, rig, seed):
    """Classic auto-rig faults, the ones weightforge's P0 gate names: speckle
    noise, a hand vertex patch following the thigh, a hard (binary) elbow,
    and a hair/hat piece on the wrong bone."""
    import random
    rnd = random.Random(seed)
    bones = [b.name for b in rig.data.bones if b.name != "Root"]
    log = []

    def grp(o, n):
        return o.vertex_groups.get(n) or o.vertex_groups.new(name=n)

    def clear(o, v):
        for g in list(v.groups):
            if o.vertex_groups[g.group].name in bones or o.vertex_groups[g.group].name == "Root":
                o.vertex_groups[g.group].remove([v.index])

    body = meshes[0]
    names = {g.index: g.name for g in body.vertex_groups}

    def top(v, o=body):
        gs = [(g.weight, o.vertex_groups[g.group].name) for g in v.groups if o.vertex_groups[g.group].name in bones]
        return max(gs)[1] if gs else None

    # 1. speckle: 2% of body verts get 0.6 weight on a random far bone
    vs = [v for v in body.data.vertices if top(v)]
    n = 0
    for v in rnd.sample(vs, max(1, len(vs) // 50)):
        b = rnd.choice(bones)
        for g in v.groups:
            g.weight *= 0.4
        grp(body, b).add([v.index], 0.6, 'ADD')
        n += 1
    log.append({"fault": "speckle", "verts": n})
    # 2. bleed: left hand verts take 0.5 of thigh_l
    hand = [v.index for v in body.data.vertices if top(v) in ("hand_l",)]
    for v in hand:
        for g in body.data.vertices[v].groups:
            g.weight *= 0.5
    grp(body, "thigh_l").add(hand, 0.5, 'ADD')
    log.append({"fault": "hand_l->thigh_l bleed", "verts": len(hand)})
    # 3. binary elbow: right arm verts snap to their top bone
    arm = {"upperarm_r", "lowerarm_r"}
    snapped = 0
    for o in meshes:
        for v in o.data.vertices:
            t = top(v, o)
            if t in arm:
                clear(o, v)
                grp(o, t).add([v.index], 1.0, 'REPLACE')
                snapped += 1
    log.append({"fault": "binary right elbow", "verts": snapped})
    # 4. pieces: hair/hat follows spine_03, not the head
    moved = 0
    for o in meshes[1:]:
        heads = [v for v in o.data.vertices if top(v, o) in ("head", "neck_01")]
        if len(heads) > 0.6 * len(o.data.vertices):
            for v in o.data.vertices:
                clear(o, v)
                grp(o, "spine_03").add([v.index], 1.0, 'REPLACE')
                moved += 1
    log.append({"fault": "head piece on spine_03", "verts": moved})
    for o in meshes:
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.select_all(action='DESELECT')
        o.select_set(True)
        bpy.ops.object.vertex_group_normalize_all(lock_active=False)
    return log


def make_messy(o, height, seed):
    import bmesh
    import random
    rnd = random.Random(seed)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.faces.ensure_lookup_table()
    # Grow ~150-face islands from random seeds, then crack along island borders.
    island = {}
    order = list(bm.faces)
    rnd.shuffle(order)
    for f0 in order:
        if f0 in island:
            continue
        iid = len(island) and max(island.values()) + 1
        front, size = [f0], 0
        island[f0] = iid
        while front and size < 150:
            f = front.pop(0)
            size += 1
            for e in f.edges:
                for g in e.link_faces:
                    if g not in island:
                        island[g] = iid
                        front.append(g)
    crack = [e for e in bm.edges if len(e.link_faces) == 2 and island[e.link_faces[0]] != island[e.link_faces[1]]
             and rnd.random() < 0.5]
    bmesh.ops.split_edges(bm, edges=crack)
    amp = 0.0015 * height
    for v in bm.verts:
        v.co.x += rnd.uniform(-amp, amp)
        v.co.y += rnd.uniform(-amp, amp)
        v.co.z += rnd.uniform(-amp, amp)
    bm.to_mesh(o.data)
    bm.free()


def export(path, rig, meshes):
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True)
    for o in meshes:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=path, use_selection=True, export_apply=True, export_animations=False,
                              export_materials='NONE', export_skins=True, export_all_influences=False,
                              export_def_bones=False, export_yup=True)


ensure_pack()
for row in CHARACTERS:
    if ONLY and row[0] not in ONLY:
        continue
    try:
        build(*row)
    except Exception as e:  # keep the rest of the corpus going, report at the end
        import traceback
        traceback.print_exc()
        print(f"FAILED {row[0]}: {e}")
