# SPDX-License-Identifier: GPL-3.0-or-later
# WeightForge Blender extension: skin-weight QA and repair via the `weights` CLI.
#
# Flow: the armature and every mesh bound to it are exported to a temp GLB
# (deform bones only, no modifiers applied, so vertex positions are the
# mesh's own), the CLI checks or fixes it, and results come back by exact
# vertex position: a `weightforge_bad` color attribute for check, vertex
# group weights for fix (one undo step). weightforge never moves vertices
# or renames bones.
#
# The engine stays MIT-licensed: this GPL extension talks to it only as a
# subprocess over files, never linked, never imported.

import json
import os
import shutil
import subprocess
import tempfile

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, PointerProperty, StringProperty

# Must equal the module name Blender loaded us under ("weightforge" on the
# sys.path test path, "bl_ext.<repo>.weightforge" as an installed extension).
ADDON_ID = __package__

METHODS = [
    ("auto", "Auto", "Try every method, keep the best per broken region"),
    ("smooth", "Clean-up", "Bleed pruning, despeckle, and smoothing of torn areas"),
    ("geodesic", "Geodesic", "Geodesic voxel binding and joint bands"),
    ("transfer", "Transfer", "Pieces from their attachment chain; whole body from a source rig"),
    ("optimize", "Optimize", "Fit linear blend skinning to dual-quaternion targets"),
]


def find_cli(explicit_path=""):
    """Locate the `weights` CLI: preference, PATH, then the repo build tree."""
    if explicit_path:
        p = bpy.path.abspath(explicit_path)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    on_path = shutil.which("weights")
    if on_path:
        return on_path
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in (("..", "..", "rust", "target", "release", "weights"), ("..", "..", "rust", "target", "debug", "weights")):
        p = os.path.normpath(os.path.join(here, *rel))
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return ""


def prefs(context):
    addon = context.preferences.addons.get(ADDON_ID)
    return addon.preferences if addon else None


def rig_of(context):
    """(armature, meshes) for the active object: an armature, or a mesh
    deformed by one. Every mesh bound to the armature comes along so
    pieces (capes, hair) are checked with their body."""
    obj = context.active_object
    arm = None
    if obj and obj.type == "ARMATURE":
        arm = obj
    elif obj and obj.type == "MESH":
        for m in obj.modifiers:
            if m.type == "ARMATURE" and m.object:
                arm = m.object
                break
        if arm is None and obj.parent and obj.parent.type == "ARMATURE":
            arm = obj.parent
    if arm is None:
        return None, []
    meshes = []
    for o in context.scene.objects:
        if o.type != "MESH":
            continue
        bound = any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)
        if bound or (o.parent == arm and o.parent_type == "ARMATURE"):
            meshes.append(o)
    return arm, meshes


def export_glb(context, path, arm, meshes, clips):
    """Exports exactly the rig to a GLB without touching the user's selection."""
    view_layer = context.view_layer
    prev_sel = [o for o in context.scene.objects if o.select_get()]
    prev_active = view_layer.objects.active
    prev_mode = context.mode
    if prev_mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    try:
        for o in prev_sel:
            o.select_set(False)
        for o in [arm] + meshes:
            o.select_set(True)
        view_layer.objects.active = arm
        bpy.ops.export_scene.gltf(
            filepath=path,
            export_format="GLB",
            use_selection=True,
            export_skins=True,
            export_def_bones=True,
            export_animations=clips,
            export_apply=False,
            export_yup=True,
            export_materials="NONE",
        )
    finally:
        for o in context.scene.objects:
            o.select_set(False)
        for o in prev_sel:
            o.select_set(True)
        view_layer.objects.active = prev_active


def run_cli(args):
    proc = subprocess.run(args, capture_output=True, text=True)
    # Exit 1 means "faults found", which is a result, not an error.
    if proc.returncode not in (0, 1):
        raise RuntimeError((proc.stderr or proc.stdout or "weights CLI failed").strip())
    return proc


def _key(x, y, z):
    return (round(x, 5), round(y, 5), round(z, 5))


def vertex_lookup(obj):
    """Blender vertex index by the position the glTF exporter writes for it
    (mesh-local, Y up: x, z, -y), with a world-space fallback map."""
    local, world = {}, {}
    mw = obj.matrix_world
    for v in obj.data.vertices:
        c = v.co
        local.setdefault(_key(c.x, c.z, -c.y), []).append(v.index)
        w = mw @ c
        world.setdefault(_key(w.x, w.z, -w.y), []).append(v.index)
    return local, world


def match_mesh(entry, meshes):
    for o in meshes:
        if o.data.name == entry["mesh"] or o.name == entry["node"]:
            return o
    return None


def per_vertex(entry, obj, values):
    """Maps per-raw-vertex `values` onto obj's vertices. Returns
    {vertex index: value} and the count of unmatched Blender vertices."""
    pos = entry["positions"]
    local, world = vertex_lookup(obj)
    keys = [_key(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]) for i in range(len(pos) // 3)]
    hits_local = sum(1 for k in keys if k in local)
    hits_world = sum(1 for k in keys if k in world)
    table = local if hits_local >= hits_world else world
    out = {}
    for i, k in enumerate(keys):
        for vi in table.get(k, ()):
            out.setdefault(vi, values[i])
    return out, len(obj.data.vertices) - len(out)


class WeightForgePreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID
    cli_path: StringProperty(name="weights CLI", subtype="FILE_PATH", description="Path to the `weights` binary (blank: PATH, then the repo build)")

    def draw(self, context):
        self.layout.prop(self, "cli_path")


class WeightForgeSettings(bpy.types.PropertyGroup):
    method: EnumProperty(name="Method", items=METHODS, default="auto")
    voxels: IntProperty(name="Voxels", default=128, min=16, max=1024, description="Voxel resolution for geodesic distances")
    clips: BoolProperty(name="Use Clips", default=True, description="Also test the rig's own animation clips")
    all_regions: BoolProperty(name="All Regions", default=False, description="Let fix change regions the check did not flag")
    poses_path: StringProperty(name="Poses", subtype="FILE_PATH", description="Range-of-motion .ron (blank: built-in by skeleton class)")
    source_path: StringProperty(name="Source Rig", subtype="FILE_PATH", description="Known-good rigged base .glb to transfer from")
    candidate_path: StringProperty(name="Candidate", subtype="FILE_PATH", description="Extra weights to score, e.g. UniRig output .glb")
    summary: StringProperty(name="Summary", default="")
    sheet_path: StringProperty(name="Sheet", default="")


def common_args(s):
    args = ["--voxels", str(s.voxels)]
    if s.poses_path:
        args += ["--poses", bpy.path.abspath(s.poses_path)]
    if not s.clips:
        args.append("--no-clips")
    return args


def load_image(path):
    img = bpy.data.images.load(path, check_existing=False)
    img.name = os.path.basename(path)
    for win in getattr(bpy.context.window_manager, "windows", []):
        for area in win.screen.areas:
            if area.type == "IMAGE_EDITOR":
                area.spaces.active.image = img
                return img
    return img


class _RigOp:
    def setup(self, context):
        self.cli = find_cli(getattr(prefs(context), "cli_path", ""))
        if not self.cli:
            self.report({"ERROR"}, "weights CLI not found (set it in the add-on preferences)")
            return None
        arm, meshes = rig_of(context)
        if arm is None or not meshes:
            self.report({"ERROR"}, "Select an armature, or a mesh deformed by one")
            return None
        self.dir = tempfile.mkdtemp(prefix="weightforge_")
        self.glb = os.path.join(self.dir, f"{bpy.path.clean_name(arm.name)}.glb")
        export_glb(context, self.glb, arm, meshes, context.scene.weightforge.clips)
        return arm, meshes


class WEIGHTFORGE_OT_check(bpy.types.Operator, _RigOp):
    bl_idname = "weightforge.check"
    bl_label = "Check Weights"
    bl_description = "Pose the rig through its range of motion and mark bad vertices (color attribute weightforge_bad)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        rig = self.setup(context)
        if rig is None:
            return {"CANCELLED"}
        arm, meshes = rig
        s = context.scene.weightforge
        vj = os.path.join(self.dir, "vertices.json")
        try:
            proc = run_cli([self.cli, "check", self.glb, "--json", "--vertex-json", vj] + common_args(s))
        except RuntimeError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        rep = json.loads(proc.stdout)
        with open(vj) as f:
            doc = json.load(f)
        painted = 0
        for entry in doc["meshes"]:
            obj = match_mesh(entry, meshes)
            if obj is None:
                continue
            emax = max(entry["energy"]) or 1.0
            vals = [(fl, en) for fl, en in zip(entry["flags"], entry["energy"])]
            mapping, _ = per_vertex(entry, obj, vals)
            attr = obj.data.color_attributes.get("weightforge_bad") or obj.data.color_attributes.new("weightforge_bad", "FLOAT_COLOR", "POINT")
            data = [0.0] * (len(obj.data.vertices) * 4)
            for vi in range(len(obj.data.vertices)):
                fl, en = mapping.get(vi, (0, 0.0))
                t = min(1.0, en / emax)
                fails = fl & ~16  # intersect is a warning (orange)
                if fails:
                    c = (1.0, 0.08, 0.05, 1.0)
                    painted += 1
                elif fl:
                    c = (1.0, 0.55, 0.1, 1.0)
                else:
                    c = (0.75 + 0.25 * t, 0.75 * (1 - t), 0.75 * (1 - t), 1.0)
                data[vi * 4:vi * 4 + 4] = c
            attr.data.foreach_set("color", data)
            obj.data.color_attributes.active_color = attr
        fails = [f for f in rep["findings"] if f["severity"] == "fail"]
        lines = [f"{'PASS' if rep['pass'] else 'FAIL'}  score {rep['score']:.1f}/100  ({rep['class']}, {rep['poses']} poses)"]
        lines += [f"{f['code']} {f['region']}: {f['verts']} verts" for f in fails[:8]]
        s.summary = "\n".join(lines)
        self.report({"INFO"}, f"{lines[0]}; {painted} bad vertices marked in weightforge_bad")
        return {"FINISHED"}


class WEIGHTFORGE_OT_sheet(bpy.types.Operator, _RigOp):
    bl_idname = "weightforge.sheet"
    bl_label = "Pose Sheet"
    bl_description = "Render the range-of-motion sheet (bad vertices in red) into an image"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if self.setup(context) is None:
            return {"CANCELLED"}
        s = context.scene.weightforge
        out = os.path.join(self.dir, "sheet.png")
        try:
            run_cli([self.cli, "sheet", self.glb, "--out", out] + common_args(s))
        except RuntimeError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        load_image(out)
        s.sheet_path = out
        self.report({"INFO"}, f"Sheet: {out}")
        return {"FINISHED"}


class WEIGHTFORGE_OT_fix(bpy.types.Operator, _RigOp):
    bl_idname = "weightforge.fix"
    bl_label = "Fix Weights"
    bl_description = "Repair bad weights (never worse than the input in any region) and write them into the vertex groups"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        rig = self.setup(context)
        if rig is None:
            return {"CANCELLED"}
        arm, meshes = rig
        s = context.scene.weightforge
        fixed = os.path.join(self.dir, "fixed.glb")
        report = os.path.join(self.dir, "fixed.report.json")
        sheet = os.path.join(self.dir, "before_after.png")
        dump = os.path.join(self.dir, "fixed.json")
        args = [self.cli, "fix", self.glb, "--out", fixed, "--report", report, "--sheet", sheet, "--method", s.method] + common_args(s)
        if s.source_path:
            args += ["--source", bpy.path.abspath(s.source_path)]
        if s.candidate_path:
            args += ["--candidate", bpy.path.abspath(s.candidate_path)]
        if s.all_regions:
            args.append("--all-regions")
        try:
            run_cli(args)
            run_cli([self.cli, "dump", fixed, "--out", dump])
        except RuntimeError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        with open(report) as f:
            rep = json.load(f)
        with open(dump) as f:
            doc = json.load(f)
        bones = {b.name for b in arm.data.bones}
        joints = doc["joints"]
        changed, unmatched = 0, 0
        for entry in doc["meshes"]:
            obj = match_mesh(entry, meshes)
            if obj is None:
                continue
            mapping, miss = per_vertex(entry, obj, entry["influences"])
            unmatched += miss
            groups = {g.name: g for g in obj.vertex_groups}
            bone_groups = [g for g in obj.vertex_groups if g.name in bones]
            for vi, infl in mapping.items():
                want = {joints[j]: w for j, w in infl}
                have = {}
                for ge in obj.data.vertices[vi].groups:
                    name = obj.vertex_groups[ge.group].name
                    if name in bones:
                        have[name] = ge.weight
                if want.keys() == have.keys() and all(abs(want[k] - have[k]) < 1e-5 for k in want):
                    continue
                changed += 1
                for g in bone_groups:
                    if g.name in have:
                        g.remove([vi])
                for name, w in want.items():
                    g = groups.get(name)
                    if g is None:
                        g = obj.vertex_groups.new(name=name)
                        groups[name] = g
                        bone_groups.append(g)
                    g.add([vi], w, "REPLACE")
        a, b = rep["before"], rep["after"]
        verdict = "FIXED" if b["pass"] else ("IMPROVED" if rep["improved"] else "UNCHANGED")
        lines = [f"{verdict}  score {a['score']:.1f} -> {b['score']:.1f}  ({changed} verts changed)"]
        lines += [f"{r['label']}: {r['before']:.0f} -> {r['after']:.0f} via {r['chosen']}" for r in rep["regions"] if r["chosen"] != "original"][:8]
        if unmatched:
            lines.append(f"{unmatched} vertices not matched (kept as they were)")
        s.summary = "\n".join(lines)
        if os.path.exists(sheet):
            load_image(sheet)
            s.sheet_path = sheet
        self.report({"INFO"}, lines[0])
        return {"FINISHED"}


class WEIGHTFORGE_PT_panel(bpy.types.Panel):
    bl_label = "WeightForge"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "WeightForge"

    def draw(self, context):
        layout = self.layout
        s = context.scene.weightforge
        arm, meshes = rig_of(context)
        if arm is None:
            layout.label(text="Select an armature or skinned mesh", icon="INFO")
        else:
            layout.label(text=f"{arm.name}: {len(meshes)} mesh(es)", icon="ARMATURE_DATA")
        col = layout.column(align=True)
        col.operator("weightforge.check", icon="VIEWZOOM")
        col.operator("weightforge.sheet", icon="IMAGE_DATA")
        col.operator("weightforge.fix", icon="MODIFIER")
        box = layout.box()
        box.prop(s, "method")
        box.prop(s, "source_path")
        box.prop(s, "candidate_path")
        row = box.row()
        row.prop(s, "all_regions")
        row.prop(s, "clips")
        box.prop(s, "poses_path")
        box.prop(s, "voxels")
        if s.summary:
            res = layout.box()
            for line in s.summary.split("\n"):
                res.label(text=line)


classes = (WeightForgePreferences, WeightForgeSettings, WEIGHTFORGE_OT_check, WEIGHTFORGE_OT_sheet, WEIGHTFORGE_OT_fix, WEIGHTFORGE_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.weightforge = PointerProperty(type=WeightForgeSettings)


def unregister():
    del bpy.types.Scene.weightforge
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
