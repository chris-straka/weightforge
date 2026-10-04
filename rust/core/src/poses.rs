//! Range-of-motion pose sets.
//!
//! Moves are geometric, not per-rig Euler angles: "bend the forearm toward
//! forward by 140°" rotates the bone's rest direction toward the character's
//! forward axis, whatever the rig's local axis convention is. Twist turns a
//! bone about its own direction. Bones are found by anatomical role
//! (`forearm.L`, `spine*`...) resolved from common naming schemes (Rigify
//! DEF-, Mixamo, Unreal, plain), or by exact bone name.

use crate::math::{Quat, Vec3, v3};
use crate::scene::{Model, Skeleton};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum Dir {
    Up,
    Down,
    Forward,
    Back,
    /// Character's left (+X in glTF's +Z-forward convention).
    Left,
    Right,
    /// Away from the body midline (left for `.L` bones, right for `.R`).
    Out,
    In,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Motion {
    Bend(Dir, f64),
    Twist(f64),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Move {
    /// Role (`forearm.L`, `spine*`, `fingers.R`) or exact bone name.
    pub bone: String,
    pub motion: Motion,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoseSpec {
    pub name: String,
    #[serde(default)]
    pub mirror: bool,
    pub moves: Vec<Move>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    pub up: (f64, f64, f64),
    pub forward: (f64, f64, f64),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoseSet {
    pub name: String,
    #[serde(default)]
    pub frame: Option<Frame>,
    pub poses: Vec<PoseSpec>,
}

/// A resolved pose: world-space rotations at joints (axes in the rest
/// frame; children inherit their parent's motion), or local TRS overrides
/// sampled from an animation clip.
#[derive(Clone, Debug)]
pub struct Pose {
    pub name: String,
    pub moves: Vec<(usize, Vec3, f64)>,
    pub locals: Vec<(usize, Option<Vec3>, Option<Quat>, Option<Vec3>)>,
}

pub const HUMANOID_RON: &str = include_str!("../../../poses/humanoid.ron");

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SkelClass {
    Humanoid,
    Custom,
}

impl SkelClass {
    pub fn as_str(self) -> &'static str {
        match self {
            SkelClass::Humanoid => "humanoid",
            SkelClass::Custom => "custom",
        }
    }
}

pub fn parse_pose_set(text: &str) -> Result<PoseSet, String> {
    ron::from_str(text).map_err(|e| format!("pose set: {e}"))
}

// ---------------------------------------------------------------- roles

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    L,
    R,
    C,
}

fn strip_prefixes(s: &str) -> String {
    let mut s = s.to_lowercase();
    for p in ["def-", "def_", "mixamorig:", "mixamorig_", "mixamorig1:", "bip01 ", "bip001 ", "bip01_", "bip001_", "j_bip_", "j_"] {
        if let Some(rest) = s.strip_prefix(p) {
            s = rest.to_string();
        }
    }
    if let Some(i) = s.rfind('|') {
        s = s[i + 1..].to_string();
    }
    s
}

fn side_of(n: &str) -> (Side, String) {
    let toks: Vec<&str> = n.split(|c: char| c == '.' || c == '_' || c == '-' || c == ' ' || c == ':').filter(|t| !t.is_empty()).collect();
    let mut side = Side::C;
    let mut rest = Vec::new();
    for t in &toks {
        match *t {
            "l" | "left" | "lft" => side = Side::L,
            "r" | "right" | "rgt" => side = Side::R,
            _ => rest.push(*t),
        }
    }
    let mut body = rest.join("_");
    if side == Side::C {
        for (pre, sd) in [("left", Side::L), ("right", Side::R)] {
            if let Some(i) = body.find(pre) {
                side = sd;
                body = format!("{}{}", &body[..i], &body[i + pre.len()..]);
                break;
            }
        }
    }
    (side, body)
}

fn part_of(body: &str) -> Option<&'static str> {
    let b = body.replace('_', "");
    let has = |w: &str| b.contains(w);
    if has("end") && (has("head") || has("top")) || has("nub") || has("twistend") {
        return None;
    }
    if has("thumb") || has("index") || has("middle") || has("ring") || has("pinky") || has("little") || has("finger") {
        return Some("fingers");
    }
    if has("forearm") || has("lowerarm") || has("elbow") {
        return Some("forearm");
    }
    if has("upperarm") || (has("arm") && !has("armature")) {
        return Some("upper_arm");
    }
    if has("hand") || has("wrist") {
        return Some("hand");
    }
    if has("shoulder") || has("clavicle") || has("collar") {
        return Some("shoulder");
    }
    if has("thigh") || has("upleg") || has("upperleg") {
        return Some("thigh");
    }
    if has("toe") {
        return Some("toe");
    }
    if has("foot") || has("ankle") {
        return Some("foot");
    }
    if has("shin") || has("calf") || has("lowerleg") || has("knee") || has("leg") {
        return Some("shin");
    }
    if has("neck") {
        return Some("neck");
    }
    if has("head") {
        return Some("head");
    }
    if has("spine") || has("chest") || has("torso") || has("abdomen") || has("hips") || has("pelvis") {
        return Some("spine");
    }
    None
}

fn depth(sk: &Skeleton, j: usize) -> usize {
    let mut d = 0;
    let mut p = sk.jparent[j];
    while let Some(pj) = p {
        d += 1;
        p = sk.jparent[pj];
        if d > sk.joints.len() {
            break;
        }
    }
    d
}

/// Role -> joints (shallowest first). Multi-bone roles: `spine*` (the
/// chain above the hips), `fingers.L/R`.
pub fn resolve_roles(sk: &Skeleton) -> BTreeMap<String, Vec<usize>> {
    let mut roles: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for j in 0..sk.joints.len() {
        let (side, body) = side_of(&strip_prefixes(&sk.names[j]));
        let Some(part) = part_of(&body) else { continue };
        let key = match side {
            Side::L => format!("{part}.L"),
            Side::R => format!("{part}.R"),
            Side::C => part.to_string(),
        };
        roles.entry(key).or_default().push(j);
    }
    for v in roles.values_mut() {
        v.sort_by_key(|&j| (depth(sk, j), j));
    }
    // The spine chain: shallowest is the hips; the rest bend together.
    if let Some(sp) = roles.get("spine").cloned() {
        if !sp.is_empty() {
            roles.insert("hips".into(), vec![sp[0]]);
            if sp.len() > 1 {
                roles.insert("spine*".into(), sp[1..].to_vec());
                roles.insert("chest".into(), vec![*sp.last().unwrap()]);
            }
        }
    }
    roles
}

pub fn classify(sk: &Skeleton) -> SkelClass {
    let r = resolve_roles(sk);
    let need = ["upper_arm.L", "upper_arm.R", "forearm.L", "forearm.R", "thigh.L", "thigh.R", "shin.L", "shin.R"];
    if need.iter().filter(|k| r.contains_key(**k)).count() >= 6 { SkelClass::Humanoid } else { SkelClass::Custom }
}

// ---------------------------------------------------------------- resolve

fn mirror_name(s: &str) -> String {
    if let Some(b) = s.strip_suffix(".L") {
        return format!("{b}.R");
    }
    if let Some(b) = s.strip_suffix(".R") {
        return format!("{b}.L");
    }
    s.to_string()
}

fn mirror_dir(d: Dir) -> Dir {
    match d {
        Dir::Left => Dir::Right,
        Dir::Right => Dir::Left,
        o => o,
    }
}

struct Basis {
    up: Vec3,
    fwd: Vec3,
    left: Vec3,
}

fn dir_vec(b: &Basis, d: Dir, side: Side) -> Vec3 {
    match d {
        Dir::Up => b.up,
        Dir::Down => -b.up,
        Dir::Forward => b.fwd,
        Dir::Back => -b.fwd,
        Dir::Left => b.left,
        Dir::Right => -b.left,
        Dir::Out => {
            if side == Side::R {
                -b.left
            } else {
                b.left
            }
        }
        Dir::In => {
            if side == Side::R {
                b.left
            } else {
                -b.left
            }
        }
    }
}

fn bone_dir(sk: &Skeleton, j: usize) -> Vec3 {
    (sk.tail[j] - sk.head[j]).normalized()
}

/// Expands a pose set against a skeleton. Moves on missing bones are
/// skipped; poses left with no moves are dropped.
pub fn resolve(set: &PoseSet, sk: &Skeleton) -> Vec<Pose> {
    let roles = resolve_roles(sk);
    let (up, fwd) = match &set.frame {
        Some(f) => (v3(f.up.0, f.up.1, f.up.2).normalized(), v3(f.forward.0, f.forward.1, f.forward.2).normalized()),
        None => (v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)),
    };
    let basis = Basis { up, fwd, left: up.cross(fwd).normalized() };
    let mut out = Vec::new();
    let mut specs: Vec<PoseSpec> = Vec::new();
    for p in &set.poses {
        if p.mirror {
            specs.push(PoseSpec { name: format!("{}.L", p.name), mirror: false, moves: p.moves.clone() });
            specs.push(PoseSpec {
                name: format!("{}.R", p.name),
                mirror: false,
                moves: p
                    .moves
                    .iter()
                    .map(|m| Move {
                        bone: mirror_name(&m.bone),
                        motion: match m.motion {
                            Motion::Bend(d, a) => Motion::Bend(mirror_dir(d), a),
                            Motion::Twist(a) => Motion::Twist(-a),
                        },
                    })
                    .collect(),
            });
        } else {
            specs.push(p.clone());
        }
    }
    for p in specs {
        let mut moves = Vec::new();
        for m in &p.moves {
            let (targets, side) = match roles.get(&m.bone) {
                Some(js) => {
                    let side = if m.bone.ends_with(".R") {
                        Side::R
                    } else if m.bone.ends_with(".L") {
                        Side::L
                    } else {
                        Side::C
                    };
                    let js = if m.bone.ends_with('*') || m.bone.starts_with("fingers") { js.clone() } else { vec![js[0]] };
                    (js, side)
                }
                None => match sk.joint_by_name(&m.bone) {
                    Some(j) => (vec![j], side_of(&strip_prefixes(&m.bone)).0),
                    None => continue,
                },
            };
            // Multi-bone roles share the angle (a spine bend spreads over
            // the chain); fingers each curl by the full angle.
            let share = if m.bone.ends_with('*') { targets.len() as f64 } else { 1.0 };
            for j in targets {
                let d = bone_dir(sk, j);
                if d.len2() == 0.0 {
                    continue;
                }
                match m.motion {
                    Motion::Twist(a) => moves.push((j, d, (a / share).to_radians())),
                    Motion::Bend(dir, a) => {
                        let t = dir_vec(&basis, dir, side);
                        let axis = d.cross(t);
                        if axis.len() < 0.15 {
                            continue; // bone already points that way
                        }
                        moves.push((j, axis.normalized(), (a / share).to_radians()));
                    }
                }
            }
        }
        if !moves.is_empty() {
            out.push(Pose { name: p.name, moves, locals: Vec::new() });
        }
    }
    out
}

/// Generic ROM for skeletons with no known naming: every weighted,
/// non-root bone bends ±45° about two axes and twists ±45°.
pub fn generic(model: &Model) -> Vec<Pose> {
    let sk = &model.skel;
    // A bone is worth posing if it carries weight, or if it owns geometry
    // (nearest bone to >= 1% of vertices) even when the weights ignore it
    // (a rig with everything on the root must still be exercised).
    let mut used = vec![0usize; sk.joints.len()];
    for vw in &model.weights {
        for &(j, w) in vw {
            if w > 0.05 {
                used[j as usize] += 1;
            }
        }
    }
    let mut territory = vec![0usize; sk.joints.len()];
    for p in &model.rest {
        let j = (0..sk.joints.len()).min_by(|&a, &b| sk.seg_dist(a, *p).partial_cmp(&sk.seg_dist(b, *p)).unwrap()).unwrap_or(0);
        territory[j] += 1;
    }
    let min_t = (model.nverts() / 100).max(1);
    for j in 0..sk.joints.len() {
        if territory[j] >= min_t {
            used[j] += 1;
        }
    }
    let mut out = Vec::new();
    for j in 0..sk.joints.len() {
        // A hierarchy root moves everything rigidly (nothing to learn); a
        // childless root in a flat skeleton is a real deforming bone.
        let hierarchy_root = sk.jparent[j].is_none() && !sk.jchildren[j].is_empty();
        if hierarchy_root || used[j] == 0 && sk.jchildren[j].iter().all(|&c| used[c] == 0) {
            continue;
        }
        let d = bone_dir(sk, j);
        if d.len2() == 0.0 {
            continue;
        }
        let a = d.any_perp();
        let b = d.cross(a).normalized();
        let n = &sk.names[j];
        let deg = 45f64.to_radians();
        for (tag, axis, ang) in
            [("bend_a+", a, deg), ("bend_a-", a, -deg), ("bend_b+", b, deg), ("bend_b-", b, -deg), ("twist+", d, deg), ("twist-", d, -deg)]
        {
            out.push(Pose { name: format!("{n}_{tag}"), moves: vec![(j, axis, ang)], locals: Vec::new() });
        }
    }
    out
}

/// Samples each animation clip at up to `per_clip` evenly spaced times.
pub fn clip_poses(model: &Model, per_clip: usize) -> Vec<Pose> {
    let glb = &model.glb;
    let mut out = Vec::new();
    for (ai, anim) in glb.arr("animations").iter().enumerate() {
        let name = anim.get("name").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| format!("clip{ai}"));
        let samplers = anim.get("samplers").and_then(Value::as_array).cloned().unwrap_or_default();
        let mut tracks = Vec::new(); // (node, path, times, values, ncomp, step)
        let (mut t0, mut t1) = (f64::MAX, f64::MIN);
        for ch in anim.get("channels").and_then(Value::as_array).into_iter().flatten() {
            let (Some(node), Some(path), Some(si)) = (
                ch.get("target").and_then(|t| t.get("node")).and_then(Value::as_u64),
                ch.get("target").and_then(|t| t.get("path")).and_then(Value::as_str),
                ch.get("sampler").and_then(Value::as_u64),
            ) else {
                continue;
            };
            if !matches!(path, "translation" | "rotation" | "scale") {
                continue;
            }
            let Some(s) = samplers.get(si as usize) else { continue };
            let (Some(ia), Some(oa)) = (s.get("input").and_then(Value::as_u64), s.get("output").and_then(Value::as_u64)) else { continue };
            let Ok((_, times)) = glb.read_f64(ia as usize) else { continue };
            let Ok((nc, vals)) = glb.read_f64(oa as usize) else { continue };
            let interp = s.get("interpolation").and_then(Value::as_str).unwrap_or("LINEAR");
            if times.is_empty() || vals.iter().any(|v| !v.is_finite()) {
                continue;
            }
            // CUBICSPLINE stores (in, value, out) triplets: take the values.
            let vals: Vec<f64> =
                if interp == "CUBICSPLINE" { vals.chunks(nc * 3).flat_map(|c| c[nc..2 * nc].to_vec()).collect() } else { vals };
            if vals.len() < times.len() * nc {
                continue;
            }
            t0 = t0.min(times[0]);
            t1 = t1.max(*times.last().unwrap());
            tracks.push((node as usize, path.to_string(), times, vals, nc, interp == "STEP"));
        }
        if tracks.is_empty() {
            continue;
        }
        let k = per_clip.max(1);
        for s in 0..k {
            let t = if k == 1 { t0 } else { t0 + (t1 - t0) * s as f64 / (k - 1) as f64 };
            let mut locals: BTreeMap<usize, (Option<Vec3>, Option<Quat>, Option<Vec3>)> = BTreeMap::new();
            for (node, path, times, vals, nc, step) in &tracks {
                let i = times.partition_point(|&x| x <= t).saturating_sub(1).min(times.len() - 1);
                let j = (i + 1).min(times.len() - 1);
                let f = if j == i || *step { 0.0 } else { ((t - times[i]) / (times[j] - times[i])).clamp(0.0, 1.0) };
                let a = &vals[i * nc..i * nc + nc];
                let b = &vals[j * nc..j * nc + nc];
                let e = locals.entry(*node).or_default();
                match path.as_str() {
                    "rotation" if *nc == 4 => {
                        let qa = Quat { x: a[0], y: a[1], z: a[2], w: a[3] }.normalized();
                        let qb = Quat { x: b[0], y: b[1], z: b[2], w: b[3] }.normalized();
                        e.1 = Some(qa.slerp(qb, f));
                    }
                    "translation" if *nc == 3 => e.0 = Some(v3(a[0], a[1], a[2]).lerp(v3(b[0], b[1], b[2]), f)),
                    "scale" if *nc == 3 => e.2 = Some(v3(a[0], a[1], a[2]).lerp(v3(b[0], b[1], b[2]), f)),
                    _ => {}
                }
            }
            out.push(Pose {
                name: format!("{name}@{t:.2}s"),
                moves: Vec::new(),
                locals: locals.into_iter().map(|(n, (t, r, s))| (n, t, r, s)).collect(),
            });
        }
    }
    out
}

/// The default pose set for a model: the class preset (or a user file),
/// plus samples of the asset's own clips.
pub fn default_poses(model: &Model, user: Option<&PoseSet>, clips: bool) -> (SkelClass, Vec<Pose>) {
    let class = classify(&model.skel);
    let mut poses = match (user, class) {
        (Some(set), _) => resolve(set, &model.skel),
        (None, SkelClass::Humanoid) => resolve(&parse_pose_set(HUMANOID_RON).expect("built-in humanoid.ron"), &model.skel),
        (None, SkelClass::Custom) => generic(model),
    };
    if clips {
        poses.extend(clip_poses(model, 6));
    }
    (class, poses)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn role_names() {
        let cases = [
            ("DEF-forearm.L", "forearm.L"),
            ("DEF-upper_arm.R", "upper_arm.R"),
            ("mixamorig:LeftForeArm", "forearm.L"),
            ("mixamorig:RightUpLeg", "thigh.R"),
            ("mixamorig:LeftLeg", "shin.L"),
            ("calf_l", "shin.L"),
            ("lowerarm_r", "forearm.R"),
            ("L_Upperarm", "upper_arm.L"),
            ("DEF-thigh.L", "thigh.L"),
            ("DEF-shin.R", "shin.R"),
            ("mixamorig:LeftHandIndex1", "fingers.L"),
            ("DEF-spine.003", "spine"),
            ("Head", "head"),
        ];
        for (n, want) in cases {
            let (side, body) = side_of(&strip_prefixes(n));
            let part = part_of(&body).unwrap_or("?");
            let got = match side {
                Side::L => format!("{part}.L"),
                Side::R => format!("{part}.R"),
                Side::C => part.to_string(),
            };
            assert_eq!(got, want, "{n}");
        }
    }
    #[test]
    fn humanoid_ron_parses() {
        let s = parse_pose_set(HUMANOID_RON).unwrap();
        assert!(s.poses.len() >= 10);
    }
}
