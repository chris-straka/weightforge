//! Procedural test mannequin with known-good ("ground truth") weights and
//! injected faults. Multi-shell like AI meshes: torso, head, arms, and legs
//! are separate overlapping tubes, plus a `Cape` piece mesh. Rigify-style
//! `DEF-*` bone names, A-pose, +Y up, +Z forward, bone +Y along the bone.

use crate::glb::{Glb, f32_bytes};
use crate::math::{Mat4, Quat, Rng, Vec3, quat_from_basis, v3};
use crate::scene::{VW, Weights, normalize_vw};
use serde_json::json;

pub const FAULTS: &[&str] = &["clean", "noise", "bleed", "no_elbow_falloff", "cape_wrong_bone", "elbow_collapse", "knee_wide_falloff"];

pub struct Bone {
    pub name: &'static str,
    pub parent: Option<usize>,
    pub head: Vec3,
    pub tail: Vec3,
}

/// One chain sample per vertex: which chain and the arc length along it.
#[derive(Clone, Copy)]
struct Tag {
    chain: usize,
    s: f64,
}

struct Chain {
    /// (bone, start s) in order; each bone runs until the next start.
    bones: Vec<(usize, f64)>,
    /// Blend half-width per boundary (len = bones.len() - 1).
    bands: Vec<f64>,
}

pub struct Fixture {
    pub bones: Vec<Bone>,
    pub body_pos: Vec<Vec3>,
    pub body_tris: Vec<[u32; 3]>,
    pub cape_pos: Vec<Vec3>,
    pub cape_tris: Vec<[u32; 3]>,
    /// Ground-truth weights per body vertex then per cape vertex.
    pub body_w: Weights,
    pub cape_w: Weights,
}

fn arm_dir(side: f64) -> Vec3 {
    let a = 70f64.to_radians();
    v3(side * a.cos(), -a.sin(), 0.0)
}

pub fn skeleton() -> Vec<Bone> {
    let mut b = vec![
        Bone { name: "DEF-spine", parent: None, head: v3(0.0, 0.95, 0.0), tail: v3(0.0, 1.05, 0.0) },
        Bone { name: "DEF-spine.001", parent: Some(0), head: v3(0.0, 1.05, 0.0), tail: v3(0.0, 1.18, 0.0) },
        Bone { name: "DEF-spine.002", parent: Some(1), head: v3(0.0, 1.18, 0.0), tail: v3(0.0, 1.32, 0.0) },
        Bone { name: "DEF-spine.003", parent: Some(2), head: v3(0.0, 1.32, 0.0), tail: v3(0.0, 1.45, 0.0) },
        Bone { name: "DEF-neck", parent: Some(3), head: v3(0.0, 1.45, 0.0), tail: v3(0.0, 1.55, 0.0) },
        Bone { name: "DEF-head", parent: Some(4), head: v3(0.0, 1.55, 0.0), tail: v3(0.0, 1.78, 0.0) },
    ];
    for (side, sfx) in [(1.0, "L"), (-1.0, "R")] {
        let names: [&'static str; 7] = if sfx == "L" {
            ["DEF-shoulder.L", "DEF-upper_arm.L", "DEF-forearm.L", "DEF-hand.L", "DEF-thigh.L", "DEF-shin.L", "DEF-foot.L"]
        } else {
            ["DEF-shoulder.R", "DEF-upper_arm.R", "DEF-forearm.R", "DEF-hand.R", "DEF-thigh.R", "DEF-shin.R", "DEF-foot.R"]
        };
        let d = arm_dir(side);
        let sh0 = v3(side * 0.03, 1.42, 0.0);
        let sh1 = v3(side * 0.17, 1.42, 0.0);
        let el = sh1 + d * 0.30;
        let wr = el + d * 0.27;
        let tip = wr + d * 0.14;
        let i = b.len();
        b.push(Bone { name: names[0], parent: Some(3), head: sh0, tail: sh1 });
        b.push(Bone { name: names[1], parent: Some(i), head: sh1, tail: el });
        b.push(Bone { name: names[2], parent: Some(i + 1), head: el, tail: wr });
        b.push(Bone { name: names[3], parent: Some(i + 2), head: wr, tail: tip });
        let hip = v3(side * 0.1, 0.92, 0.0);
        let knee = v3(side * 0.1, 0.5, 0.0);
        let ankle = v3(side * 0.1, 0.09, 0.0);
        let toe = v3(side * 0.1, 0.03, 0.15);
        b.push(Bone { name: names[4], parent: Some(0), head: hip, tail: knee });
        b.push(Bone { name: names[5], parent: Some(i + 4), head: knee, tail: ankle });
        b.push(Bone { name: names[6], parent: Some(i + 5), head: ankle, tail: toe });
    }
    b
}

fn idx(bones: &[Bone], n: &str) -> usize {
    bones.iter().position(|b| b.name == n).unwrap()
}

/// Tube along a polyline with parallel-transport frames, capped ends.
/// `radius(s)` returns (rx, ry) along the two frame axes.
fn tube(
    pos: &mut Vec<Vec3>,
    tris: &mut Vec<[u32; 3]>,
    tags: &mut Vec<Tag>,
    chain: usize,
    path: &[Vec3],
    up0: Vec3,
    ds: f64,
    nseg: usize,
    radius: &dyn Fn(f64) -> (f64, f64),
) {
    // Resample the path at ~ds spacing.
    let mut samples: Vec<(Vec3, f64, Vec3)> = Vec::new(); // point, s, tangent
    let mut s0 = 0.0;
    for k in 0..path.len() - 1 {
        let (a, b) = (path[k], path[k + 1]);
        let len = (b - a).len();
        let n = ((len / ds).ceil() as usize).max(1);
        let t = (b - a).normalized();
        for i in 0..n {
            let f = i as f64 / n as f64;
            samples.push((a.lerp(b, f), s0 + len * f, t));
        }
        s0 += len;
    }
    samples.push((*path.last().unwrap(), s0, (path[path.len() - 1] - path[path.len() - 2]).normalized()));
    // Smooth tangents at corners.
    let m = samples.len();
    let tangents: Vec<Vec3> = (0..m)
        .map(|i| {
            let a = samples[i.saturating_sub(1)].0;
            let b = samples[(i + 1).min(m - 1)].0;
            (b - a).normalized()
        })
        .collect();
    let mut normal = (up0 - tangents[0] * up0.dot(tangents[0])).normalized();
    let base = pos.len() as u32;
    for i in 0..m {
        if i > 0 {
            let q = Quat::from_to(tangents[i - 1], tangents[i]);
            normal = q.rotate(normal);
            normal = (normal - tangents[i] * normal.dot(tangents[i])).normalized();
        }
        let bin = tangents[i].cross(normal);
        let (rx, ry) = radius(samples[i].1);
        for k in 0..nseg {
            let a = std::f64::consts::TAU * k as f64 / nseg as f64;
            pos.push(samples[i].0 + normal * (a.cos() * rx) + bin * (a.sin() * ry));
            tags.push(Tag { chain, s: samples[i].1 });
        }
    }
    for i in 0..m - 1 {
        for k in 0..nseg {
            let a = base + (i * nseg + k) as u32;
            let b = base + (i * nseg + (k + 1) % nseg) as u32;
            let c = a + nseg as u32;
            let d = b + nseg as u32;
            tris.push([a, b, d]);
            tris.push([a, d, c]);
        }
    }
    for (ring, s, flip) in [(0usize, 0.0, true), (m - 1, s0, false)] {
        let center = pos.len() as u32;
        pos.push(samples[ring].0);
        tags.push(Tag { chain, s });
        for k in 0..nseg {
            let a = base + (ring * nseg + k) as u32;
            let b = base + (ring * nseg + (k + 1) % nseg) as u32;
            tris.push(if flip { [center, b, a] } else { [center, a, b] });
        }
    }
}

fn smoothstep(x: f64) -> f64 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn chain_weights(c: &Chain, s: f64) -> VW {
    let n = c.bones.len();
    let mut w = vec![1.0; n];
    for k in 1..n {
        let b = c.bones[k].1;
        let bw = c.bands[k - 1];
        let t = if bw <= 0.0 { if s >= b { 1.0 } else { 0.0 } } else { smoothstep((s - (b - bw)) / (2.0 * bw)) };
        for (i, wi) in w.iter_mut().enumerate() {
            *wi *= if i < k { 1.0 - t } else { t };
        }
    }
    normalize_vw(c.bones.iter().zip(w).map(|(&(j, _), w)| (j as u32, w)).collect())
}

pub fn build(fault: &str) -> Fixture {
    let bones = skeleton();
    let mut pos = Vec::new();
    let mut tris = Vec::new();
    let mut tags = Vec::new();
    let mut chains: Vec<Chain> = Vec::new();
    let j = |n: &str| idx(&bones, n);

    // Torso: y 0.85 -> 1.5, elliptical.
    chains.push(Chain {
        bones: vec![(j("DEF-spine"), 0.0), (j("DEF-spine.001"), 0.20), (j("DEF-spine.002"), 0.33), (j("DEF-spine.003"), 0.47)],
        bands: vec![0.04, 0.04, 0.04],
    });
    tube(&mut pos, &mut tris, &mut tags, 0, &[v3(0.0, 0.85, 0.0), v3(0.0, 1.5, 0.0)], v3(1.0, 0.0, 0.0), 0.025, 24, &|s| {
        let y = 0.85 + s;
        let w = 0.15 + 0.02 * ((y - 1.1) * 4.0).sin();
        (w, 0.1)
    });
    // Neck + head.
    chains.push(Chain { bones: vec![(j("DEF-neck"), 0.0), (j("DEF-head"), 0.12)], bands: vec![0.03] });
    tube(&mut pos, &mut tris, &mut tags, 1, &[v3(0.0, 1.43, 0.0), v3(0.0, 1.8, 0.0)], v3(1.0, 0.0, 0.0), 0.02, 20, &|s| {
        let y = 1.43 + s;
        let r = if y < 1.56 { 0.05 } else { 0.05 + 0.055 * (((y - 1.56) / 0.24) * std::f64::consts::PI).sin().max(0.0).sqrt() };
        (r.max(0.02), r.max(0.02))
    });
    // Arms and legs.
    for side in [1.0, -1.0] {
        let sfx = if side > 0.0 { "L" } else { "R" };
        let d = arm_dir(side);
        let sh1 = v3(side * 0.17, 1.42, 0.0);
        let el = sh1 + d * 0.30;
        let wr = el + d * 0.27;
        let tip = wr + d * 0.14;
        let start = v3(side * 0.11, 1.42, 0.0);
        let s_sh = (sh1 - start).len();
        let ci = chains.len();
        chains.push(Chain {
            bones: vec![
                (j(&format!("DEF-shoulder.{sfx}")), 0.0),
                (j(&format!("DEF-upper_arm.{sfx}")), s_sh),
                (j(&format!("DEF-forearm.{sfx}")), s_sh + 0.30),
                (j(&format!("DEF-hand.{sfx}")), s_sh + 0.57),
            ],
            bands: vec![0.07, 0.05, 0.035],
        });
        tube(&mut pos, &mut tris, &mut tags, ci, &[start, sh1, el, wr, tip], v3(0.0, 0.0, 1.0), 0.015, 16, &|s| {
            let u = s - s_sh;
            let r = if u < 0.0 {
                0.055
            } else if u < 0.57 {
                0.055 - 0.02 * (u / 0.57)
            } else {
                0.035 + 0.01 * ((u - 0.57) / 0.14 * 3.0).min(1.0)
            };
            let flat = if u > 0.57 { 0.6 } else { 1.0 };
            (r, r * flat)
        });
        let hip_top = v3(side * 0.1, 0.97, 0.0);
        let hip = v3(side * 0.1, 0.92, 0.0);
        let knee = v3(side * 0.1, 0.5, 0.0);
        let ankle = v3(side * 0.1, 0.09, 0.0);
        let toe = v3(side * 0.1, 0.04, 0.17);
        let ci = chains.len();
        chains.push(Chain {
            bones: vec![
                (j("DEF-spine"), 0.0),
                (j(&format!("DEF-thigh.{sfx}")), 0.05),
                (j(&format!("DEF-shin.{sfx}")), 0.47),
                (j(&format!("DEF-foot.{sfx}")), 0.88),
            ],
            bands: vec![0.07, 0.06, 0.04],
        });
        tube(&mut pos, &mut tris, &mut tags, ci, &[hip_top, hip, knee, ankle, toe], v3(0.0, 0.0, 1.0), 0.02, 16, &|s| {
            let r = if s < 0.47 {
                0.085 - 0.02 * (s / 0.47)
            } else if s < 0.88 {
                0.065 - 0.02 * ((s - 0.47) / 0.41)
            } else {
                0.045
            };
            (r, r)
        });
    }
    let body_tags = tags.clone();
    let mut body_w: Weights = body_tags.iter().map(|t| chain_weights(&chains[t.chain], t.s)).collect();

    // Cape: open sheet behind the back.
    let (nx, ny) = (12usize, 22usize);
    let mut cape_pos = Vec::new();
    let mut cape_tris = Vec::new();
    for iy in 0..=ny {
        for ix in 0..=nx {
            let u = ix as f64 / nx as f64;
            let v = iy as f64 / ny as f64;
            let x = -0.2 + 0.4 * u;
            let y = 1.44 - 0.84 * v;
            let z = -0.125 - 0.06 * v - 0.03 * (1.0 - (2.0 * u - 1.0).powi(2));
            cape_pos.push(v3(x * (1.0 + 0.3 * v), y, z));
        }
    }
    for iy in 0..ny {
        for ix in 0..nx {
            let a = (iy * (nx + 1) + ix) as u32;
            let b = a + 1;
            let c = a + (nx + 1) as u32;
            let d = c + 1;
            cape_tris.push([a, c, d]);
            cape_tris.push([a, d, b]);
        }
    }
    // A cape hangs from the upper back and follows the spine chain down to
    // the hips (never the legs or arms): torso weights by height.
    let mut cape_w: Weights = cape_pos.iter().map(|p| chain_weights(&chains[0], (p.y - 0.85).max(0.0))).collect();

    let mut rng = Rng::new(7);
    match fault {
        "noise" => {
            let spine = [j("DEF-spine"), j("DEF-spine.001"), j("DEF-spine.002"), j("DEF-spine.003")];
            for (v, t) in body_tags.iter().enumerate() {
                if t.chain == 0 && (0.2..0.5).contains(&t.s) && rng.f64() < 0.5 {
                    let b = spine[(rng.next_u64() % 4) as usize];
                    let mut vw = body_w[v].clone();
                    vw.push((b as u32, 0.4 + 0.4 * rng.f64()));
                    body_w[v] = normalize_vw(vw);
                }
            }
        }
        "bleed" => {
            let hand = j("DEF-hand.L") as u32;
            let thigh = j("DEF-thigh.L") as u32;
            for vw in body_w.iter_mut() {
                if vw.iter().any(|a| a.0 == hand && a.1 > 0.5) {
                    let mut x: VW = vw.iter().map(|&(b, w)| (b, w * 0.6)).collect();
                    x.push((thigh, 0.4));
                    *vw = normalize_vw(x);
                }
            }
        }
        "no_elbow_falloff" => {
            let ci = 2; // left arm chain
            let mut hard = Chain { bones: chains[ci].bones.clone(), bands: chains[ci].bands.clone() };
            hard.bands[1] = 0.0;
            for (v, t) in body_tags.iter().enumerate() {
                if t.chain == ci {
                    body_w[v] = chain_weights(&hard, t.s);
                }
            }
        }
        "elbow_collapse" => {
            let ci = 4; // right arm chain
            let el = chains[ci].bones[2].1;
            let (ua, fa) = (chains[ci].bones[1].0 as u32, chains[ci].bones[2].0 as u32);
            for (v, t) in body_tags.iter().enumerate() {
                if t.chain == ci && (t.s - el).abs() < 0.12 {
                    body_w[v] = normalize_vw(vec![(ua, 0.5), (fa, 0.5)]);
                }
            }
        }
        "knee_wide_falloff" => {
            // Smooth but far too wide: the whole knee region blends 50/50,
            // so a bent knee collapses over a long stretch (no tear).
            let ci = 5; // right leg chain
            let mut wide = Chain { bones: chains[ci].bones.clone(), bands: chains[ci].bands.clone() };
            wide.bands[1] = 0.3;
            for (v, t) in body_tags.iter().enumerate() {
                if t.chain == ci {
                    body_w[v] = chain_weights(&wide, t.s);
                }
            }
        }
        "cape_wrong_bone" => {
            let b = j("DEF-upper_arm.R") as u32;
            for vw in cape_w.iter_mut() {
                *vw = vec![(b, 1.0)];
            }
        }
        _ => {}
    }
    Fixture { bones, body_pos: pos, body_tris: tris, cape_pos, cape_tris, body_w, cape_w }
}

/// Ground truth for comparison: the clean twin's weights.
pub fn ground_truth() -> (Weights, Weights) {
    let f = build("clean");
    (f.body_w, f.cape_w)
}

fn normals(pos: &[Vec3], tris: &[[u32; 3]]) -> Vec<Vec3> {
    let mut n = vec![Vec3::ZERO; pos.len()];
    for t in tris {
        let f = (pos[t[1] as usize] - pos[t[0] as usize]).cross(pos[t[2] as usize] - pos[t[0] as usize]);
        for &i in t {
            n[i as usize] += f;
        }
    }
    n.into_iter()
        .map(|x| {
            let l = x.normalized();
            if l.len2() == 0.0 { v3(0.0, 1.0, 0.0) } else { l }
        })
        .collect()
}

/// Bone rest frames: +Y along the bone (Blender's convention), roll from
/// the world axis least aligned with it.
fn bone_frame(b: &Bone) -> Quat {
    let y = (b.tail - b.head).normalized();
    let refv = if y.dot(v3(0.0, 0.0, 1.0)).abs() < 0.9 { v3(0.0, 0.0, 1.0) } else { v3(1.0, 0.0, 0.0) };
    let x = y.cross(refv).normalized();
    let z = x.cross(y).normalized();
    quat_from_basis(x, y, z)
}

pub fn to_glb(f: &Fixture) -> Glb {
    let mut glb = Glb {
        json: json!({"asset": {"version": "2.0", "generator": "weightforge fixtures"}, "buffers": [{"byteLength": 0}]}),
        bin: Vec::new(),
        json_raw: None,
    };
    let nb = f.bones.len();
    let globals: Vec<Mat4> = f.bones.iter().map(|b| Mat4::from_trs(b.head, bone_frame(b), v3(1.0, 1.0, 1.0))).collect();
    let mut nodes = Vec::new();
    for (i, b) in f.bones.iter().enumerate() {
        let local = match b.parent {
            Some(p) => globals[p].inverse() * globals[i],
            None => globals[i],
        };
        let r = local.rotation();
        let t = local.translation();
        let children: Vec<usize> = f.bones.iter().enumerate().filter(|(_, c)| c.parent == Some(i)).map(|(k, _)| k).collect();
        let mut n = json!({"name": b.name, "translation": [t.x, t.y, t.z], "rotation": [r.x, r.y, r.z, r.w]});
        if !children.is_empty() {
            n["children"] = json!(children);
        }
        nodes.push(n);
    }
    let ibm: Vec<f32> = globals.iter().flat_map(|g| g.inverse().to_cols_vec().into_iter().map(|x| x as f32)).collect();
    let bv = glb.append_view(&f32_bytes(&ibm), None);
    let ibm_acc = glb.push_accessor(json!({"bufferView": bv, "componentType": 5126, "count": nb, "type": "MAT4"}));

    let mut meshes = Vec::new();
    for (name, pos, tris, w) in [("Body", &f.body_pos, &f.body_tris, &f.body_w), ("Cape", &f.cape_pos, &f.cape_tris, &f.cape_w)] {
        let p32: Vec<f32> = pos.iter().flat_map(|p| p.to_f32()).collect();
        let n32: Vec<f32> = normals(pos, tris).iter().flat_map(|p| p.to_f32()).collect();
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for c in p32.chunks(3) {
            for k in 0..3 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        let bvp = glb.append_view(&f32_bytes(&p32), Some(34962));
        let ap =
            glb.push_accessor(json!({"bufferView": bvp, "componentType": 5126, "count": pos.len(), "type": "VEC3", "min": lo, "max": hi}));
        let bvn = glb.append_view(&f32_bytes(&n32), Some(34962));
        let an = glb.push_accessor(json!({"bufferView": bvn, "componentType": 5126, "count": pos.len(), "type": "VEC3"}));
        let mut jb = Vec::new();
        let mut wf = Vec::new();
        for vw in w {
            let mut v = vw.clone();
            v.truncate(4);
            let v = normalize_vw(v);
            for k in 0..4 {
                let (jj, ww) = v.get(k).copied().unwrap_or((0, 0.0));
                jb.push(jj as u8);
                wf.push(ww as f32);
            }
        }
        let bvj = glb.append_view(&jb, Some(34962));
        let aj = glb.push_accessor(json!({"bufferView": bvj, "componentType": 5121, "count": pos.len(), "type": "VEC4"}));
        let bvw = glb.append_view(&f32_bytes(&wf), Some(34962));
        let aw = glb.push_accessor(json!({"bufferView": bvw, "componentType": 5126, "count": pos.len(), "type": "VEC4"}));
        let idx: Vec<u8> = tris.iter().flat_map(|t| t.iter().flat_map(|i| i.to_le_bytes())).collect();
        let bvi = glb.append_view(&idx, Some(34963));
        let ai = glb.push_accessor(json!({"bufferView": bvi, "componentType": 5125, "count": tris.len() * 3, "type": "SCALAR"}));
        meshes.push(json!({"name": name, "primitives": [{"attributes": {"POSITION": ap, "NORMAL": an, "JOINTS_0": aj, "WEIGHTS_0": aw}, "indices": ai}]}));
    }
    let body_node = nodes.len();
    nodes.push(json!({"name": "Body", "mesh": 0, "skin": 0}));
    nodes.push(json!({"name": "Cape", "mesh": 1, "skin": 0}));
    let arm = nodes.len();
    nodes.push(json!({"name": "Armature", "children": [0]}));
    glb.json["nodes"] = json!(nodes);
    glb.json["meshes"] = json!(meshes);
    glb.json["skins"] = json!([{"name": "Armature", "inverseBindMatrices": ibm_acc, "joints": (0..nb).collect::<Vec<_>>(), "skeleton": 0}]);
    glb.json["scenes"] = json!([{"nodes": [arm, body_node, body_node + 1]}]);
    glb.json["scene"] = json!(0);
    glb.sync_buffer_len();
    glb
}
