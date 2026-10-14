//! Scene model: skeleton + every skinned primitive, welded into one analysis
//! mesh. Seam duplicates (same mesh, same position) collapse to one vertex so
//! smoothing and geodesics see the real surface; separate meshes stay
//! separate shells. Weights are edited on welded vertices and written back
//! to every raw copy, so seams can never crack.

use crate::glb::{Error, Glb, Result, err};
use crate::math::{Mat4, Quat, Vec3, segment_distance, v3};
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Influences of one vertex: (joint index, weight), sorted by weight desc.
pub type VW = Vec<(u32, f64)>;
pub type Weights = Vec<VW>;

/// Mesh-name words that mark a detachable piece: rfcheck's `P_PIECE_BONES`
/// rule reads the same list from glbkit.
pub use glbkit::rig::PIECE_WORDS;

#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub parent: Option<usize>,
    pub t: Vec3,
    pub r: Quat,
    pub s: Vec3,
    pub matrix: Option<Mat4>,
}

impl Node {
    pub fn local(&self) -> Mat4 {
        self.matrix.unwrap_or_else(|| Mat4::from_trs(self.t, self.r, self.s))
    }
}

#[derive(Clone, Debug)]
pub struct Skeleton {
    pub nodes: Vec<Node>,
    /// Topological order (parents before children).
    pub order: Vec<usize>,
    /// Node index per joint.
    pub joints: Vec<usize>,
    pub names: Vec<String>,
    pub ibm: Vec<Mat4>,
    /// Nearest ancestor joint per joint.
    pub jparent: Vec<Option<usize>>,
    pub jchildren: Vec<Vec<usize>>,
    /// Rest global transforms per node.
    pub rest_global: Vec<Mat4>,
    pub head: Vec<Vec3>,
    pub tail: Vec<Vec3>,
    /// Twist/helper bones posed from their drivers (`helpers.rs`).
    pub helpers: Vec<crate::helpers::Helper>,
}

#[derive(Clone, Debug)]
pub struct Part {
    pub name: String,
    pub mesh: usize,
    pub node: usize,
    pub piece: bool,
}

#[derive(Clone, Debug)]
pub struct Prim {
    pub part: usize,
    pub mesh: usize,
    pub prim: usize,
    /// (JOINTS_n, WEIGHTS_n) accessor pairs.
    pub sets: Vec<(usize, usize)>,
    pub raw_to_weld: Vec<u32>,
    /// Stored JOINTS_0/WEIGHTS_0 rows (single-set prims), rewritten verbatim
    /// for vertices whose weights did not change.
    pub raw_rows: Option<Vec<([f64; 4], [f64; 4])>>,
}

pub struct Model {
    pub glb: Glb,
    pub skel: Skeleton,
    pub parts: Vec<Part>,
    pub prims: Vec<Prim>,
    /// Welded bind-space positions.
    pub bind: Vec<Vec3>,
    /// Rest-pose world positions (the reference shape for every metric).
    pub rest: Vec<Vec3>,
    pub tris: Vec<[u32; 3]>,
    pub vpart: Vec<u32>,
    pub weights: Weights,
    /// Welded vertices whose raw seam copies carried different weights.
    pub seam_split: Vec<u32>,
    /// Edges (i<j), unique, sorted.
    pub edges: Vec<[u32; 2]>,
    /// One-ring adjacency.
    pub adj: Vec<Vec<u32>>,
    /// Bounding-box diagonal of the rest shape (the scale every threshold uses).
    pub scale: f64,
}

fn vec3_of(v: Option<&Value>, d: Vec3) -> Vec3 {
    match v.and_then(Value::as_array) {
        Some(a) if a.len() == 3 => v3(a[0].as_f64().unwrap_or(d.x), a[1].as_f64().unwrap_or(d.y), a[2].as_f64().unwrap_or(d.z)),
        _ => d,
    }
}

pub fn parse_nodes(glb: &Glb) -> Vec<Node> {
    let raw = glb.arr("nodes");
    let mut nodes: Vec<Node> = raw
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let r = match n.get("rotation").and_then(Value::as_array) {
                Some(a) if a.len() == 4 => Quat {
                    x: a[0].as_f64().unwrap_or(0.0),
                    y: a[1].as_f64().unwrap_or(0.0),
                    z: a[2].as_f64().unwrap_or(0.0),
                    w: a[3].as_f64().unwrap_or(1.0),
                }
                .normalized(),
                _ => Quat::IDENTITY,
            };
            let matrix = n.get("matrix").and_then(Value::as_array).and_then(|a| {
                (a.len() == 16).then(|| Mat4::from_cols_slice(&a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect::<Vec<_>>()))
            });
            Node {
                name: n.get("name").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| format!("node{i}")),
                parent: None,
                t: vec3_of(n.get("translation"), Vec3::ZERO),
                r,
                s: vec3_of(n.get("scale"), v3(1.0, 1.0, 1.0)),
                matrix,
            }
        })
        .collect();
    for (i, n) in raw.iter().enumerate() {
        if let Some(ch) = n.get("children").and_then(Value::as_array) {
            for c in ch.iter().filter_map(Value::as_u64) {
                if let Some(nd) = nodes.get_mut(c as usize) {
                    nd.parent = Some(i);
                }
            }
        }
    }
    nodes
}

pub fn topo_order(nodes: &[Node]) -> Vec<usize> {
    let mut depth = vec![usize::MAX; nodes.len()];
    fn d(i: usize, nodes: &[Node], depth: &mut [usize], guard: usize) -> usize {
        if depth[i] != usize::MAX {
            return depth[i];
        }
        let v = match nodes[i].parent {
            Some(p) if guard < nodes.len() => d(p, nodes, depth, guard + 1) + 1,
            _ => 0,
        };
        depth[i] = v;
        v
    }
    for i in 0..nodes.len() {
        d(i, nodes, &mut depth, 0);
    }
    let mut order: Vec<usize> = (0..nodes.len()).collect();
    order.sort_by_key(|&i| (depth[i], i));
    order
}

pub fn globals(nodes: &[Node], order: &[usize]) -> Vec<Mat4> {
    let mut g = vec![Mat4::IDENTITY; nodes.len()];
    for &i in order {
        let l = nodes[i].local();
        g[i] = match nodes[i].parent {
            Some(p) => g[p] * l,
            None => l,
        };
    }
    g
}

impl Skeleton {
    fn build(glb: &Glb, skin_idx: usize) -> Result<Skeleton> {
        let nodes = parse_nodes(glb);
        let order = topo_order(&nodes);
        let skin = &glb.arr("skins")[skin_idx];
        let joints: Vec<usize> = skin
            .get("joints")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
            .unwrap_or_default();
        if joints.is_empty() {
            return err("skin has no joints");
        }
        if joints.iter().any(|&j| j >= nodes.len()) {
            return err("skin joint references a missing node");
        }
        let ibm = match skin.get("inverseBindMatrices").and_then(Value::as_u64) {
            Some(a) => {
                let (n, v) = glb.read_f64(a as usize)?;
                if n != 16 || v.len() / 16 < joints.len() {
                    return err("inverseBindMatrices accessor malformed");
                }
                v.chunks(16).take(joints.len()).map(Mat4::from_cols_slice).collect()
            }
            None => vec![Mat4::IDENTITY; joints.len()],
        };
        let names: Vec<String> = joints.iter().map(|&j| nodes[j].name.clone()).collect();
        let mut node_to_joint = vec![usize::MAX; nodes.len()];
        for (ji, &n) in joints.iter().enumerate() {
            node_to_joint[n] = ji;
        }
        let jparent: Vec<Option<usize>> = joints
            .iter()
            .map(|&n| {
                let mut p = nodes[n].parent;
                let mut guard = 0;
                while let Some(pi) = p {
                    if node_to_joint[pi] != usize::MAX {
                        return Some(node_to_joint[pi]);
                    }
                    p = nodes[pi].parent;
                    guard += 1;
                    if guard > nodes.len() {
                        break;
                    }
                }
                None
            })
            .collect();
        let mut jchildren = vec![Vec::new(); joints.len()];
        for (j, p) in jparent.iter().enumerate() {
            if let Some(p) = p {
                jchildren[*p].push(j);
            }
        }
        let rest_global = globals(&nodes, &order);
        let head: Vec<Vec3> = joints.iter().map(|&n| rest_global[n].translation()).collect();
        let helpers = crate::helpers::detect(glb.arr("nodes"), &nodes, &joints, &names);
        let mut sk = Skeleton { nodes, order, joints, names, ibm, jparent, jchildren, rest_global, tail: head.clone(), head, helpers };
        sk.tail = sk.estimate_tails();
        Ok(sk)
    }

    /// glTF stores no bone tails. If the rig follows Blender's convention
    /// (bone runs along the joint's local +Y, checked on single-child
    /// joints), tails follow +Y; otherwise they point at the child heads.
    /// Leaves extend along their own (or the parent's) direction.
    fn estimate_tails(&self) -> Vec<Vec3> {
        let n = self.joints.len();
        let yaxis: Vec<Vec3> =
            self.joints.iter().map(|&nd| self.rest_global[nd].transform_vector(v3(0.0, 1.0, 0.0)).normalized()).collect();
        let (mut agree, mut total) = (0, 0);
        for j in 0..n {
            if self.jchildren[j].len() == 1 {
                let d = self.head[self.jchildren[j][0]] - self.head[j];
                if d.len() > 1e-9 {
                    total += 1;
                    if d.normalized().dot(yaxis[j]) > 0.98 {
                        agree += 1;
                    }
                }
            }
        }
        let y_convention = total > 0 && agree * 10 >= total * 8;
        let mut tails = vec![Vec3::ZERO; n];
        let mut lens = vec![0.0; n];
        let mut done = vec![false; n];
        // Non-leaves first.
        for j in 0..n {
            let ch = &self.jchildren[j];
            let heads: Vec<Vec3> = ch.iter().map(|&c| self.head[c]).filter(|h| (*h - self.head[j]).len() > 1e-9).collect();
            if heads.is_empty() {
                continue;
            }
            if y_convention {
                let len = heads.iter().map(|h| (*h - self.head[j]).dot(yaxis[j])).fold(0.0, f64::max);
                if len > 1e-9 {
                    tails[j] = self.head[j] + yaxis[j] * len;
                    lens[j] = len;
                    done[j] = true;
                    continue;
                }
            }
            // The child that continues the chain: furthest from the head
            // along the parent's direction, else the farthest child.
            let dir = match self.jparent[j] {
                Some(p) => (self.head[j] - self.head[p]).normalized(),
                None => Vec3::ZERO,
            };
            let best = heads
                .iter()
                .copied()
                .max_by(|a, b| {
                    let ka = (*a - self.head[j]).dot(dir) + 1e-3 * (*a - self.head[j]).len();
                    let kb = (*b - self.head[j]).dot(dir) + 1e-3 * (*b - self.head[j]).len();
                    ka.partial_cmp(&kb).unwrap()
                })
                .unwrap();
            let best = if heads.len() == 1 { heads[0] } else { best };
            tails[j] = best;
            lens[j] = (best - self.head[j]).len();
            done[j] = true;
        }
        for j in 0..n {
            if done[j] {
                continue;
            }
            let plen = self.jparent[j].map(|p| lens[p]).unwrap_or(0.0);
            let len = if plen > 1e-9 { plen * 0.6 } else { 0.05 };
            let dir = if y_convention {
                yaxis[j]
            } else {
                match self.jparent[j] {
                    Some(p) if (self.head[j] - self.head[p]).len() > 1e-9 => (self.head[j] - self.head[p]).normalized(),
                    _ => yaxis[j],
                }
            };
            tails[j] = self.head[j] + dir * len;
            lens[j] = len;
        }
        tails
    }

    pub fn joint_by_name(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// Distance from p to joint j's rest segment.
    pub fn seg_dist(&self, j: usize, p: Vec3) -> f64 {
        segment_distance(p, self.head[j], self.tail[j]).1
    }

    pub fn bone_len(&self, j: usize) -> f64 {
        (self.tail[j] - self.head[j]).len()
    }
}

/// Picks the skin used by the most skinned vertices.
fn pick_skin(glb: &Glb) -> Option<usize> {
    let nskins = glb.arr("skins").len();
    if nskins == 0 {
        return None;
    }
    let mut votes = vec![0usize; nskins];
    for n in glb.arr("nodes") {
        if let (Some(m), Some(s)) = (n.get("mesh").and_then(Value::as_u64), n.get("skin").and_then(Value::as_u64)) {
            if let Some(mesh) = glb.arr("meshes").get(m as usize) {
                let c: usize = mesh
                    .get("primitives")
                    .and_then(Value::as_array)
                    .map(|ps| {
                        ps.iter()
                            .filter_map(|p| p.get("attributes")?.get("POSITION")?.as_u64())
                            .filter_map(|a| glb.arr("accessors").get(a as usize)?.get("count")?.as_u64())
                            .sum::<u64>() as usize
                    })
                    .unwrap_or(0);
                if (s as usize) < nskins {
                    votes[s as usize] += c.max(1);
                }
            }
        }
    }
    (0..nskins).max_by_key(|&s| (votes[s], usize::MAX - s))
}

fn tri_list(mode: u64, idx: &[u32]) -> Vec<[u32; 3]> {
    match mode {
        4 => idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
        5 => (2..idx.len()).map(|i| if i % 2 == 0 { [idx[i - 2], idx[i - 1], idx[i]] } else { [idx[i - 1], idx[i - 2], idx[i]] }).collect(),
        6 => (2..idx.len()).map(|i| [idx[0], idx[i - 1], idx[i]]).collect(),
        _ => Vec::new(),
    }
}

/// Merges, prunes zeros, and normalizes an influence list (sorted desc,
/// ties by joint index for determinism).
pub fn normalize_vw(mut vw: VW) -> VW {
    vw.sort_by_key(|a| a.0);
    let mut out: VW = Vec::with_capacity(vw.len());
    for (j, w) in vw {
        if !(w > 0.0) || !w.is_finite() {
            continue;
        }
        match out.last_mut() {
            Some(l) if l.0 == j => l.1 += w,
            _ => out.push((j, w)),
        }
    }
    let s: f64 = out.iter().map(|a| a.1).sum();
    if s > 0.0 {
        for a in &mut out {
            a.1 /= s;
        }
    }
    sort_vw(&mut out);
    out
}

pub fn sort_vw(vw: &mut VW) {
    vw.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
}

/// Keeps the `k` largest influences, drops those below `min_w`, renormalizes.
pub fn prune_vw(vw: &VW, k: usize, min_w: f64) -> VW {
    let mut v = normalize_vw(vw.clone());
    v.truncate(k);
    if v.len() > 1 {
        v.retain(|a| a.1 >= min_w);
    }
    normalize_vw(v)
}

pub fn weight_of(vw: &VW, j: u32) -> f64 {
    vw.iter().find(|a| a.0 == j).map(|a| a.1).unwrap_or(0.0)
}

impl Model {
    pub fn load(path: &std::path::Path) -> Result<Model> {
        Model::from_glb(Glb::read_path(path)?)
    }

    pub fn from_glb(glb: Glb) -> Result<Model> {
        let skin_idx = pick_skin(&glb).ok_or_else(|| Error("no skin in file (nothing to check)".into()))?;
        let skel = Skeleton::build(&glb, skin_idx)?;
        let nj = skel.joints.len();
        let mut parts = Vec::new();
        let mut prims = Vec::new();
        let mut bind: Vec<Vec3> = Vec::new();
        let mut vpart: Vec<u32> = Vec::new();
        let mut tris: Vec<[u32; 3]> = Vec::new();
        let mut acc_w: Vec<Vec<VW>> = Vec::new(); // raw weight copies per welded vertex
        let mut seen_mesh = std::collections::BTreeSet::new();

        // Quantization step for welding.
        let mut lo = v3(f64::MAX, f64::MAX, f64::MAX);
        let mut hi = -lo;
        for n in glb.arr("nodes") {
            if n.get("skin").and_then(Value::as_u64) != Some(skin_idx as u64) {
                continue;
            }
            let Some(m) = n.get("mesh").and_then(Value::as_u64) else { continue };
            for p in glb.arr("meshes").get(m as usize).and_then(|x| x.get("primitives")).and_then(Value::as_array).into_iter().flatten() {
                if let Some(a) = p.get("attributes").and_then(|a| a.get("POSITION")).and_then(Value::as_u64) {
                    if let Some(acc) = glb.arr("accessors").get(a as usize) {
                        lo = lo.min(vec3_of(acc.get("min"), lo));
                        hi = hi.max(vec3_of(acc.get("max"), hi));
                    }
                }
            }
        }
        let step = if hi.x >= lo.x { ((hi - lo).len() * 1e-6).max(1e-9) } else { 1e-6 };

        for (ni, n) in glb.arr("nodes").iter().enumerate() {
            if n.get("skin").and_then(Value::as_u64) != Some(skin_idx as u64) {
                continue;
            }
            let Some(mi) = n.get("mesh").and_then(Value::as_u64).map(|x| x as usize) else { continue };
            if !seen_mesh.insert(mi) {
                continue;
            }
            let Some(mesh) = glb.arr("meshes").get(mi) else { continue };
            let name = mesh.get("name").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| skel.nodes[ni].name.clone());
            let part = parts.len();
            let piece = glbkit::rig::is_piece_name(&name);
            parts.push(Part { name, mesh: mi, node: ni, piece });
            let mut weld: BTreeMap<(i64, i64, i64), u32> = BTreeMap::new();
            for (pi, p) in mesh.get("primitives").and_then(Value::as_array).into_iter().flatten().enumerate() {
                let attrs = p.get("attributes");
                let Some(pos_a) = attrs.and_then(|a| a.get("POSITION")).and_then(Value::as_u64) else { continue };
                let mut sets = Vec::new();
                for s in 0..8 {
                    let j = attrs.and_then(|a| a.get(format!("JOINTS_{s}"))).and_then(Value::as_u64);
                    let w = attrs.and_then(|a| a.get(format!("WEIGHTS_{s}"))).and_then(Value::as_u64);
                    match (j, w) {
                        (Some(j), Some(w)) => sets.push((j as usize, w as usize)),
                        _ => break,
                    }
                }
                if sets.is_empty() {
                    continue;
                }
                let pos = glb.read_vec3(pos_a as usize)?;
                let raw_rows = if sets.len() == 1 {
                    let (_, jv) = glb.read_f64(sets[0].0)?;
                    let (_, wv) = glb.read_f64(sets[0].1)?;
                    (jv.len() == pos.len() * 4 && wv.len() == pos.len() * 4).then(|| {
                        (0..pos.len())
                            .map(|v| {
                                (
                                    [jv[v * 4], jv[v * 4 + 1], jv[v * 4 + 2], jv[v * 4 + 3]],
                                    [wv[v * 4], wv[v * 4 + 1], wv[v * 4 + 2], wv[v * 4 + 3]],
                                )
                            })
                            .collect()
                    })
                } else {
                    None
                };
                let mut raw_w: Vec<VW> = vec![Vec::new(); pos.len()];
                for &(ja, wa) in &sets {
                    let (_, jv) = glb.read_f64(ja)?;
                    let (_, wv) = glb.read_f64(wa)?;
                    if jv.len() != pos.len() * 4 || wv.len() != pos.len() * 4 {
                        return err(format!("mesh {mi} prim {pi}: skinning accessor count mismatch"));
                    }
                    for v in 0..pos.len() {
                        for c in 0..4 {
                            let j = jv[v * 4 + c] as u32;
                            let w = wv[v * 4 + c];
                            if w > 0.0 && (j as usize) < nj {
                                raw_w[v].push((j, w));
                            }
                        }
                    }
                }
                let mut raw_to_weld = Vec::with_capacity(pos.len());
                for (v, p3) in pos.iter().enumerate() {
                    let key =
                        ((p3[0] as f64 / step).round() as i64, (p3[1] as f64 / step).round() as i64, (p3[2] as f64 / step).round() as i64);
                    let id = *weld.entry(key).or_insert_with(|| {
                        bind.push(Vec3::from_f32(*p3));
                        vpart.push(part as u32);
                        acc_w.push(Vec::new());
                        (bind.len() - 1) as u32
                    });
                    acc_w[id as usize].push(normalize_vw(std::mem::take(&mut raw_w[v])));
                    raw_to_weld.push(id);
                }
                let idx = match p.get("indices").and_then(Value::as_u64) {
                    Some(a) => glb.read_indices(a as usize)?,
                    None => (0..pos.len() as u32).collect(),
                };
                let mode = p.get("mode").and_then(Value::as_u64).unwrap_or(4);
                for t in tri_list(mode, &idx) {
                    if t.iter().any(|&i| i as usize >= pos.len()) {
                        return err(format!("mesh {mi} prim {pi}: index out of range"));
                    }
                    let w = [raw_to_weld[t[0] as usize], raw_to_weld[t[1] as usize], raw_to_weld[t[2] as usize]];
                    if w[0] != w[1] && w[1] != w[2] && w[0] != w[2] {
                        tris.push(w);
                    }
                }
                prims.push(Prim { part, mesh: mi, prim: pi, sets, raw_to_weld, raw_rows });
            }
        }
        if bind.is_empty() {
            return err("no skinned primitives (JOINTS_0/WEIGHTS_0) in file");
        }
        let mut seam_split = Vec::new();
        let weights: Weights = acc_w
            .into_iter()
            .enumerate()
            .map(|(i, copies)| {
                if copies.len() == 1 {
                    return copies.into_iter().next().unwrap();
                }
                let differ = copies
                    .iter()
                    .any(|c| c.len() != copies[0].len() || c.iter().any(|&(j, w)| (weight_of(&copies[0], j) - w).abs() > 1e-3));
                if differ {
                    seam_split.push(i as u32);
                }
                let n = copies.len() as f64;
                normalize_vw(copies.into_iter().flatten().map(|(j, w)| (j, w / n)).collect())
            })
            .collect();

        // Rest shape: S_j = G_j * IBM_j. Uniform S (bind == rest) is the norm;
        // otherwise blend with the file's own weights.
        let s: Vec<Mat4> = (0..nj).map(|j| skel.rest_global[skel.joints[j]] * skel.ibm[j]).collect();
        let uniform = s.iter().all(|m| m.m.iter().flatten().zip(s[0].m.iter().flatten()).all(|(a, b)| (a - b).abs() < 1e-4));
        let rest: Vec<Vec3> = bind
            .iter()
            .zip(&weights)
            .map(|(p, vw)| {
                if uniform || vw.is_empty() {
                    s[0].transform_point(*p)
                } else {
                    let mut acc = Vec3::ZERO;
                    for &(j, w) in vw {
                        acc += s[j as usize].transform_point(*p) * w;
                    }
                    acc
                }
            })
            .collect();
        let mut lo = v3(f64::MAX, f64::MAX, f64::MAX);
        let mut hi = -lo;
        for p in &rest {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let scale = (hi - lo).len().max(1e-9);
        let (edges, adj) = build_edges(bind.len(), &tris);
        Ok(Model { glb, skel, parts, prims, bind, rest, tris, vpart, weights, seam_split, edges, adj, scale })
    }

    /// Per mesh (in file order): (mesh name, node name, raw positions of
    /// all its skinned primitives concatenated, raw -> welded index).
    pub fn raw_meshes(&self) -> Vec<(String, String, Vec<[f32; 3]>, Vec<u32>)> {
        let mut out: Vec<(String, String, Vec<[f32; 3]>, Vec<u32>)> = Vec::new();
        for prim in &self.prims {
            let part = &self.parts[prim.part];
            let pos_acc =
                self.glb.json["meshes"][prim.mesh]["primitives"][prim.prim]["attributes"]["POSITION"].as_u64().unwrap_or(0) as usize;
            let pos = self.glb.read_vec3(pos_acc).unwrap_or_default();
            if out.last().is_none_or(|l| l.0 != part.name || l.1 != self.skel.nodes[part.node].name) {
                out.push((part.name.clone(), self.skel.nodes[part.node].name.clone(), Vec::new(), Vec::new()));
            }
            let last = out.last_mut().unwrap();
            last.2.extend(pos);
            last.3.extend(prim.raw_to_weld.iter().copied());
        }
        out
    }

    pub fn nverts(&self) -> usize {
        self.bind.len()
    }

    pub fn njoints(&self) -> usize {
        self.skel.joints.len()
    }

    /// Writes `w` into a copy of the source GLB: top-4 influences per
    /// vertex, exact partition of unity in the accessor's own storage type,
    /// in place when the storage allows, else as fresh accessors. Extra
    /// JOINTS_n/WEIGHTS_n sets are dropped (rfcheck allows one set).
    pub fn write_weights(&self, w: &Weights) -> Result<Glb> {
        let mut glb = Glb { json: self.glb.json.clone(), bin: self.glb.bin.clone(), json_raw: self.glb.json_raw.clone() };
        let nj = self.njoints();
        let mut written: BTreeMap<usize, Vec<[f64; 4]>> = BTreeMap::new();
        for prim in &self.prims {
            let (ja, wa) = prim.sets[0];
            let (wct, wnorm) = glb.component_type(wa).unwrap_or((5126, false));
            let mut jrows = Vec::with_capacity(prim.raw_to_weld.len());
            let mut wrows = Vec::with_capacity(prim.raw_to_weld.len());
            for (ri, &wi) in prim.raw_to_weld.iter().enumerate() {
                if let Some(raw) = &prim.raw_rows {
                    if w[wi as usize] == self.weights[wi as usize] && !self.seam_split.contains(&wi) {
                        jrows.push(raw[ri].0);
                        wrows.push(raw[ri].1);
                        continue;
                    }
                }
                let (jr, wr) = quantize_row(&w[wi as usize], wct, wnorm);
                jrows.push(jr);
                wrows.push(wr);
            }
            let mut put = |glb: &mut Glb, acc: usize, rows: Vec<[f64; 4]>, joints: bool| -> Result<usize> {
                if let Some(prev) = written.get(&acc) {
                    if *prev == rows {
                        return Ok(acc);
                    }
                } else {
                    let fits = if joints {
                        let (ct, _) = glb.component_type(acc).unwrap_or((0, false));
                        (ct == 5121 && nj <= 256) || ct == 5123
                    } else {
                        true
                    };
                    if fits && glb.overwrite_vec4(acc, &rows)? {
                        written.insert(acc, rows);
                        return Ok(acc);
                    }
                }
                let ct = if joints { 5123 } else { 5126 };
                let rows = if !joints && wct != 5126 {
                    // Re-quantized as floats: keep exact unit sums.
                    rows.iter()
                        .map(|r| {
                            let s: f64 = r.iter().sum();
                            if s > 0.0 { [r[0] / s, r[1] / s, r[2] / s, r[3] / s] } else { *r }
                        })
                        .collect()
                } else {
                    rows
                };
                let a = glb.append_vec4(&rows, ct);
                written.insert(a, rows);
                Ok(a)
            };
            let new_j = put(&mut glb, ja, jrows, true)?;
            let new_w = put(&mut glb, wa, wrows, false)?;
            let attrs = &mut glb.json["meshes"][prim.mesh]["primitives"][prim.prim]["attributes"];
            attrs["JOINTS_0"] = json!(new_j);
            attrs["WEIGHTS_0"] = json!(new_w);
            if let Some(o) = attrs.as_object_mut() {
                for s in 1..8 {
                    o.shift_remove(&format!("JOINTS_{s}"));
                    o.shift_remove(&format!("WEIGHTS_{s}"));
                }
            }
        }
        glb.sync_buffer_len();
        Ok(glb)
    }
}

/// Top-4 influences quantized so the stored weights sum to exactly one
/// unit of the storage type (largest weight absorbs rounding).
fn quantize_row(vw: &VW, ctype: u32, normalized: bool) -> ([f64; 4], [f64; 4]) {
    let mut v = normalize_vw(vw.clone());
    v.truncate(4);
    let v = normalize_vw(v);
    let mut j = [0.0; 4];
    let mut w = [0.0; 4];
    if v.is_empty() {
        return (j, w);
    }
    let unit = match (ctype, normalized) {
        (5121, true) => 255.0,
        (5123, true) => 65535.0,
        _ => 0.0,
    };
    for (i, &(ji, wi)) in v.iter().enumerate() {
        j[i] = ji as f64;
        w[i] = wi;
    }
    if unit > 0.0 {
        let mut q: Vec<f64> = w.iter().map(|x| (x * unit).round()).collect();
        let s: f64 = q.iter().sum();
        q[0] += unit - s;
        for i in 0..4 {
            w[i] = q[i] / unit;
        }
    } else {
        // f32 storage: make the f32 sum as close to 1 as representable.
        let f: Vec<f32> = w.iter().map(|&x| x as f32).collect();
        let s: f32 = f[1] + f[2] + f[3];
        w[0] = (1.0f32 - s) as f64;
        for i in 1..4 {
            w[i] = f[i] as f64;
        }
    }
    // Zero-weight slots point at joint 0 by convention.
    for i in 0..4 {
        if w[i] == 0.0 {
            j[i] = 0.0;
        }
    }
    (j, w)
}

pub fn build_edges(n: usize, tris: &[[u32; 3]]) -> (Vec<[u32; 2]>, Vec<Vec<u32>>) {
    let mut e: Vec<[u32; 2]> = Vec::with_capacity(tris.len() * 3);
    for t in tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            e.push(if a < b { [a, b] } else { [b, a] });
        }
    }
    e.sort_unstable();
    e.dedup();
    let mut adj = vec![Vec::new(); n];
    for &[a, b] in &e {
        adj[a as usize].push(b);
        adj[b as usize].push(a);
    }
    (e, adj)
}

/// Area-weighted vertex normals.
pub fn vertex_normals(pos: &[Vec3], tris: &[[u32; 3]]) -> Vec<Vec3> {
    let mut n = vec![Vec3::ZERO; pos.len()];
    for t in tris {
        let f = (pos[t[1] as usize] - pos[t[0] as usize]).cross(pos[t[2] as usize] - pos[t[0] as usize]);
        for &i in t {
            n[i as usize] += f;
        }
    }
    n.into_iter().map(Vec3::normalized).collect()
}
