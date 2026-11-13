//! Scene model: skeleton + every skinned primitive, welded into one analysis
//! mesh. Seam duplicates (same mesh, same position) collapse to one vertex so
//! smoothing and geodesics see the real surface; separate meshes stay
//! separate shells. Weights are edited on welded vertices and written back
//! to every raw copy, so seams can never crack.

use crate::glb::{Glb, Result, err};
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
    /// Nearest ancestor joint per joint, or for a joint the file leaves
    /// without one (a skeleton exported flat), the joint it hangs from by
    /// position (`vlink`).
    pub jparent: Vec<Option<usize>>,
    /// Joints whose `jparent` is such a link and not in the file: a pose
    /// moves them with that joint.
    pub vlink: Vec<bool>,
    /// Per node: the node its flat joint hangs from (`link_flat`).
    pub vlink_node: Vec<Option<usize>>,
    /// Node order for posing with those links (parents first).
    pub vorder: Vec<usize>,
    pub jchildren: Vec<Vec<usize>>,
    /// Rest global transforms per node.
    pub rest_global: Vec<Mat4>,
    pub head: Vec<Vec3>,
    pub tail: Vec<Vec3>,
    /// Twist/helper bones posed from their drivers (`helpers.rs`).
    pub helpers: Vec<crate::helpers::Helper>,
    /// Bone roles from the skeleton's structure and the body's shape
    /// (`anatomy.rs`), set once the mesh is loaded.
    pub anatomy: crate::anatomy::Anatomy,
}

#[derive(Clone, Debug)]
pub struct Part {
    pub name: String,
    pub mesh: usize,
    pub node: usize,
    pub piece: bool,
    /// An unskinned mesh riding a joint (checked, never written).
    pub rigid: bool,
}

#[derive(Clone, Debug)]
pub struct Prim {
    pub part: usize,
    pub mesh: usize,
    pub prim: usize,
    /// Position in `Model::skins` of the skin posing it; None for a rigid
    /// mesh (no weights in the file).
    pub skin: Option<usize>,
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
    /// Every skin of the file, joints mapped into `skel`'s joint list.
    pub skins: Vec<SkinInfo>,
    pub parts: Vec<Part>,
    pub prims: Vec<Prim>,
    /// Welded bind-space positions.
    pub bind: Vec<Vec3>,
    /// Rest-pose world positions (the reference shape for every metric).
    pub rest: Vec<Vec3>,
    pub tris: Vec<[u32; 3]>,
    pub vpart: Vec<u32>,
    /// Vertices of rigid-parented meshes (one weight, on their joint).
    pub rigid: Vec<bool>,
    pub weights: Weights,
    /// Welded vertices whose raw seam copies carried different weights.
    pub seam_split: Vec<u32>,
    /// Edges (i<j), unique, sorted.
    pub edges: Vec<[u32; 2]>,
    /// One-ring adjacency.
    pub adj: Vec<Vec<u32>>,
    /// Bounding-box diagonal of the rest shape (the scale every threshold uses).
    pub scale: f64,
    /// Welded vertices with weight on a joint index past the end of the
    /// skin's joint list (an invalid file), and the largest such index.
    /// Those weights are dropped and the rest renormalised on load.
    pub bad_joints: (usize, u32),
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

/// One skin of the file: its joints as indices into the model's joint list
/// (every skin's joints, merged), and its own inverse bind matrices.
#[derive(Clone, Debug)]
pub struct SkinInfo {
    pub index: usize,
    pub joints: Vec<u32>,
    pub ibm: Vec<Mat4>,
}

/// Reads every skin. Joints are merged by node in file order (skin 0's
/// first), so a one-skin file keeps its joint indices.
fn read_skins(glb: &Glb, nnodes: usize) -> Result<(Vec<usize>, Vec<Mat4>, Vec<SkinInfo>)> {
    let mut joints: Vec<usize> = Vec::new();
    let mut ibm: Vec<Mat4> = Vec::new();
    let mut of_node: BTreeMap<usize, u32> = BTreeMap::new();
    let mut skins = Vec::new();
    for (si, skin) in glb.arr("skins").iter().enumerate() {
        let local: Vec<usize> = skin
            .get("joints")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
            .unwrap_or_default();
        if local.is_empty() {
            continue;
        }
        if local.iter().any(|&j| j >= nnodes) {
            return err(format!("skin {si} joint references a missing node"));
        }
        let m: Vec<Mat4> = match skin.get("inverseBindMatrices").and_then(Value::as_u64) {
            Some(a) => {
                let (n, v) = glb.read_f64(a as usize)?;
                if n != 16 || v.len() / 16 < local.len() {
                    return err(format!("skin {si}: inverseBindMatrices accessor malformed"));
                }
                v.chunks(16).take(local.len()).map(Mat4::from_cols_slice).collect()
            }
            None => vec![Mat4::IDENTITY; local.len()],
        };
        let mut map = Vec::with_capacity(local.len());
        for (l, &n) in local.iter().enumerate() {
            let id = *of_node.entry(n).or_insert_with(|| {
                joints.push(n);
                ibm.push(m[l]);
                (joints.len() - 1) as u32
            });
            map.push(id);
        }
        skins.push(SkinInfo { index: si, joints: map, ibm: m });
    }
    if joints.is_empty() {
        return err("no skin with joints in file (nothing to check)");
    }
    Ok((joints, ibm, skins))
}

impl Skeleton {
    fn build(glb: &Glb, nodes: Vec<Node>, joints: Vec<usize>, ibm: Vec<Mat4>) -> Result<Skeleton> {
        let order = topo_order(&nodes);
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
        let n_joints = joints.len();
        let head: Vec<Vec3> = joints.iter().map(|&n| rest_global[n].translation()).collect();
        let helpers = crate::helpers::detect(glb.arr("nodes"), &nodes, &joints, &names);
        let mut sk = Skeleton {
            nodes,
            order,
            joints,
            names,
            ibm,
            jparent,
            jchildren,
            vlink: vec![false; n_joints],
            vlink_node: Vec::new(),
            vorder: Vec::new(),
            rest_global,
            tail: head.clone(),
            head,
            helpers,
            anatomy: Default::default(),
        };
        sk.tail = sk.estimate_tails();
        Ok(sk)
    }

    /// glTF stores no bone tails, so each joint's is decided on its own
    /// (it used to be one vote over the whole skeleton, and 30 grafted
    /// finger bones that follow Blender's +Y convention flipped it for a
    /// body that does not, swinging the thighs ~55 degrees):
    /// - a joint with children ends at a child's head: the child on its
    ///   own +Y axis when there is one (Blender's convention, kept per
    ///   joint), else the child that continues the chain;
    /// - a leaf extends along its +Y when its parent's +Y pointed at it
    ///   (that branch follows the convention), else along its parent's
    ///   direction.
    fn estimate_tails(&self) -> Vec<Vec3> {
        let n = self.joints.len();
        let yaxis: Vec<Vec3> =
            self.joints.iter().map(|&nd| self.rest_global[nd].transform_vector(v3(0.0, 1.0, 0.0)).normalized()).collect();
        let mut tails = vec![Vec3::ZERO; n];
        let mut lens = vec![0.0; n];
        let mut done = vec![false; n];
        // Joints whose +Y axis runs through the child their tail is pinned to.
        let mut on_y = vec![false; n];
        for j in 0..n {
            let ch: Vec<usize> = self.jchildren[j].iter().copied().filter(|&c| (self.head[c] - self.head[j]).len() > 1e-9).collect();
            if ch.is_empty() {
                continue;
            }
            let d = |c: usize| self.head[c] - self.head[j];
            // A child on the joint's own +Y line (within ~11 degrees),
            // the nearest such one.
            let along_y = ch
                .iter()
                .copied()
                .filter(|&c| d(c).normalized().dot(yaxis[j]) > 0.98)
                .min_by(|&a, &b| d(a).len().partial_cmp(&d(b).len()).unwrap().then(a.cmp(&b)));
            let pick = match along_y {
                Some(c) => {
                    on_y[j] = true;
                    c
                }
                None if ch.len() == 1 => ch[0],
                None => {
                    // The child that continues the chain: furthest from the
                    // head along the parent's direction, else the farthest.
                    let dir = match self.jparent[j] {
                        Some(p) => (self.head[j] - self.head[p]).normalized(),
                        None => Vec3::ZERO,
                    };
                    let key = |c: usize| d(c).dot(dir) + 1e-3 * d(c).len();
                    ch.iter().copied().max_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap().then(b.cmp(&a))).unwrap()
                }
            };
            tails[j] = self.head[pick];
            lens[j] = d(pick).len();
            done[j] = true;
        }
        for j in 0..n {
            if done[j] {
                continue;
            }
            let plen = self.jparent[j].map(|p| lens[p]).unwrap_or(0.0);
            let len = if plen > 1e-9 { plen * 0.6 } else { 0.05 };
            let parent_dir = match self.jparent[j] {
                Some(p) if (self.head[j] - self.head[p]).len() > 1e-9 => Some((self.head[j] - self.head[p]).normalized()),
                _ => None,
            };
            let follows_y = self.jparent[j].is_none_or(|p| on_y[p]);
            let dir = match parent_dir {
                Some(pd) if !follows_y => pd,
                _ => yaxis[j],
            };
            tails[j] = self.head[j] + dir * len;
            lens[j] = len;
        }
        tails
    }

    /// A skeleton exported flat (deform bones without the bones that
    /// joined them: a forearm with no hand under it) bends one bone at a
    /// time and tears at every joint. When three or more joints carry
    /// weight with no joint above them, each such joint is hung from the
    /// joint it continues: the one whose +Y axis runs through its head
    /// (Blender's bones point at their children), else the nearest head,
    /// a shared vertex counting for it. The tree grows from the joint
    /// nearest the pelvis (half the body's height, its middle), cheapest
    /// link first, so chains join in order. Links are marked in `vlink`.
    pub fn link_flat(&mut self, rest: &[Vec3], weights: &Weights) {
        let n = self.joints.len();
        let mut weighted = vec![false; n];
        let mut shared: std::collections::BTreeSet<(u32, u32)> = std::collections::BTreeSet::new();
        for vw in weights {
            for (a, &(j, x)) in vw.iter().enumerate() {
                if x >= 0.02 {
                    weighted[j as usize] = true;
                    for &(k, y) in &vw[a + 1..] {
                        if y >= 0.02 {
                            shared.insert((j.min(k), j.max(k)));
                        }
                    }
                }
            }
        }
        let roots: Vec<usize> = (0..n).filter(|&j| self.jparent[j].is_none()).collect();
        if roots.iter().filter(|&&r| self.subtree(r).iter().any(|&j| weighted[j])).count() < 3 || rest.is_empty() {
            return;
        }
        let (mut lo, mut hi) = (rest[0], rest[0]);
        for p in rest {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let extent = (hi - lo).len().max(1e-9);
        let pelvis = v3(0.5 * (lo.x + hi.x), lo.y + 0.5 * (hi.y - lo.y), 0.5 * (lo.z + hi.z));
        let yaxis: Vec<Vec3> =
            self.joints.iter().map(|&nd| self.rest_global[nd].transform_vector(v3(0.0, 1.0, 0.0)).normalized()).collect();
        // How far each joint's own vertices reach along its +Y: where its
        // segment ends, so where a child continuing it starts.
        let mut reach: Vec<Vec<f64>> = vec![Vec::new(); n];
        for (p, vw) in rest.iter().zip(weights) {
            if let Some(&(j, x)) = vw.first() {
                if x >= 0.5 {
                    let j = j as usize;
                    reach[j].push((*p - self.head[j]).dot(yaxis[j]));
                }
            }
        }
        let ext: Vec<Option<f64>> = reach
            .into_iter()
            .map(|mut r| {
                if r.len() < 4 {
                    return None;
                }
                r.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Some(r[(r.len() * 95 / 100).min(r.len() - 1)])
            })
            .collect();
        // Cost of hanging orphan r from joint k: on k's +Y axis near where
        // k's vertices end is a continuation; anything else costs its
        // distance. A shared vertex counts for the pair.
        let on_ray = |r: usize, k: usize| -> Option<f64> {
            let d = self.head[r] - self.head[k];
            let t = d.dot(yaxis[k]);
            let perp = (d - yaxis[k] * t).len();
            (t > 1e-9 && perp < 0.03 * extent).then(|| {
                let end = match ext[k] {
                    Some(e) => (t - e).abs(),
                    None => 0.3 * t,
                };
                (perp + end) / extent
            })
        };
        let cost = |r: usize, k: usize| -> f64 {
            let d = self.head[r] - self.head[k];
            let mut c = on_ray(r, k).unwrap_or(0.5 + d.len() / extent);
            if shared.contains(&(r.min(k) as u32, r.max(k) as u32)) {
                c -= 0.2;
            }
            c
        };
        let subs: Vec<Vec<usize>> = (0..n).map(|j| if self.jparent[j].is_none() { self.subtree(j) } else { Vec::new() }).collect();
        let mut in_sub = vec![usize::MAX; n];
        for &r in &roots {
            for &j in &subs[r] {
                in_sub[j] = r;
            }
        }
        // First every orphan that continues a bone (on its axis) takes the
        // best such bone in the whole skeleton; loops are cut at their
        // dearest link.
        let mut link: Vec<Option<(usize, f64)>> = vec![None; n];
        for &r in &roots {
            link[r] = (0..n)
                .filter(|&k| in_sub[k] != r)
                .filter_map(|k| on_ray(r, k).map(|c| (k, c - if shared.contains(&(r.min(k) as u32, r.max(k) as u32)) { 0.2 } else { 0.0 })))
                .filter(|x| x.1 < 0.05)
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        }
        for &r in &roots {
            // Walk the orphan roots' links; a walk back to r is a loop.
            let mut seen = vec![r];
            let mut cur = r;
            while let Some((k, _)) = link[cur] {
                let next = in_sub[k];
                if next == usize::MAX {
                    break;
                }
                if seen.contains(&next) {
                    if next == r {
                        let worst = *seen.iter().max_by(|&&a, &&b| link[a].unwrap().1.partial_cmp(&link[b].unwrap().1).unwrap()).unwrap();
                        link[worst] = None;
                    }
                    break;
                }
                seen.push(next);
                cur = next;
            }
        }
        // Then the root (the hips: between the tops of the two lowest
        // chains, one each side), and Prim's tree for the chain tops left.
        let top_of = |mut j: usize| -> usize {
            for _ in 0..n {
                let up = self.jparent[j].or_else(|| link[j].map(|x| x.0));
                match up {
                    Some(p) if self.head[p].y >= self.head[j].y - 1e-9 => j = p,
                    _ => break,
                }
            }
            j
        };
        let cx = 0.5 * (lo.x + hi.x);
        let lowest = |side: f64| {
            (0..n)
                .filter(|&j| weighted[j] && (self.head[j].x - cx) * side > 0.01 * extent)
                .min_by(|&a, &b| self.head[a].y.partial_cmp(&self.head[b].y).unwrap().then(a.cmp(&b)))
        };
        let hips_at = match (lowest(1.0), lowest(-1.0)) {
            (Some(a), Some(b)) => (self.head[top_of(a)] + self.head[top_of(b)]) * 0.5,
            _ => pelvis,
        };
        // Units: an orphan with its certain chain below it.
        let tops: Vec<usize> = roots.iter().copied().filter(|&r| link[r].is_none()).collect();
        let main = tops
            .iter()
            .copied()
            .filter(|&r| subs[r].iter().any(|&j| weighted[j]))
            .min_by(|&a, &b| (self.head[a] - hips_at).len().partial_cmp(&(self.head[b] - hips_at).len()).unwrap().then(a.cmp(&b)))
            .unwrap_or(roots[0]);
        let mut links: Vec<(usize, usize)> = roots.iter().filter_map(|&r| link[r].map(|x| (r, x.0))).collect();
        // Members of each top's unit (its subtree plus certain chains).
        let unit_of = |j: usize| -> usize {
            let mut r = in_sub[j];
            for _ in 0..n {
                match link[r] {
                    Some((k, _)) if in_sub[k] != usize::MAX => r = in_sub[k],
                    _ => break,
                }
            }
            r
        };
        let unit: Vec<usize> = (0..n).map(|j| if in_sub[j] == usize::MAX { usize::MAX } else { unit_of(j) }).collect();
        let mut in_tree = vec![false; n];
        let mut best: Vec<(f64, usize)> = vec![(f64::MAX, usize::MAX); n];
        let mut pending: Vec<usize> = tops.iter().copied().filter(|&r| r != main).collect();
        let members = |u: usize| -> Vec<usize> { (0..n).filter(|&j| unit[j] == u).collect() };
        let join = |u: usize, in_tree: &mut Vec<bool>, best: &mut Vec<(f64, usize)>, pending: &[usize]| {
            let m = members(u);
            for &k in &m {
                in_tree[k] = true;
            }
            for &r in pending {
                for &k in &m {
                    let c = cost(r, k);
                    if c < best[r].0 {
                        best[r] = (c, k);
                    }
                }
            }
        };
        join(main, &mut in_tree, &mut best, &pending);
        while !pending.is_empty() {
            let (li, &r) =
                pending.iter().enumerate().min_by(|a, b| best[*a.1].0.partial_cmp(&best[*b.1].0).unwrap().then(a.1.cmp(b.1))).unwrap();
            pending.swap_remove(li);
            links.push((r, best[r].1));
            join(r, &mut in_tree, &mut best, &pending);
        }
        for (r, k) in links {
            self.jparent[r] = Some(k);
            self.jchildren[k].push(r);
            self.vlink[r] = true;
        }
        for c in &mut self.jchildren {
            c.sort_unstable();
        }
        self.tail = self.estimate_tails();
        self.vlink_node = vec![None; self.nodes.len()];
        for j in 0..n {
            if self.vlink[j] {
                self.vlink_node[self.joints[j]] = self.jparent[j].map(|p| self.joints[p]);
            }
        }
        self.vorder = self.pose_order();
    }

    /// Node order for posing: parents (and the joints a flat joint hangs
    /// from) before children.
    pub fn pose_order(&self) -> Vec<usize> {
        if !self.vlink.iter().any(|&v| v) {
            return self.order.clone();
        }
        let nn = self.nodes.len();
        let mut joint_of = vec![usize::MAX; nn];
        for (j, &nd) in self.joints.iter().enumerate() {
            joint_of[nd] = j;
        }
        let mut done = vec![false; nn];
        let mut out = Vec::with_capacity(nn);
        fn visit(i: usize, s: &Skeleton, joint_of: &[usize], done: &mut [bool], out: &mut Vec<usize>, depth: usize) {
            if done[i] || depth > s.nodes.len() {
                return;
            }
            done[i] = true;
            if let Some(p) = s.nodes[i].parent {
                visit(p, s, joint_of, done, out, depth + 1);
            }
            let j = joint_of[i];
            if j != usize::MAX && s.vlink[j] {
                if let Some(pj) = s.jparent[j] {
                    visit(s.joints[pj], s, joint_of, done, out, depth + 1);
                }
            }
            out.push(i);
        }
        for &i in &self.order {
            visit(i, self, &joint_of, &mut done, &mut out, 0);
        }
        out
    }

    /// The node a flat joint's node hangs from, when `link_flat` linked it.
    pub fn vlink_parent_node(&self, node: usize) -> Option<usize> {
        self.vlink_node.get(node).copied().flatten()
    }

    /// Node order for posing (`pose_order`, cached).
    pub fn posing_order(&self) -> &[usize] {
        if self.vorder.is_empty() { &self.order } else { &self.vorder }
    }

    /// A joint and every joint under it.
    pub fn subtree(&self, j: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = vec![j];
        while let Some(x) = stack.pop() {
            out.push(x);
            stack.extend(self.jchildren[x].iter().copied());
        }
        out
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
        let nodes = parse_nodes(&glb);
        let (joint_nodes, ibm, skins) = read_skins(&glb, nodes.len())?;
        let mut skel = Skeleton::build(&glb, nodes, joint_nodes, ibm)?;
        let mut node_joint = vec![None; skel.nodes.len()];
        for (j, &n) in skel.joints.iter().enumerate() {
            node_joint[n] = Some(j as u32);
        }
        let skin_pos = |s: u64| skins.iter().position(|k| k.index == s as usize);
        // Rest shape per skin: S_j = G_j * IBM_j. Uniform S (bind == rest)
        // is the norm; otherwise each vertex blends its skin's S by weight.
        let skin_s: Vec<Vec<Mat4>> = skins
            .iter()
            .map(|k| k.joints.iter().zip(&k.ibm).map(|(&j, m)| skel.rest_global[skel.joints[j as usize]] * *m).collect())
            .collect();
        let skin_uniform: Vec<bool> = skin_s
            .iter()
            .map(|s| s.iter().all(|m| m.m.iter().flatten().zip(s[0].m.iter().flatten()).all(|(a, b)| (a - b).abs() < 1e-4)))
            .collect();
        let mut parts = Vec::new();
        let mut prims = Vec::new();
        let mut bind: Vec<Vec3> = Vec::new();
        let mut rest: Vec<Vec3> = Vec::new();
        let mut vpart: Vec<u32> = Vec::new();
        let mut rigid: Vec<bool> = Vec::new();
        let mut tris: Vec<[u32; 3]> = Vec::new();
        let mut acc_w: Vec<Vec<VW>> = Vec::new(); // raw weight copies per welded vertex (skin-local joints)
        let mut vskin: Vec<Option<usize>> = Vec::new();
        let mut bad_joint: Vec<bool> = Vec::new(); // per welded vertex
        let mut bad_joint_max = 0u32;
        let mut seen_mesh = std::collections::BTreeSet::new();

        for (ni, n) in glb.arr("nodes").iter().enumerate() {
            let Some(mi) = n.get("mesh").and_then(Value::as_u64).map(|x| x as usize) else { continue };
            let Some(mesh) = glb.arr("meshes").get(mi) else { continue };
            // A skinned mesh is posed by its skin; an unskinned mesh under a
            // joint (a robot's plates, a sword in the hand) rides that joint
            // rigidly and is checked too (it can pass through the body),
            // but its weights are never written.
            let (skin, attach) = match n.get("skin").and_then(Value::as_u64) {
                Some(s) => match skin_pos(s) {
                    Some(k) => (Some(k), None),
                    None => continue,
                },
                None => {
                    let mut p = Some(ni);
                    let mut found = None;
                    let mut guard = 0;
                    while let Some(q) = p {
                        if let Some(j) = node_joint[q] {
                            found = Some(j);
                            break;
                        }
                        p = skel.nodes[q].parent;
                        guard += 1;
                        if guard > skel.nodes.len() {
                            break;
                        }
                    }
                    match found {
                        Some(j) => (None, Some(j)),
                        None => continue,
                    }
                }
            };
            // The same mesh on two nodes of one skin is one surface; a rigid
            // mesh instanced twice is two.
            if !seen_mesh.insert((mi, skin.map(|k| k as isize).unwrap_or(-1 - ni as isize))) {
                continue;
            }
            let name = mesh.get("name").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| skel.nodes[ni].name.clone());
            let part = parts.len();
            let piece = glbkit::rig::is_piece_name(&name);
            parts.push(Part { name, mesh: mi, node: ni, piece, rigid: attach.is_some() });
            let node_g = skel.rest_global[ni];
            let mut weld: BTreeMap<(i64, i64, i64), u32> = BTreeMap::new();
            let prim_list: Vec<&Value> = mesh.get("primitives").and_then(Value::as_array).into_iter().flatten().collect();
            // Weld step from this mesh's extent.
            let (mut lo, mut hi) = (v3(f64::MAX, f64::MAX, f64::MAX), v3(f64::MIN, f64::MIN, f64::MIN));
            for p in &prim_list {
                if let Some(acc) = p
                    .get("attributes")
                    .and_then(|a| a.get("POSITION"))
                    .and_then(Value::as_u64)
                    .and_then(|a| glb.arr("accessors").get(a as usize))
                {
                    lo = lo.min(vec3_of(acc.get("min"), lo));
                    hi = hi.max(vec3_of(acc.get("max"), hi));
                }
            }
            let step = if hi.x >= lo.x { ((hi - lo).len() * 1e-6).max(1e-9) } else { 1e-6 };
            let snj = skin.map(|k| skins[k].joints.len()).unwrap_or(0);
            for (pi, p) in prim_list.iter().enumerate() {
                let attrs = p.get("attributes");
                let Some(pos_a) = attrs.and_then(|a| a.get("POSITION")).and_then(Value::as_u64) else { continue };
                let mut sets = Vec::new();
                if skin.is_some() {
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
                }
                let mode = p.get("mode").and_then(Value::as_u64).unwrap_or(4);
                if !matches!(mode, 4..=6) {
                    continue; // points and lines have no surface to check
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
                let mut raw_bad = vec![false; pos.len()];
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
                            if w > 0.0 && (j as usize) < snj {
                                raw_w[v].push((j, w));
                            } else if w > 0.0 {
                                raw_bad[v] = true;
                                bad_joint_max = bad_joint_max.max(j);
                            }
                        }
                    }
                }
                if let Some(j) = attach {
                    for vw in &mut raw_w {
                        vw.push((j, 1.0));
                    }
                }
                let mut raw_to_weld = Vec::with_capacity(pos.len());
                for (v, p3) in pos.iter().enumerate() {
                    let key =
                        ((p3[0] as f64 / step).round() as i64, (p3[1] as f64 / step).round() as i64, (p3[2] as f64 / step).round() as i64);
                    let id = *weld.entry(key).or_insert_with(|| {
                        bind.push(Vec3::from_f32(*p3));
                        rest.push(Vec3::ZERO);
                        vpart.push(part as u32);
                        rigid.push(attach.is_some());
                        vskin.push(skin);
                        acc_w.push(Vec::new());
                        bad_joint.push(false);
                        (bind.len() - 1) as u32
                    });
                    bad_joint[id as usize] |= raw_bad[v];
                    acc_w[id as usize].push(normalize_vw(std::mem::take(&mut raw_w[v])));
                    raw_to_weld.push(id);
                }
                let idx = match p.get("indices").and_then(Value::as_u64) {
                    Some(a) => glb.read_indices(a as usize)?,
                    None => (0..pos.len() as u32).collect(),
                };
                for t in tri_list(mode, &idx) {
                    if t.iter().any(|&i| i as usize >= pos.len()) {
                        return err(format!("mesh {mi} prim {pi}: index out of range"));
                    }
                    let w = [raw_to_weld[t[0] as usize], raw_to_weld[t[1] as usize], raw_to_weld[t[2] as usize]];
                    if w[0] != w[1] && w[1] != w[2] && w[0] != w[2] {
                        tris.push(w);
                    }
                }
                prims.push(Prim { part, mesh: mi, prim: pi, skin, sets, raw_to_weld, raw_rows });
            }
            // Rigid meshes: rest is the node's rest transform of the mesh.
            if attach.is_some() {
                for v in 0..bind.len() {
                    if vpart[v] as usize == part {
                        rest[v] = node_g.transform_point(bind[v]);
                    }
                }
            }
        }
        if bind.is_empty() {
            return err("no skinned primitives (JOINTS_0/WEIGHTS_0) in file");
        }
        let mut seam_split = Vec::new();
        // Welded weights, still in each skin's own joint numbering.
        let local_w: Weights = acc_w
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
        for v in 0..bind.len() {
            if let Some(k) = vskin[v] {
                let (s, vw) = (&skin_s[k], &local_w[v]);
                rest[v] = if skin_uniform[k] || vw.is_empty() {
                    s[0].transform_point(bind[v])
                } else {
                    let mut acc = Vec3::ZERO;
                    for &(j, w) in vw {
                        acc += s[j as usize].transform_point(bind[v]) * w;
                    }
                    acc
                };
            }
        }
        let weights: Weights = local_w
            .into_iter()
            .zip(&vskin)
            .map(|(vw, sk)| match sk {
                Some(k) => normalize_vw(vw.into_iter().map(|(j, w)| (skins[*k].joints[j as usize], w)).collect()),
                None => vw,
            })
            .collect();
        skel.link_flat(&rest, &weights);
        skel.anatomy = crate::anatomy::analyze(&skel, &rest, &weights);
        let mut lo = v3(f64::MAX, f64::MAX, f64::MAX);
        let mut hi = -lo;
        for p in &rest {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let scale = (hi - lo).len().max(1e-9);
        let (edges, adj) = build_edges(bind.len(), &tris);
        let bad_joints = (bad_joint.iter().filter(|&&b| b).count(), bad_joint_max);
        Ok(Model { glb, skel, skins, parts, prims, bind, rest, tris, vpart, rigid, weights, seam_split, edges, adj, scale, bad_joints })
    }

    /// Puts back the file's own weights on rigid-parented vertices: those
    /// ride their joint whatever a repair proposes, and are never written.
    pub fn lock_rigid(&self, w: &mut Weights) {
        for v in 0..self.nverts() {
            if self.rigid[v] {
                w[v] = self.weights[v].clone();
            }
        }
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
        // Each skin's own joint numbering; a joint a skin's vertices now
        // need but the skin lacks is appended to it, with an inverse bind
        // matrix that keeps the skin's rest shape.
        let mut local: Vec<BTreeMap<u32, u32>> =
            self.skins.iter().map(|k| k.joints.iter().enumerate().map(|(l, &j)| (j, l as u32)).collect()).collect();
        let mut added: Vec<Vec<u32>> = vec![Vec::new(); self.skins.len()];
        for prim in &self.prims {
            let Some(k) = prim.skin else { continue };
            for &wi in &prim.raw_to_weld {
                for &(j, x) in &w[wi as usize] {
                    if x > 0.0 && !local[k].contains_key(&j) {
                        let l = local[k].len() as u32;
                        local[k].insert(j, l);
                        added[k].push(j);
                    }
                }
            }
        }
        for (k, add) in added.iter().enumerate() {
            if add.is_empty() {
                continue;
            }
            let info = &self.skins[k];
            let s_ref = self.skel.rest_global[self.skel.joints[info.joints[0] as usize]] * info.ibm[0];
            let mut ibm: Vec<f32> = info.ibm.iter().flat_map(|m| m.to_cols_vec()).map(|x| x as f32).collect();
            for &j in add {
                let m = self.skel.rest_global[self.skel.joints[j as usize]].inverse() * s_ref;
                ibm.extend(m.to_cols_vec().into_iter().map(|x| x as f32));
            }
            let count = info.joints.len() + add.len();
            let bv = glb.append_view(&crate::glb::f32_bytes(&ibm), None);
            let acc = glb.push_accessor(json!({"bufferView": bv, "componentType": 5126, "count": count, "type": "MAT4"}));
            let skin = &mut glb.json["skins"][info.index];
            if let Some(js) = skin.get_mut("joints").and_then(Value::as_array_mut) {
                for &j in add {
                    js.push(json!(self.skel.joints[j as usize]));
                }
            }
            skin["inverseBindMatrices"] = json!(acc);
        }
        let mut written: BTreeMap<usize, Vec<[f64; 4]>> = BTreeMap::new();
        for prim in &self.prims {
            let Some(k) = prim.skin else { continue };
            let nj = local[k].len();
            let to_local = |vw: &VW| -> VW { vw.iter().filter_map(|&(j, x)| local[k].get(&j).map(|&l| (l, x))).collect() };
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
                let (jr, wr) = quantize_row(&to_local(&w[wi as usize]), wct, wnorm);
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
