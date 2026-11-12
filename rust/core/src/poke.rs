//! How deep one body part goes into another in a pose (a "poke": the
//! checker's `intersect` flag), measured for `weights probe --by
//! intersect` so a person's "fine" and "broken" can be laid against it.
//!
//! The checker only knows which triangles cross; this measures how far the
//! skin behind those crossings goes in. Each crossing pair names the part
//! being entered by the joint that carries most of the other triangle's
//! weight. From the flagged vertices, the skin is walked over its edges
//! for as long as it stays under that part's surface (behind its
//! triangles, whose outward side is the side away from the part's bone);
//! the deepest of those vertices, as its distance to that surface over the
//! part's limb radius (the median rest distance of its vertices to its
//! bone, clothes included), is the depth. 0 is touching, 1 is a full limb
//! radius in, about to come out of the far side.

use crate::math::{Vec3, segment_distance, v3};
use crate::metrics::self_intersection_pairs;
use crate::scene::Model;
use crate::transfer::closest_on_tri;
use std::collections::BTreeMap;

/// The deepest poke in one posed mesh.
#[derive(Clone, Debug, Default)]
pub struct Depth {
    /// Depth over the entered part's limb radius (0 when nothing goes in).
    pub share: f64,
    /// The same depth in model units.
    pub dist: f64,
    /// The joint whose part is entered deepest.
    pub into: Option<usize>,
}

/// The joint carrying the most weight on each vertex.
pub fn dominant_joints(model: &Model) -> Vec<Option<usize>> {
    model
        .weights
        .iter()
        .map(|vw| vw.iter().filter(|(_, x)| *x > 0.0).max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0))).map(|(j, _)| *j as usize))
        .collect()
}

/// Each joint's limb radius: the median rest distance to its bone of the
/// vertices it dominates, skin and clothes alike (0 when it has none).
pub fn joint_radii(model: &Model, dom: &[Option<usize>]) -> Vec<f64> {
    let mut per: Vec<Vec<f64>> = vec![Vec::new(); model.njoints()];
    for (v, j) in dom.iter().enumerate() {
        if let Some(j) = *j {
            per[j].push(model.skel.seg_dist(j, model.rest[v]));
        }
    }
    per.into_iter()
        .map(|mut d| {
            if d.is_empty() {
                return 0.0;
            }
            d.sort_by(|a, b| a.total_cmp(b));
            d[d.len() / 2]
        })
        .collect()
}

/// A triangle's part: the joint dominating two or three of its vertices,
/// else its first vertex's.
fn tri_joint(t: [u32; 3], dom: &[Option<usize>]) -> Option<usize> {
    let d = t.map(|v| dom[v as usize]);
    if d[1] == d[2] { d[1] } else { d[0] }
}

/// The deepest poke in `posed` (the mesh in one pose), whose posed bone
/// segments are `segs`; `sep` as in `metrics::self_intersections`.
pub fn deepest(model: &Model, posed: &[Vec3], segs: &[(Vec3, Vec3)], sep: f64) -> Depth {
    let dom = dominant_joints(model);
    let radius = joint_radii(model, &dom);
    let tj: Vec<Option<usize>> = model.tris.iter().map(|t| tri_joint(*t, &dom)).collect();
    // Seeds: the vertices of each crossing triangle, by the part they cross
    // into. A crossing inside one part has no other part to go into.
    let mut seeds: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for (a, b) in self_intersection_pairs(model, posed, sep) {
        let (ja, jb) = (tj[a as usize], tj[b as usize]);
        for (t, other) in [(a, jb), (b, ja)] {
            let (Some(j), mine) = (other, tj[t as usize]) else { continue };
            if mine == Some(j) {
                continue;
            }
            seeds.entry(j).or_default().extend(model.tris[t as usize].iter().filter(|&&v| dom[v as usize] != Some(j)));
        }
    }
    let mut best = Depth::default();
    for (j, mut start) in seeds {
        let r = radius[j];
        if r <= 0.0 || j >= segs.len() {
            continue;
        }
        start.sort_unstable();
        start.dedup();
        let d = depth_into(model, posed, &dom, &tj, j, segs[j], r, &start);
        if d / r > best.share {
            best = Depth { share: d / r, dist: d, into: Some(j) };
        }
    }
    best
}

/// How far the skin from `start` goes under part `j`'s surface (its
/// triangles `tj == j`, outward away from its bone `seg`, limb radius
/// `r`), walking over edges while it stays under.
#[allow(clippy::too_many_arguments)]
fn depth_into(
    model: &Model,
    posed: &[Vec3],
    dom: &[Option<usize>],
    tj: &[Option<usize>],
    j: usize,
    seg: (Vec3, Vec3),
    r: f64,
    start: &[u32],
) -> f64 {
    // The part's triangles near the crossing, each with its outward normal.
    let (mut lo, mut hi) = (v3(f64::MAX, f64::MAX, f64::MAX), v3(f64::MIN, f64::MIN, f64::MIN));
    for &v in start {
        lo = lo.min(posed[v as usize]);
        hi = hi.max(posed[v as usize]);
    }
    let pad = v3(3.0 * r, 3.0 * r, 3.0 * r);
    let (lo, hi) = (lo - pad, hi + pad);
    let inside_box = |p: Vec3| p.x >= lo.x && p.y >= lo.y && p.z >= lo.z && p.x <= hi.x && p.y <= hi.y && p.z <= hi.z;
    let mut part: Vec<([Vec3; 3], [u32; 3], Vec3)> = Vec::new();
    let mut edge_use: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for (ti, t) in model.tris.iter().enumerate() {
        if tj[ti] != Some(j) {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *edge_use.entry((a.min(b), a.max(b))).or_default() += 1;
        }
        let p = t.map(|v| posed[v as usize]);
        if !p.iter().any(|q| inside_box(*q)) {
            continue;
        }
        let mut n = (p[1] - p[0]).cross(p[2] - p[0]).normalized();
        let c = (p[0] + p[1] + p[2]) / 3.0;
        let (s, _) = segment_distance(c, seg.0, seg.1);
        if n.dot(c - (seg.0 + (seg.1 - seg.0) * s)) < 0.0 {
            n = -n;
        }
        part.push((p, *t, n));
    }
    if part.is_empty() {
        return 0.0;
    }
    // A closest point on the part's open edge says nothing about inside.
    let open_edge = |a: u32, b: u32| edge_use.get(&(a.min(b), a.max(b))).copied().unwrap_or(0) < 2;
    let open_vert = |v: u32| part.iter().any(|(_, t, _)| t.contains(&v) && t.iter().any(|&w| w != v && open_edge(v, w)));
    // Distance under the surface, or None when `p` is outside it.
    let under = |p: Vec3| -> Option<f64> {
        let mut near: Option<(f64, usize, [f64; 3])> = None;
        for (i, (q, _, _)) in part.iter().enumerate() {
            let b = closest_on_tri(p, q[0], q[1], q[2]);
            let d = (p - (q[0] * b[0] + q[1] * b[1] + q[2] * b[2])).len();
            if near.is_none_or(|n| d < n.0) {
                near = Some((d, i, b));
            }
        }
        let (d, i, b) = near?;
        let (q, t, n) = &part[i];
        if d > 2.0 * r {
            return None;
        }
        let zero: Vec<usize> = (0..3).filter(|&k| b[k] <= 1e-9).collect();
        match zero.len() {
            1 => {
                let k = zero[0];
                if open_edge(t[(k + 1) % 3], t[(k + 2) % 3]) {
                    return None;
                }
            }
            2 => {
                let k = (0..3).find(|k| !zero.contains(k)).unwrap_or(0);
                if open_vert(t[k]) {
                    return None;
                }
            }
            _ => {}
        }
        let foot = q[0] * b[0] + q[1] * b[1] + q[2] * b[2];
        ((p - foot).dot(*n) < 0.0).then_some(d)
    };
    let mut seen = vec![false; model.nverts()];
    let mut stack: Vec<u32> = start.to_vec();
    for &v in start {
        seen[v as usize] = true;
    }
    let mut deepest = 0.0f64;
    while let Some(v) = stack.pop() {
        let Some(d) = under(posed[v as usize]) else { continue };
        deepest = deepest.max(d);
        for &w in &model.adj[v as usize] {
            if !seen[w as usize] && dom[w as usize] != Some(j) {
                seen[w as usize] = true;
                stack.push(w);
            }
        }
    }
    deepest
}
