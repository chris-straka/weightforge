//! Deformation metrics. `Ctx` holds everything that does not depend on the
//! weights (poses, joint matrices, geodesics, regions); `evaluate` scores
//! one weight set against it, so `fix` can score many candidates cheaply.

use crate::math::{Mat4, Vec3, segment_distance};
use crate::poses::{Pose, PoseSet, SkelClass, default_poses, resolve_roles};
use crate::scene::{Model, Weights, weight_of};
use crate::skin::{joint_matrices, lbs};
use crate::voxel::{Geo, geodesics};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub struct Thresholds {
    /// Edge length ratio posed/rest above this is a tear.
    pub stretch: f64,
    /// Thickness ratio (distance to own bones, posed/rest) below this is a
    /// collapse (candy wrapper, crushed elbow).
    pub thin: f64,
    /// ...but only when, in one pose, the collapsed zone spans more than
    /// this many limb radii along the bone. Linear blend skinning always
    /// crushes the joint ring of a 140° elbow; bad weights crush a long
    /// stretch of the limb.
    pub thin_extent: f64,
    /// Weight >= this on a bone whose geodesic distance exceeds the
    /// nearest bone's by `bleed_excess * scale` is bleed.
    pub bleed_weight: f64,
    pub bleed_excess: f64,
    /// Lower median of the L1 weight distance to each one-ring neighbor:
    /// high only when a vertex disagrees with most of its neighbors
    /// (speckle), not along a clean hard boundary.
    pub noise: f64,
    /// Triangle pairs closer than this (fraction of scale) at rest are a
    /// local fold, not a self-intersection.
    pub intersect_sep: f64,
    /// Region fails when it has at least this many bad vertices.
    pub region_min_bad: usize,
    /// A piece vertex that ends up farther than this (fraction of scale)
    /// from where Data Transfer weights from the nearest body surface would
    /// put it does not follow the body.
    pub follow: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            stretch: 2.5,
            thin: 0.5,
            thin_extent: 1.5,
            bleed_weight: 0.05,
            bleed_excess: 0.15,
            noise: 0.5,
            intersect_sep: 0.12,
            region_min_bad: 3,
            follow: 0.08,
        }
    }
}

pub const F_STRETCH: u8 = 1;
pub const F_THIN: u8 = 2;
pub const F_BLEED: u8 = 4;
pub const F_NOISE: u8 = 8;
pub const F_INTERSECT: u8 = 16;
pub const F_UNWEIGHTED: u8 = 32;
pub const F_PIECE: u8 = 64;
pub const F_FOLLOW: u8 = 128;
pub const FLAG_NAMES: [(u8, &str); 8] = [
    (F_STRETCH, "stretch"),
    (F_THIN, "volume"),
    (F_BLEED, "bleed"),
    (F_NOISE, "noise"),
    (F_INTERSECT, "intersect"),
    (F_UNWEIGHTED, "unweighted"),
    (F_PIECE, "piece_bones"),
    (F_FOLLOW, "piece_follow"),
];

#[derive(Clone, Debug, Serialize)]
pub struct Region {
    pub name: String,
    pub label: String,
    pub piece: bool,
}

pub struct Ctx {
    pub class: SkelClass,
    pub poses: Vec<Pose>,
    pub mats: Vec<Vec<Mat4>>,
    pub geo: Geo,
    pub regions: Vec<Region>,
    pub vregion: Vec<u32>,
    pub rest_edge: Vec<f64>,
    pub th: Thresholds,
    /// Posed bone segments per pose (head, tail).
    pub segs: Vec<Vec<(Vec3, Vec3)>>,
    pub intersect: bool,
    /// The bone a (non-piece) region is named after.
    pub region_bone: Vec<Option<usize>>,
    /// Piece vertices: nearest body triangle, barycentrics, and whether the
    /// match is trusted (close and facing the same way) or inpainted.
    pub piece_matches: Vec<(u32, [u32; 3], [f64; 3], bool)>,
}

pub struct CtxOpts<'a> {
    pub voxel_res: usize,
    pub poses: Option<&'a PoseSet>,
    pub clips: bool,
    pub th: Thresholds,
    pub intersect: bool,
    /// Robust-transfer match distance (fraction of scale).
    pub match_dist: f64,
}

impl Default for CtxOpts<'_> {
    fn default() -> Self {
        CtxOpts { voxel_res: 128, poses: None, clips: true, th: Thresholds::default(), intersect: true, match_dist: 0.03 }
    }
}

fn pretty_role(role: &str) -> String {
    let (base, side) = match role.rsplit_once('.') {
        Some((b, "L")) => (b, "left "),
        Some((b, "R")) => (b, "right "),
        _ => (role, ""),
    };
    format!("{side}{}", base.replace('_', " ").replace('*', ""))
}

impl Ctx {
    pub fn new(model: &Model, o: &CtxOpts) -> Ctx {
        let (class, poses) = default_poses(model, o.poses, o.clips);
        let mats: Vec<Vec<Mat4>> = poses.par_iter().map(|p| joint_matrices(model, p)).collect();
        let all: Vec<usize> = (0..model.njoints()).collect();
        let geo = geodesics(model, o.voxel_res, &all);
        let sk = &model.skel;
        let segs = mats
            .iter()
            .map(|m| (0..model.njoints()).map(|j| (m[j].transform_point(sk.head[j]), m[j].transform_point(sk.tail[j]))).collect())
            .collect();
        // Regions: piece meshes by name; everything else by nearest bone
        // (geodesic, so bad weights cannot move a vertex's region).
        let roles = resolve_roles(sk);
        let mut role_of: BTreeMap<usize, String> = BTreeMap::new();
        for (r, js) in &roles {
            if r.ends_with('*') || r == "spine" {
                continue;
            }
            for &j in js {
                role_of.entry(j).or_insert_with(|| r.clone());
            }
        }
        let mut regions: Vec<Region> = Vec::new();
        let mut index: BTreeMap<String, u32> = BTreeMap::new();
        let mut vregion = Vec::with_capacity(model.nverts());
        for v in 0..model.nverts() {
            let part = &model.parts[model.vpart[v] as usize];
            let (name, label, piece) = if part.piece {
                (format!("piece:{}", part.name), part.name.clone(), true)
            } else {
                let j = geo.nearest(v).map(|e| e.0 as usize).unwrap_or(0);
                let label = role_of.get(&j).map(|r| pretty_role(r)).unwrap_or_else(|| sk.names[j].clone());
                (sk.names[j].clone(), label, false)
            };
            let id = *index.entry(name.clone()).or_insert_with(|| {
                regions.push(Region { name, label, piece });
                (regions.len() - 1) as u32
            });
            vregion.push(id);
        }
        let rest_edge = model.edges.iter().map(|e| (model.rest[e[0] as usize] - model.rest[e[1] as usize]).len()).collect();
        let body_tris: Vec<[u32; 3]> =
            model.tris.iter().filter(|t| t.iter().all(|&i| !model.parts[model.vpart[i as usize] as usize].piece)).copied().collect();
        let piece_verts: Vec<u32> = (0..model.nverts() as u32).filter(|&v| model.parts[model.vpart[v as usize] as usize].piece).collect();
        let piece_matches = if piece_verts.is_empty() || body_tris.is_empty() {
            Vec::new()
        } else {
            let normals = crate::scene::vertex_normals(&model.rest, &model.tris);
            let index = crate::transfer::TriIndex::new(&model.rest, body_tris);
            crate::transfer::match_subset(
                &index,
                &normals,
                &model.rest,
                &normals,
                &piece_verts,
                o.match_dist * model.scale,
                35f64.to_radians().cos(),
            )
        };
        let region_bone = regions.iter().map(|r| if r.piece { None } else { sk.joint_by_name(&r.name) }).collect();
        Ctx {
            class,
            poses,
            mats,
            geo,
            regions,
            vregion,
            rest_edge,
            th: o.th.clone(),
            segs,
            intersect: o.intersect,
            region_bone,
            piece_matches,
        }
    }
}

/// Per-vertex results for one weight set.
pub struct Eval {
    pub flags: Vec<u8>,
    /// Continuous deformation energy per vertex (lower is better).
    pub energy: Vec<f64>,
    /// Pose index where each vertex was worst (for the sheet).
    pub worst_pose: Vec<u32>,
    /// Per pose: vertices with a failing flag (self-intersection, a
    /// warning, not counted).
    pub pose_bad: Vec<usize>,
    /// Per pose: (vertex, flags) of every vertex that is bad in that pose.
    pub bad_in_pose: Vec<Vec<(u32, u8)>>,
    /// Bleed details: vertex -> (bone, weight, excess distance).
    pub bleed: Vec<Option<(u32, f64, f64)>>,
    pub piece_bones: Vec<(String, Vec<String>)>,
    pub max_stretch: Vec<f64>,
    pub min_thin: Vec<f64>,
}

/// Speckle score: high only when a vertex disagrees with its neighbors both
/// against their mean (so not a smooth gradient, where the mean matches)
/// and against most of them individually (so not a clean hard step, where
/// half the neighbors agree).
pub fn noise_of(model: &Model, w: &Weights, v: usize) -> f64 {
    let n = &model.adj[v];
    if n.len() < 3 {
        return 0.0;
    }
    let l1 = |a: &crate::scene::VW, b: &crate::scene::VW| -> f64 {
        a.iter().map(|&(j, x)| (x - weight_of(b, j)).abs()).sum::<f64>()
            + b.iter().filter(|e| weight_of(a, e.0) == 0.0).map(|e| e.1).sum::<f64>()
    };
    let mut d: Vec<f64> = n.iter().map(|&u| l1(&w[v], &w[u as usize])).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = d[(d.len() - 1) / 2];
    let mut mean: crate::scene::VW = Vec::new();
    let k = n.len() as f64;
    for &u in n {
        for &(j, x) in &w[u as usize] {
            match mean.iter_mut().find(|e| e.0 == j) {
                Some(e) => e.1 += x / k,
                None => mean.push((j, x / k)),
            }
        }
    }
    median.min(l1(&w[v], &mean))
}

/// Static (pose-free) bleed: weight on a bone far from the vertex through
/// the body. Returns the worst offender.
pub fn bleed_of(ctx: &Ctx, model: &Model, w: &Weights, v: usize) -> Option<(u32, f64, f64)> {
    let near = ctx.geo.nearest(v)?.1 as f64;
    let mut worst: Option<(u32, f64, f64)> = None;
    for &(j, wt) in &w[v] {
        if wt < ctx.th.bleed_weight {
            continue;
        }
        let ex = ctx.geo.dist(v, j) as f64 - near;
        if ex > ctx.th.bleed_excess * model.scale && worst.is_none_or(|b| ex * wt > b.2 * b.1) {
            worst = Some((j, wt, ex));
        }
    }
    worst
}

/// Soft bleed energy: weight times excess geodesic distance (in scale units).
fn bleed_energy(ctx: &Ctx, model: &Model, w: &Weights, v: usize) -> f64 {
    let Some(near) = ctx.geo.nearest(v) else { return 0.0 };
    let mut e = 0.0;
    for &(j, wt) in &w[v] {
        let ex = (ctx.geo.dist(v, j) - near.1) as f64 / model.scale;
        e += wt * (ex - 0.5 * ctx.th.bleed_excess).max(0.0);
    }
    e * 4.0
}

/// Detachable pieces skinned to bones no body mesh uses (rfcheck's
/// `P_PIECE_BONES`, same rule).
pub fn piece_bones(model: &Model, w: &Weights) -> Vec<(String, Vec<String>, Vec<u32>)> {
    let nj = model.njoints();
    let mut body = vec![false; nj];
    for v in 0..model.nverts() {
        if !model.parts[model.vpart[v] as usize].piece {
            for &(j, wt) in &w[v] {
                if wt > 0.0 {
                    body[j as usize] = true;
                }
            }
        }
    }
    if !body.iter().any(|&b| b) {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (pi, part) in model.parts.iter().enumerate() {
        if !part.piece {
            continue;
        }
        let mut bad: Vec<u32> = Vec::new();
        let mut verts = Vec::new();
        for v in 0..model.nverts() {
            if model.vpart[v] as usize != pi {
                continue;
            }
            let mut hit = false;
            for &(j, wt) in &w[v] {
                if wt > 0.0 && !body[j as usize] {
                    bad.push(j);
                    hit = true;
                }
            }
            if hit {
                verts.push(v as u32);
            }
        }
        bad.sort_unstable();
        bad.dedup();
        if !bad.is_empty() {
            let mut names: Vec<String> = bad.iter().map(|&j| model.skel.names[j as usize].clone()).collect();
            names.sort();
            out.push((part.name.clone(), names, verts));
        }
    }
    out
}

pub fn evaluate(model: &Model, ctx: &Ctx, w: &Weights) -> Eval {
    let nv = model.nverts();
    let th = &ctx.th;
    let sk = &model.skel;
    // Rest thickness: distance to the vertex's own (w >= 0.1) bone segments.
    let own: Vec<Vec<u32>> = w
        .iter()
        .map(|vw| {
            let mut o: Vec<u32> = vw.iter().filter(|a| a.1 >= 0.1).map(|a| a.0).collect();
            if o.is_empty() {
                o.extend(vw.first().map(|a| a.0));
            }
            o
        })
        .collect();
    let rest_thick: Vec<f64> = (0..nv)
        .map(|v| {
            own[v].iter().map(|&j| segment_distance(model.rest[v], sk.head[j as usize], sk.tail[j as usize]).1).fold(f64::MAX, f64::min)
        })
        .collect();
    let min_thick = 0.01 * model.scale;

    struct PoseOut {
        stretch: Vec<f64>,
        thin: Vec<f64>,
        isect: Vec<bool>,
        follow: Vec<f64>,
    }
    // Attached piece vertices (close to the body, facing the same way) must
    // move like Data Transfer from the body would move them. Free-hanging
    // parts are an artistic choice; stretch and noise still cover them.
    let mut expected: Vec<Option<crate::scene::VW>> = vec![None; nv];
    for &(v, t, b, ok) in &ctx.piece_matches {
        if ok {
            expected[v as usize] = Some(crate::transfer::blend(w, t, b));
        }
    }
    let per_pose: Vec<PoseOut> = (0..ctx.poses.len())
        .into_par_iter()
        .map(|p| {
            let posed = lbs(model, &ctx.mats[p], w);
            let mut stretch = vec![1.0f64; nv];
            for (ei, e) in model.edges.iter().enumerate() {
                let r0 = ctx.rest_edge[ei];
                if r0 < 1e-12 {
                    continue;
                }
                let r = (posed[e[0] as usize] - posed[e[1] as usize]).len() / r0;
                for &k in e {
                    if r > stretch[k as usize] {
                        stretch[k as usize] = r;
                    }
                }
            }
            let segs = &ctx.segs[p];
            let thin: Vec<f64> = (0..nv)
                .map(|v| {
                    if rest_thick[v] < min_thick || own[v].is_empty() {
                        return 1.0;
                    }
                    let d = own[v]
                        .iter()
                        .map(|&j| segment_distance(posed[v], segs[j as usize].0, segs[j as usize].1).1)
                        .fold(f64::MAX, f64::min);
                    d / rest_thick[v]
                })
                .collect();
            let isect = if ctx.intersect { self_intersections(model, &posed, th.intersect_sep * model.scale) } else { vec![false; nv] };
            let follow: Vec<f64> = (0..nv)
                .map(|v| match &expected[v] {
                    Some(e) => (crate::skin::lbs_vertex(&ctx.mats[p], e, model.rest[v]) - posed[v]).len() / model.scale,
                    None => 0.0,
                })
                .collect();
            PoseOut { stretch, thin, isect, follow }
        })
        .collect();

    let mut flags = vec![0u8; nv];
    let mut energy = vec![0.0f64; nv];
    let mut pose_max = vec![0.0f64; nv];
    let mut worst_pose = vec![0u32; nv];
    let mut worst_val = vec![0.0f64; nv];
    let mut pose_bad = vec![0usize; ctx.poses.len()];
    let mut bad_in_pose = vec![Vec::new(); ctx.poses.len()];
    let mut max_stretch = vec![1.0f64; nv];
    let mut min_thin = vec![1.0f64; nv];
    let np = ctx.poses.len().max(1) as f64;
    // The attached vertices carry the whole piece: their drift energy is
    // scaled by piece size / attached count, so a torn-off cape costs like
    // a cape, not like nine vertices.
    let mut attach_scale = vec![0.0f64; nv];
    {
        let mut piece_n = vec![0usize; model.parts.len()];
        let mut att_n = vec![0usize; model.parts.len()];
        for v in 0..nv {
            piece_n[model.vpart[v] as usize] += 1;
        }
        for m in ctx.piece_matches.iter().filter(|m| m.3) {
            att_n[model.vpart[m.0 as usize] as usize] += 1;
        }
        for m in ctx.piece_matches.iter().filter(|m| m.3) {
            let p = model.vpart[m.0 as usize] as usize;
            attach_scale[m.0 as usize] = piece_n[p] as f64 / att_n[p].max(1) as f64;
        }
    }
    // Limb radius per region: median rest thickness.
    let nr = ctx.regions.len();
    let mut radius = vec![0.0f64; nr];
    {
        let mut per: Vec<Vec<f64>> = vec![Vec::new(); nr];
        for v in 0..nv {
            if rest_thick[v] < f64::MAX {
                per[ctx.vregion[v] as usize].push(rest_thick[v]);
            }
        }
        for (r, mut xs) in per.into_iter().enumerate() {
            if !xs.is_empty() {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                radius[r] = xs[xs.len() / 2];
            }
        }
    }
    for (p, po) in per_pose.iter().enumerate() {
        // Long collapse zones only: per region, the thin vertices of this
        // pose must span > thin_extent radii along the region's bone.
        let mut thin_ok = vec![false; nv];
        let mut by_region: Vec<Vec<u32>> = vec![Vec::new(); nr];
        for v in 0..nv {
            if po.thin[v] < th.thin {
                by_region[ctx.vregion[v] as usize].push(v as u32);
            }
        }
        for (r, vs) in by_region.iter().enumerate() {
            let Some(j) = ctx.region_bone[r] else { continue };
            if vs.len() < th.region_min_bad || radius[r] <= 0.0 {
                continue;
            }
            let (h, d) = (sk.head[j], (sk.tail[j] - sk.head[j]).normalized());
            let (mut lo, mut hi) = (f64::MAX, f64::MIN);
            for &v in vs {
                let t = (model.rest[v as usize] - h).dot(d);
                lo = lo.min(t);
                hi = hi.max(t);
            }
            if hi - lo > th.thin_extent * radius[r] {
                for &v in vs {
                    thin_ok[v as usize] = true;
                }
            }
        }
        for v in 0..nv {
            let s = po.stretch[v];
            let t = po.thin[v];
            max_stretch[v] = max_stretch[v].max(s);
            min_thin[v] = min_thin[v].min(t);
            let mut f = 0u8;
            if s > th.stretch {
                f |= F_STRETCH;
            }
            if thin_ok[v] {
                f |= F_THIN;
            }
            if po.isect[v] {
                f |= F_INTERSECT;
            }
            if po.follow[v] > th.follow {
                f |= F_FOLLOW;
            }
            // Soft costs: little for what clean LBS does anyway (moderate
            // stretch, the crushed joint ring, folds), a lot for tears,
            // long collapses, and pieces leaving the body.
            let e = (s - 1.6).max(0.0)
                + 2.0 * (0.5 - t).max(0.0)
                + if thin_ok[v] { 0.5 + 2.0 * (th.thin - t).max(0.0) } else { 0.0 }
                + if po.isect[v] { 0.05 } else { 0.0 }
                + 20.0 * (po.follow[v] - 0.03).max(0.0) * attach_scale[v];
            // Half the pose mean, half the worst pose: a tear that shows in
            // one pose of thirty-five still counts.
            energy[v] += 0.5 * e / np;
            pose_max[v] = pose_max[v].max(e);
            let badness = (s - th.stretch).max(0.0)
                + (th.thin - t).max(0.0)
                + (po.follow[v] - th.follow).max(0.0)
                + if po.isect[v] { 0.01 } else { 0.0 };
            if badness > worst_val[v] {
                worst_val[v] = badness;
                worst_pose[v] = p as u32;
            }
            if f & !F_INTERSECT != 0 {
                pose_bad[p] += 1;
            }
            if f != 0 {
                bad_in_pose[p].push((v as u32, f));
            }
            flags[v] |= f;
        }
    }
    for v in 0..nv {
        energy[v] += 0.5 * pose_max[v];
    }
    let bleed: Vec<Option<(u32, f64, f64)>> = (0..nv).into_par_iter().map(|v| bleed_of(ctx, model, w, v)).collect();
    let noise: Vec<f64> = (0..nv).into_par_iter().map(|v| noise_of(model, w, v)).collect();
    for v in 0..nv {
        if bleed[v].is_some() {
            flags[v] |= F_BLEED;
        }
        if noise[v] > th.noise {
            flags[v] |= F_NOISE;
        }
        if w[v].is_empty() {
            flags[v] |= F_UNWEIGHTED;
        }
        energy[v] += bleed_energy(ctx, model, w, v) + 2.0 * (noise[v] - 0.3).max(0.0) + if w[v].is_empty() { 1.0 } else { 0.0 };
    }
    let pb = piece_bones(model, w);
    for (_, _, verts) in &pb {
        for &v in verts {
            flags[v as usize] |= F_PIECE;
            energy[v as usize] += 1.0;
        }
    }
    Eval {
        flags,
        energy,
        worst_pose,
        pose_bad,
        bad_in_pose,
        bleed,
        piece_bones: pb.into_iter().map(|(a, b, _)| (a, b)).collect(),
        max_stretch,
        min_thin,
    }
}

// ------------------------------------------------------- self-intersection

fn tri_tri(a: [Vec3; 3], b: [Vec3; 3]) -> bool {
    // Separating axis test on the 2 normals + 9 edge cross products.
    let ea = [a[1] - a[0], a[2] - a[1], a[0] - a[2]];
    let eb = [b[1] - b[0], b[2] - b[1], b[0] - b[2]];
    let na = ea[0].cross(ea[1]);
    let nb = eb[0].cross(eb[1]);
    let mut axes = vec![na, nb];
    for x in &ea {
        for y in &eb {
            axes.push(x.cross(*y));
        }
    }
    for ax in axes {
        if ax.len2() < 1e-24 {
            continue;
        }
        let pa = a.map(|p| p.dot(ax));
        let pb = b.map(|p| p.dot(ax));
        let (amin, amax) = (pa[0].min(pa[1]).min(pa[2]), pa[0].max(pa[1]).max(pa[2]));
        let (bmin, bmax) = (pb[0].min(pb[1]).min(pb[2]), pb[0].max(pb[1]).max(pb[2]));
        let eps = 1e-12 * (1.0 + amax.abs() + bmax.abs());
        if amax < bmin + eps || bmax < amin + eps {
            return false;
        }
    }
    true
}

/// Vertices of triangles that intersect in the pose but not at rest, and
/// whose rest centroids are farther apart than `sep` (so the natural fold
/// inside a bent elbow does not count).
pub fn self_intersections(model: &Model, posed: &[Vec3], sep: f64) -> Vec<bool> {
    let nv = model.nverts();
    let mut hit = vec![false; nv];
    let tris = &model.tris;
    if tris.is_empty() {
        return hit;
    }
    let mean_edge = {
        let mut s = 0.0;
        for t in tris.iter().take(4096) {
            s += (posed[t[0] as usize] - posed[t[1] as usize]).len();
        }
        (s / tris.len().min(4096) as f64).max(1e-9)
    };
    let cell = mean_edge * 2.0;
    let mut grid: BTreeMap<(i64, i64, i64), Vec<u32>> = BTreeMap::new();
    for (ti, t) in tris.iter().enumerate() {
        let p = t.map(|i| posed[i as usize]);
        let lo = p[0].min(p[1]).min(p[2]);
        let hi = p[0].max(p[1]).max(p[2]);
        let (x0, y0, z0) = ((lo.x / cell).floor() as i64, (lo.y / cell).floor() as i64, (lo.z / cell).floor() as i64);
        let (x1, y1, z1) = ((hi.x / cell).floor() as i64, (hi.y / cell).floor() as i64, (hi.z / cell).floor() as i64);
        if (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1) > 64 {
            continue; // huge stretched triangle: the stretch metric owns it
        }
        for x in x0..=x1 {
            for y in y0..=y1 {
                for z in z0..=z1 {
                    grid.entry((x, y, z)).or_default().push(ti as u32);
                }
            }
        }
    }
    let rest_c: Vec<Vec3> =
        tris.iter().map(|t| (model.rest[t[0] as usize] + model.rest[t[1] as usize] + model.rest[t[2] as usize]) / 3.0).collect();
    let mut tested = std::collections::BTreeSet::new();
    for list in grid.values() {
        if list.len() > 256 {
            continue;
        }
        for i in 0..list.len() {
            for k in i + 1..list.len() {
                let (a, b) = (list[i] as usize, list[k] as usize);
                let (ta, tb) = (tris[a], tris[b]);
                if ta.iter().any(|x| tb.contains(x)) {
                    continue;
                }
                if (rest_c[a] - rest_c[b]).len() < sep {
                    continue;
                }
                if !tested.insert((a.min(b), a.max(b))) {
                    continue;
                }
                let pa = ta.map(|x| posed[x as usize]);
                let pb = tb.map(|x| posed[x as usize]);
                if tri_tri(pa, pb) && !tri_tri(ta.map(|x| model.rest[x as usize]), tb.map(|x| model.rest[x as usize])) {
                    for &x in ta.iter().chain(tb.iter()) {
                        hit[x as usize] = true;
                    }
                }
            }
        }
    }
    hit
}
