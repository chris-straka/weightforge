//! Weight repair. Builds candidate weight sets, scores each on the same pose
//! set, keeps the best candidate per broken region, blends at region seams,
//! and never returns anything that scores worse than the input in any
//! region (the input is always candidate 0).

use crate::math::Vec3;
use crate::metrics::{Ctx, Eval, evaluate, evaluate_with};
use crate::report::{Report, build, region_energy};
use crate::scene::{Model, VW, Weights, normalize_vw, prune_vw, sort_vw, weight_of};
use crate::skin::dqs;
use crate::transfer::{TriIndex, inpaint_from_matches, match_subset};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Method {
    Smooth,
    Geodesic,
    Transfer,
    Optimize,
}

impl Method {
    pub fn parse(s: &str) -> Option<Vec<Method>> {
        Some(match s {
            "auto" => vec![Method::Smooth, Method::Geodesic, Method::Transfer, Method::Optimize],
            "smooth" | "cleanup" => vec![Method::Smooth],
            "geodesic" => vec![Method::Geodesic],
            "transfer" => vec![Method::Transfer],
            "optimize" => vec![Method::Optimize],
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Method::Smooth => "smooth",
            Method::Geodesic => "geodesic",
            Method::Transfer => "transfer",
            Method::Optimize => "optimize",
        }
    }
}

pub struct FixOpts<'a> {
    pub methods: Vec<Method>,
    /// Known-good rigged base to transfer from (e.g. wrapforge's fit).
    pub source: Option<&'a Model>,
    /// Extra candidates (e.g. UniRig output), scored like the rest.
    pub external: Vec<(String, &'a Model)>,
    /// Let regions without findings change too (default: fix only what
    /// the check flags).
    pub all_regions: bool,
}

impl Default for FixOpts<'_> {
    fn default() -> Self {
        FixOpts { methods: Method::parse("auto").unwrap(), source: None, external: Vec::new(), all_regions: false }
    }
}

pub struct Candidate {
    pub name: String,
    pub w: Weights,
    pub region_e: Vec<f64>,
}

#[derive(Serialize)]
pub struct RegionChoice {
    pub name: String,
    pub label: String,
    pub before: f64,
    pub after: f64,
    pub chosen: String,
    pub candidates: BTreeMap<String, f64>,
}

#[derive(Serialize)]
pub struct Summary {
    pub score: f64,
    pub pass: bool,
    pub fails: usize,
}

#[derive(Serialize)]
pub struct FixReport {
    pub tool: String,
    pub version: String,
    pub input: String,
    pub methods: Vec<String>,
    pub before: Summary,
    pub after: Summary,
    pub improved: bool,
    pub verts_changed: usize,
    pub mean_l1_change: f64,
    pub max_influences: usize,
    pub candidates: BTreeMap<String, f64>,
    pub regions: Vec<RegionChoice>,
    pub findings_before: Vec<crate::report::Finding>,
    pub findings_after: Vec<crate::report::Finding>,
}

pub struct FixResult {
    pub weights: Weights,
    pub eval: Eval,
    pub report_before: Report,
    pub report_after: Report,
    pub fix: FixReport,
}

// ------------------------------------------------------------------ helpers

/// Weight Laplacian smoothing restricted to `mask`ed vertices.
pub fn smooth(model: &Model, w: &Weights, mask: &[bool], iters: usize, lambda: f64) -> Weights {
    let mut cur = w.clone();
    for _ in 0..iters {
        cur = (0..model.nverts())
            .into_par_iter()
            .map(|v| {
                if !mask[v] || model.adj[v].is_empty() {
                    return cur[v].clone();
                }
                let k = model.adj[v].len() as f64;
                let mut acc: VW = cur[v].iter().map(|&(j, x)| (j, x * (1.0 - lambda))).collect();
                for &u in &model.adj[v] {
                    acc.extend(cur[u as usize].iter().map(|&(j, x)| (j, x * lambda / k)));
                }
                normalize_vw(acc)
            })
            .collect();
    }
    cur
}

pub fn dilate(model: &Model, mask: &[bool], rings: usize) -> Vec<bool> {
    let mut m = mask.to_vec();
    for _ in 0..rings {
        let prev = m.clone();
        for v in 0..model.nverts() {
            if !prev[v] && model.adj[v].iter().any(|&u| prev[u as usize]) {
                m[v] = true;
            }
        }
    }
    m
}

/// Keeps only connected flagged clusters of at least `min` vertices.
fn drop_small_clusters(model: &Model, mask: &[bool], min: usize) -> Vec<bool> {
    let n = model.nverts();
    let mut out = vec![false; n];
    let mut seen = vec![false; n];
    for s in 0..n {
        if !mask[s] || seen[s] {
            continue;
        }
        let mut comp = vec![s];
        seen[s] = true;
        let mut i = 0;
        while i < comp.len() {
            for &u in &model.adj[comp[i]] {
                let u = u as usize;
                if mask[u] && !seen[u] {
                    seen[u] = true;
                    comp.push(u);
                }
            }
            i += 1;
        }
        // A failing piece is flagged whole regardless of size.
        if comp.len() >= min || is_piece(model, s) {
            for v in comp {
                out[v] = true;
            }
        }
    }
    out
}

fn prune_all(w: &Weights) -> Weights {
    w.par_iter().map(|vw| prune_vw(vw, 4, 0.01)).collect()
}

fn is_piece(model: &Model, v: usize) -> bool {
    model.parts[model.vpart[v] as usize].piece
}

/// Pieces hang from where they attach: each piece follows the bones at
/// its attached vertices (close to the body, facing the same way) and
/// their parent chain, never sibling limbs. Piece vertices take
/// nearest-surface weights from body surface dominated by that chain
/// (a cape follows the spine down to the hips, not the arms or thighs).
/// A piece with no attached vertices falls back to robust transfer.
fn pieces_from_body(model: &Model, ctx: &Ctx, w: &Weights) -> Weights {
    let mut out = w.clone();
    if ctx.piece_transfer.is_empty() {
        return out;
    }
    let robust = inpaint_from_matches(&model.adj, &ctx.piece_transfer, w);
    let sk = &model.skel;
    for (pi, part) in model.parts.iter().enumerate() {
        if !part.piece {
            continue;
        }
        let verts: Vec<usize> = (0..model.nverts()).filter(|&v| model.vpart[v] as usize == pi).collect();
        let mut allowed = vec![false; model.njoints()];
        for m in ctx.piece_matches.iter().filter(|m| m.3 && model.vpart[m.0 as usize] as usize == pi) {
            for &(j, x) in &crate::transfer::blend(w, m.1, m.2) {
                if x > 0.1 {
                    let mut k = Some(j as usize);
                    while let Some(b) = k {
                        if allowed[b] {
                            break;
                        }
                        allowed[b] = true;
                        k = sk.jparent[b];
                    }
                }
            }
        }
        let chain_tris: Vec<[u32; 3]> = model
            .tris
            .iter()
            .filter(|t| t.iter().all(|&i| !is_piece(model, i as usize) && w[i as usize].first().is_some_and(|e| allowed[e.0 as usize])))
            .copied()
            .collect();
        if chain_tris.is_empty() {
            for &v in &verts {
                out[v] = robust[v].clone();
            }
            continue;
        }
        let index = TriIndex::new(&model.rest, chain_tris);
        for &v in &verts {
            out[v] = match index.closest(model.rest[v]) {
                Some(h) => crate::transfer::blend(w, index.tri(h.tri), h.bary),
                None => robust[v].clone(),
            };
        }
    }
    let piece_mask: Vec<bool> = (0..model.nverts()).map(|v| is_piece(model, v)).collect();
    smooth(model, &out, &piece_mask, 2, 0.5)
}

/// Bones that carry weight anywhere in the input (the rig's deform set).
fn deform_bones(model: &Model) -> Vec<bool> {
    let mut used = vec![false; model.njoints()];
    for vw in &model.weights {
        for &(j, x) in vw {
            if x > 0.01 {
                used[j as usize] = true;
            }
        }
    }
    if !used.iter().any(|&u| u) {
        used.iter_mut().for_each(|u| *u = true);
    }
    used
}

// --------------------------------------------------------------- candidates

/// Method 1, clean-up: drop bleed influences (bones far through the body),
/// treat speckled vertices as unknown and inpaint them from their clean
/// neighbors (a median-like repair that does not smear noise around),
/// smooth only where the check saw tears or collapse, prune to 4.
pub fn cleanup(model: &Model, ctx: &Ctx, ev: &Eval, smooth_tears: bool) -> Weights {
    use crate::metrics::{F_INTERSECT, F_NOISE};
    let th = &ctx.th;
    let w: Weights = (0..model.nverts())
        .into_par_iter()
        .map(|v| {
            let vw = &model.weights[v];
            let Some(near) = ctx.geo.nearest(v) else { return vw.clone() };
            let kept: VW =
                vw.iter().copied().filter(|&(j, _)| ((ctx.geo.dist(v, j) - near.1) as f64) <= th.bleed_excess * model.scale).collect();
            let kept = if kept.is_empty() { vec![(near.0, 1.0)] } else { kept };
            prune_vw(&normalize_vw(kept), 4, 0.01)
        })
        .collect();
    // Speckle is impulse noise: inside noisy areas, a vertex far from the
    // componentwise median of its neighbors takes that median (two passes,
    // so a speckle next to a speckle is caught too). Clean vertices, steep
    // falloffs, and hard steps all sit on their neighbors' median.
    let noisy_area = dilate(model, &ev.flags.iter().map(|&f| f & F_NOISE != 0).collect::<Vec<_>>(), 2);
    // Worst outliers first and in sequence, so each median sees neighbors
    // that are already cleaned (a plain parallel median breaks down when
    // half the area is speckled).
    let median_of = |w: &Weights, v: usize| -> VW {
        let n = &model.adj[v];
        let mut bones: Vec<u32> = n.iter().flat_map(|&u| w[u as usize].iter().map(|e| e.0)).collect();
        bones.sort_unstable();
        bones.dedup();
        normalize_vw(
            bones
                .iter()
                .map(|&j| {
                    let mut xs: Vec<f64> = n.iter().map(|&u| weight_of(&w[u as usize], j)).collect();
                    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let k = xs.len();
                    (j, if k % 2 == 1 { xs[k / 2] } else { 0.5 * (xs[k / 2 - 1] + xs[k / 2]) })
                })
                .collect(),
        )
    };
    let mut w = w;
    let thr = 0.4 * th.noise;
    for _ in 0..3 {
        let mut order: Vec<(f64, usize)> = (0..model.nverts())
            .filter(|&v| noisy_area[v] && model.adj[v].len() >= 3)
            .map(|v| (l1(&w[v], &median_of(&w, v)), v))
            .filter(|x| x.0 > thr)
            .collect();
        if order.is_empty() {
            break;
        }
        order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        for (_, v) in order {
            let med = median_of(&w, v);
            if !med.is_empty() && l1(&w[v], &med) > thr {
                w[v] = prune_vw(&med, 4, 0.01);
            }
        }
    }
    if !smooth_tears {
        return prune_all(&w);
    }
    let torn: Vec<bool> = ev.flags.iter().map(|&f| f & !(F_INTERSECT | F_NOISE) != 0).collect();
    let mask = dilate(model, &torn, 3);
    prune_all(&smooth(model, &w, &mask, 12, 0.5))
}

/// Method 2, geodesic voxel binding (Dionne & de Lasa 2013): weight falls
/// off with geodesic distance through the voxelized body,
/// `w = (1 / ((1 - a) d + a d^2))^2`, top 4, then light smoothing.
/// Pieces then get robust transfer from the new body weights.
pub fn geodesic(model: &Model, ctx: &Ctx) -> Weights {
    let deform = deform_bones(model);
    let alpha = 0.5;
    let w: Weights = (0..model.nverts())
        .into_par_iter()
        .map(|v| {
            let mut vw: VW = ctx.geo.near[v]
                .iter()
                .filter(|e| deform[e.0 as usize] && e.1 < ctx.geo.cap)
                .map(|&(j, d)| {
                    let d = (d as f64 / model.scale).max(1e-4);
                    (j, (1.0 / ((1.0 - alpha) * d + alpha * d * d)).powi(2))
                })
                .collect();
            if vw.is_empty() {
                // Unreachable through the voxels: nearest bone segment.
                let j = (0..model.njoints())
                    .filter(|&j| deform[j])
                    .min_by(|&a, &b| model.skel.seg_dist(a, model.rest[v]).partial_cmp(&model.skel.seg_dist(b, model.rest[v])).unwrap())
                    .unwrap_or(0);
                vw.push((j as u32, 1.0));
            }
            prune_vw(&normalize_vw(vw), 4, 0.01)
        })
        .collect();
    let body: Vec<bool> = (0..model.nverts()).map(|v| !is_piece(model, v)).collect();
    let w = prune_all(&smooth(model, &w, &body, 4, 0.5));
    prune_all(&pieces_from_body(model, ctx, &w))
}

/// Method 2b, joint bands: each body vertex belongs to its geodesically
/// nearest bone, and blends with the connected parent or child bone over
/// a band one limb radius wide on each side of the joint (smoothstep).
/// Restores a standard falloff where the input's is missing, too hard, or
/// far too wide. Pieces get robust transfer from the new body weights.
pub fn joint_bands(model: &Model, ctx: &Ctx) -> Weights {
    let sk = &model.skel;
    let deform = deform_bones(model);
    let nj = model.njoints();
    let nearest = |v: usize| -> Option<u32> { ctx.geo.near[v].iter().find(|e| deform[e.0 as usize] && e.1 < ctx.geo.cap).map(|e| e.0) };
    // Median limb radius per bone.
    let mut dists: Vec<Vec<f64>> = vec![Vec::new(); nj];
    for v in 0..model.nverts() {
        if !is_piece(model, v) {
            if let Some(j) = nearest(v) {
                dists[j as usize].push(sk.seg_dist(j as usize, model.rest[v]));
            }
        }
    }
    let radius: Vec<f64> = dists
        .into_iter()
        .map(|mut d| {
            if d.is_empty() {
                return 0.0;
            }
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            d[d.len() / 2]
        })
        .collect();
    let connected = |child: usize| -> Option<usize> {
        let p = sk.jparent[child]?;
        (deform[p] && (sk.head[child] - sk.tail[p]).len() < 0.02 * model.scale).then_some(p)
    };
    let smoothstep = |x: f64| {
        let t = x.clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let w: Weights = (0..model.nverts())
        .into_par_iter()
        .map(|v| {
            if is_piece(model, v) {
                return model.weights[v].clone();
            }
            let Some(j) = nearest(v) else { return model.weights[v].clone() };
            let j = j as usize;
            let x = model.rest[v];
            let dir = (sk.tail[j] - sk.head[j]).normalized();
            // Joint at our head (with parent)?
            if let Some(p) = connected(j) {
                let bw = radius[j].max(radius[p]).max(1e-6);
                let s = (x - sk.head[j]).dot(dir);
                if s < bw {
                    let t = smoothstep((s + bw) / (2.0 * bw));
                    return normalize_vw(vec![(j as u32, t), (p as u32, 1.0 - t)]);
                }
            }
            // Joint at our tail (with the nearest connected child)?
            let child = sk.jchildren[j]
                .iter()
                .copied()
                .filter(|&c| connected(c) == Some(j))
                .min_by(|&a, &b| (sk.head[a] - x).len().partial_cmp(&(sk.head[b] - x).len()).unwrap().then(a.cmp(&b)));
            if let Some(c) = child {
                let bw = radius[j].max(radius[c]).max(1e-6);
                let cdir = (sk.tail[c] - sk.head[c]).normalized();
                let s = (x - sk.head[c]).dot(cdir);
                if s > -bw {
                    let t = smoothstep((s + bw) / (2.0 * bw));
                    return normalize_vw(vec![(c as u32, t), (j as u32, 1.0 - t)]);
                }
            }
            vec![(j as u32, 1.0)]
        })
        .collect();
    let body: Vec<bool> = (0..model.nverts()).map(|v| !is_piece(model, v)).collect();
    let w = prune_all(&smooth(model, &w, &body, 0, 0.5));
    prune_all(&pieces_from_body(model, ctx, &w))
}

/// Maps source joints onto target joints: by name, else by nearest rest
/// head within 5% of the target's size.
pub fn map_joints(src: &Model, dst: &Model) -> Vec<Option<u32>> {
    (0..src.njoints())
        .map(|j| {
            if let Some(t) = dst.skel.joint_by_name(&src.skel.names[j]) {
                return Some(t as u32);
            }
            let h = src.skel.head[j];
            let (best, d) = (0..dst.njoints())
                .map(|t| (t, (dst.skel.head[t] - h).len()))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)))?;
            (d <= 0.05 * dst.scale).then_some(best as u32)
        })
        .collect()
}

/// Method 3, transfer + inpaint (Abdrashitov et al. 2023). With a source
/// rig: body vertices take the source's weights where the surfaces match
/// (close, same facing) and are inpainted elsewhere. Pieces always get
/// robust transfer from the (new) body.
pub fn transfer(model: &Model, ctx: &Ctx, source: Option<&Model>, match_dist: f64) -> Weights {
    let mut w = model.weights.clone();
    if let Some(src) = source {
        let jm = map_joints(src, model);
        let src_w: Weights =
            src.weights.iter().map(|vw| normalize_vw(vw.iter().filter_map(|&(j, x)| jm[j as usize].map(|t| (t, x))).collect())).collect();
        let body: Vec<u32> = (0..model.nverts() as u32).filter(|&v| !is_piece(model, v as usize)).collect();
        let src_tris: Vec<[u32; 3]> = src.tris.iter().filter(|t| t.iter().all(|&i| !is_piece(src, i as usize))).copied().collect();
        let sn = crate::scene::vertex_normals(&src.rest, &src.tris);
        let tn = crate::scene::vertex_normals(&model.rest, &model.tris);
        let index = TriIndex::new(&src.rest, src_tris);
        let matches = match_subset(&index, &sn, &model.rest, &tn, &body, match_dist * model.scale, 35f64.to_radians().cos());
        let r = inpaint_from_matches(&model.adj, &matches, &src_w);
        for m in &matches {
            w[m.0 as usize] = r[m.0 as usize].clone();
        }
    }
    prune_all(&pieces_from_body(model, ctx, &w))
}

/// Euclidean projection of v onto the probability simplex.
fn project_simplex(v: &mut [f64]) {
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut css = 0.0;
    let mut theta = 0.0;
    for (i, &ui) in u.iter().enumerate() {
        css += ui;
        let t = (css - 1.0) / (i + 1) as f64;
        if ui - t > 0.0 {
            theta = t;
        }
    }
    for x in v.iter_mut() {
        *x = (*x - theta).max(0.0);
    }
}

/// Method 4, optimize: per vertex, fit linear-blend weights to the dual
/// quaternion deformation of the start weights over the whole pose set
/// (DQS does not collapse at bends or twists), with smoothness and
/// stay-close terms, weights on the simplex, bones limited to the start
/// set plus geodesically near ones. Only flagged areas move.
pub fn optimize(model: &Model, ctx: &Ctx, start: &Weights, ev: &Eval) -> Weights {
    let flagged: Vec<bool> = ev.flags.iter().map(|&f| f & !crate::metrics::F_INTERSECT != 0).collect();
    let active = dilate(model, &flagged, 4);
    let targets: Vec<Vec<Vec3>> = ctx.mats.par_iter().map(|m| dqs(model, m, start)).collect();
    let np = ctx.poses.len().max(1) as f64;
    let s2 = model.scale * model.scale;
    let (ls, l0) = (0.05, 0.01);
    let mut cur = start.clone();
    for _sweep in 0..4 {
        let prev = cur.clone();
        cur = (0..model.nverts())
            .into_par_iter()
            .map(|v| {
                if !active[v] || is_piece(model, v) {
                    return prev[v].clone();
                }
                // Candidate bones.
                let mut bones: Vec<u32> = start[v].iter().map(|e| e.0).collect();
                if let Some(near) = ctx.geo.nearest(v) {
                    for &(j, d) in ctx.geo.near[v].iter().take(4) {
                        if ((d - near.1) as f64) < ctx.th.bleed_excess * model.scale * 0.5 && !bones.contains(&j) {
                            bones.push(j);
                        }
                    }
                }
                bones.truncate(6);
                let k = bones.len();
                if k <= 1 {
                    return prev[v].clone();
                }
                let x = model.rest[v];
                // H = sum_p A_p^T A_p / (P s^2) + (ls + l0) I ; g likewise.
                let mut h = vec![0.0; k * k];
                let mut g = vec![0.0; k];
                for (p, m) in ctx.mats.iter().enumerate() {
                    let cols: Vec<Vec3> = bones.iter().map(|&j| m[j as usize].transform_point(x)).collect();
                    let t = targets[p][v];
                    for a in 0..k {
                        g[a] += cols[a].dot(t) / (np * s2);
                        for b in 0..k {
                            h[a * k + b] += cols[a].dot(cols[b]) / (np * s2);
                        }
                    }
                }
                let n = &model.adj[v];
                let kn = n.len().max(1) as f64;
                for (a, &j) in bones.iter().enumerate() {
                    let mean = n.iter().map(|&u| weight_of(&prev[u as usize], j)).sum::<f64>() / kn;
                    g[a] += ls * mean + l0 * weight_of(&start[v], j);
                    h[a * k + a] += ls + l0;
                }
                let lip: f64 = (0..k).map(|a| h[a * k + a]).sum::<f64>().max(1e-12);
                let mut wv: Vec<f64> = bones.iter().map(|&j| weight_of(&prev[v], j)).collect();
                project_simplex(&mut wv);
                for _ in 0..200 {
                    let grad: Vec<f64> = (0..k).map(|a| (0..k).map(|b| h[a * k + b] * wv[b]).sum::<f64>() - g[a]).collect();
                    for a in 0..k {
                        wv[a] -= grad[a] / lip;
                    }
                    project_simplex(&mut wv);
                }
                let mut out: VW = bones.iter().zip(&wv).map(|(&j, &x)| (j, x)).collect();
                sort_vw(&mut out);
                prune_vw(&out, 4, 0.01)
            })
            .collect();
    }
    prune_all(&cur)
}

/// External candidate (e.g. UniRig output) on a possibly different mesh
/// and skeleton: joints mapped by name or head position, weights moved by
/// robust transfer.
pub fn external(model: &Model, ctx: &Ctx, other: &Model, match_dist: f64) -> Weights {
    transfer(model, ctx, Some(other), match_dist)
}

// ---------------------------------------------------------------- picking

fn l1(a: &VW, b: &VW) -> f64 {
    a.iter().map(|&(j, x)| (x - weight_of(b, j)).abs()).sum::<f64>()
        + b.iter().filter(|e| weight_of(a, e.0) == 0.0).map(|e| e.1).sum::<f64>()
}

/// Blends per-region candidate choices with a few rings of smoothing at
/// region seams.
fn blend(model: &Model, ctx: &Ctx, cands: &[Candidate], choice: &[usize], active: &[bool], rings: usize) -> Weights {
    let nv = model.nverts();
    let nc = cands.len();
    let mut alpha: Vec<Vec<f64>> = (0..nv)
        .map(|v| {
            let mut a = vec![0.0; nc];
            a[if active[v] { choice[ctx.vregion[v] as usize] } else { 0 }] = 1.0;
            a
        })
        .collect();
    for _ in 0..rings {
        alpha = (0..nv)
            .map(|v| {
                if model.adj[v].is_empty() {
                    return alpha[v].clone();
                }
                let k = model.adj[v].len() as f64;
                (0..nc).map(|c| 0.5 * alpha[v][c] + 0.5 * model.adj[v].iter().map(|&u| alpha[u as usize][c]).sum::<f64>() / k).collect()
            })
            .collect();
    }
    (0..nv)
        .into_par_iter()
        .map(|v| {
            if alpha[v][0] > 1.0 - 1e-9 {
                return cands[0].w[v].clone(); // untouched: byte-identical
            }
            let mut acc: VW = Vec::new();
            for c in 0..nc {
                if alpha[v][c] > 1e-6 {
                    acc.extend(cands[c].w[v].iter().map(|&(j, x)| (j, x * alpha[v][c])));
                }
            }
            prune_vw(&normalize_vw(acc), 4, 0.01)
        })
        .collect()
}

fn summary(r: &Report) -> Summary {
    Summary { score: r.score, pass: r.pass, fails: r.findings.iter().filter(|f| f.severity == "fail").count() }
}

pub fn fix(file: &str, model: &Model, ctx: &Ctx, o: &FixOpts) -> FixResult {
    let ev0 = evaluate(model, ctx, &model.weights);
    let rep0 = build(file, model, ctx, &ev0);
    let score = |w: &Weights| -> (Eval, Vec<f64>) {
        let ev = evaluate_with(model, ctx, w, false);
        let re = region_energy(ctx, &ev);
        (ev, re)
    };
    // Regions eligible for change: those with failing findings.
    let nr = ctx.regions.len();
    let failing: Vec<bool> =
        ctx.regions.iter().map(|r| o.all_regions || rep0.findings.iter().any(|f| f.severity == "fail" && f.region == r.name)).collect();
    // Only the reported bad vertices (plus a margin) change; a failing
    // piece changes as a whole. The rest of a region keeps its weights.
    let mask = crate::report::finding_mask(ctx, &ev0, &rep0);
    let flagged: Vec<bool> = (0..model.nverts())
        .map(|v| o.all_regions || mask[v] & !crate::metrics::F_INTERSECT != 0 || (is_piece(model, v) && failing[ctx.vregion[v] as usize]))
        .collect();
    // Isolated flags (a stray vertex or two) do not open an area for change.
    let flagged = drop_small_clusters(model, &flagged, ctx.th.region_min_bad);
    let active = dilate(model, &flagged, 6);
    let mut cands: Vec<Candidate> = Vec::new();
    // Candidates are scored as they will be applied: only on the active
    // area (seam-blended), the input everywhere else.
    let push = |name: String, w: Weights, cands: &mut Vec<Candidate>| {
        let local = if cands.is_empty() {
            w.clone()
        } else {
            let pair = [
                Candidate { name: String::new(), w: model.weights.clone(), region_e: Vec::new() },
                Candidate { name: String::new(), w: w.clone(), region_e: Vec::new() },
            ];
            blend(model, ctx, &pair, &vec![1; nr], &active, 3)
        };
        let (_, region_e) = score(&local);
        cands.push(Candidate { name, w, region_e });
    };
    push("original".into(), model.weights.clone(), &mut cands);
    let match_dist = 0.03;
    for &m in &o.methods {
        match m {
            Method::Smooth => {
                push("despeckle".into(), cleanup(model, ctx, &ev0, false), &mut cands);
                push("smooth".into(), cleanup(model, ctx, &ev0, true), &mut cands);
            }
            Method::Geodesic => {
                push("geodesic".into(), geodesic(model, ctx), &mut cands);
                push("geodesic-band".into(), joint_bands(model, ctx), &mut cands);
            }
            Method::Transfer => push(
                if o.source.is_some() { "transfer".into() } else { "transfer(pieces)".into() },
                transfer(model, ctx, o.source, match_dist),
                &mut cands,
            ),
            Method::Optimize => {}
        }
    }
    for (name, other) in &o.external {
        push(format!("external:{name}"), external(model, ctx, other, match_dist), &mut cands);
    }

    let orig_e = cands[0].region_e.clone();
    let mut choice = vec![0usize; nr];
    let mut w = model.weights.clone();
    let mut cur_e = orig_e.clone();

    // Greedy, worst region first: take the best-scoring candidate for that
    // region whose blended result lowers the total and leaves no region
    // worse than the input.
    // Region adjacency (regions sharing a mesh edge).
    let mut adjr = vec![std::collections::BTreeSet::new(); nr];
    for e in &model.edges {
        let (a, b) = (ctx.vregion[e[0] as usize] as usize, ctx.vregion[e[1] as usize] as usize);
        if a != b {
            adjr[a].insert(b);
            adjr[b].insert(a);
        }
    }
    // A region may get at most this much worse than the input while its
    // neighbor is repaired (seams); the total must still drop.
    let slack = std::cell::Cell::new(0.01);
    let tolerated = |k: usize| orig_e[k] * 1.1 + slack.get();
    let greedy = |cands: &[Candidate], choice: &mut Vec<usize>, w: &mut Weights, cur_e: &mut Vec<f64>| {
        let change: Vec<Vec<f64>> = cands
            .iter()
            .map(|c| {
                let mut sum = vec![0.0; nr];
                let mut cnt = vec![0usize; nr];
                for v in 0..model.nverts() {
                    if active[v] {
                        let r = ctx.vregion[v] as usize;
                        sum[r] += l1(&c.w[v], &model.weights[v]);
                        cnt[r] += 1;
                    }
                }
                sum.iter().zip(&cnt).map(|(s, &n)| s / n.max(1) as f64).collect()
            })
            .collect();
        let mut order: Vec<usize> = (0..nr).filter(|&r| failing[r]).collect();
        order.sort_by(|&a, &b| cur_e[b].partial_cmp(&cur_e[a]).unwrap().then(a.cmp(&b)));
        for r in order {
            let mut opts: Vec<usize> = (1..cands.len()).filter(|&c| c != choice[r] && cands[c].region_e[r] < cur_e[r] - 1e-9).collect();
            // Smallest edit first among candidates about as good as the
            // best; the rest by score.
            let best = opts.iter().map(|&c| cands[c].region_e[r]).fold(f64::MAX, f64::min);
            let good = |c: usize| cands[c].region_e[r] <= best + (0.25 * best).max(0.01);
            opts.sort_by(|&a, &b| {
                (!good(a), if good(a) { change[a][r] } else { cands[a].region_e[r] })
                    .partial_cmp(&(!good(b), if good(b) { change[b][r] } else { cands[b].region_e[r] }))
                    .unwrap()
                    .then(a.cmp(&b))
            });
            // The region alone, with its failing neighbors, and with all
            // neighbors (a fault across a joint spans two regions; only
            // active vertices near the flags ever change).
            let near_fail: Vec<usize> = std::iter::once(r).chain(adjr[r].iter().copied().filter(|&k| failing[k])).collect();
            let near_all: Vec<usize> = std::iter::once(r).chain(adjr[r].iter().copied()).collect();
            for c in opts {
                let mut sets: Vec<&[usize]> = vec![&near_fail[..1]];
                if near_fail.len() > 1 {
                    sets.push(&near_fail[..]);
                }
                if near_all.len() > near_fail.len() {
                    sets.push(&near_all[..]);
                }
                // Best valid region set for this candidate.
                let mut best: Option<(f64, Vec<usize>, Weights, Vec<f64>)> = None;
                for members in sets {
                    let mut trial = choice.clone();
                    for &k in members {
                        trial[k] = c;
                    }
                    let wt = blend(model, ctx, cands, &trial, &active, 3);
                    let (_, re) = score(&wt);
                    let total: f64 = re.iter().sum();
                    let total_ok = total < cur_e.iter().sum::<f64>() - 1e-9;
                    let none_worse = (0..nr).all(|k| re[k] <= tolerated(k));
                    if std::env::var_os("WF_DEBUG").is_some() {
                        eprintln!(
                            "try {} <- {} ({} regions): total {:.4} -> {total:.4}, region {:.4} -> {:.4}",
                            ctx.regions[r].name,
                            cands[c].name,
                            members.len(),
                            cur_e.iter().sum::<f64>(),
                            cur_e[r],
                            re[r]
                        );
                    }
                    // Only real repairs, no polishing of what already works.
                    let meaningful = re[r] < cur_e[r] - (0.2 * cur_e[r]).max(0.005);
                    if total_ok && none_worse && meaningful && best.as_ref().is_none_or(|b| total < b.0) {
                        best = Some((total, trial, wt, re));
                    }
                }
                if let Some((_, trial, wt, re)) = best {
                    *choice = trial;
                    *w = wt;
                    *cur_e = re;
                    break;
                }
            }
        }
    };
    greedy(&cands, &mut choice, &mut w, &mut cur_e);

    // Method 4 starts from the best mix so far, then gets its own pass.
    if o.methods.contains(&Method::Optimize) {
        let (ev_mix, _) = score(&w);
        let opt = optimize(model, ctx, &w, &ev_mix);
        push("optimize".into(), opt, &mut cands);
        // Re-anchor: blends now combine the chosen candidates with optimize.
        greedy(&cands, &mut choice, &mut w, &mut cur_e);
    }

    let mut ev = evaluate(model, ctx, &w);
    let mut rep1 = build(file, model, ctx, &ev);
    // Hard rule: no region may gain a failing finding it did not have.
    let new_fails = |rep: &Report| {
        rep.findings.iter().any(|f| f.severity == "fail" && !rep0.findings.iter().any(|g| g.severity == "fail" && g.region == f.region))
    };
    if new_fails(&rep1) {
        slack.set(0.002);
        choice = vec![0; nr];
        w = model.weights.clone();
        cur_e = orig_e.clone();
        greedy(&cands, &mut choice, &mut w, &mut cur_e);
        ev = evaluate(model, ctx, &w);
        rep1 = build(file, model, ctx, &ev);
    }
    if rep1.score < rep0.score || new_fails(&rep1) {
        w = model.weights.clone();
        ev = ev0;
        rep1 = build(file, model, ctx, &ev);
        choice = vec![0; nr];
    }

    let re1 = region_energy(ctx, &ev);
    let regions = (0..nr)
        .filter(|&r| failing[r] || choice[r] != 0)
        .map(|r| RegionChoice {
            name: ctx.regions[r].name.clone(),
            label: ctx.regions[r].label.clone(),
            before: crate::report::energy_to_score(orig_e[r]),
            after: crate::report::energy_to_score(re1[r]),
            chosen: cands[choice[r]].name.clone(),
            candidates: cands.iter().map(|c| (c.name.clone(), crate::report::energy_to_score(c.region_e[r]))).collect(),
        })
        .collect();
    let changes: Vec<f64> = (0..model.nverts()).map(|v| l1(&w[v], &model.weights[v])).collect();
    let fix = FixReport {
        tool: "weightforge".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        input: file.into(),
        methods: o.methods.iter().map(|m| m.name().to_string()).chain(o.external.iter().map(|e| format!("external:{}", e.0))).collect(),
        before: summary(&rep0),
        after: summary(&rep1),
        improved: rep1.score > rep0.score,
        verts_changed: changes.iter().filter(|&&c| c > 1e-4).count(),
        mean_l1_change: (changes.iter().sum::<f64>() / model.nverts().max(1) as f64 * 1e6).round() / 1e6,
        max_influences: w.iter().map(Vec::len).max().unwrap_or(0),
        candidates: cands
            .iter()
            .map(|c| {
                let mean: f64 = c.region_e.iter().sum::<f64>() / nr.max(1) as f64;
                (c.name.clone(), crate::report::energy_to_score(mean))
            })
            .collect(),
        regions,
        findings_before: rep0.findings.clone(),
        findings_after: rep1.findings.clone(),
    };
    FixResult { weights: w, eval: ev, report_before: rep0, report_after: rep1, fix }
}

/// Mean L1 distance (halved, so 0..1 per vertex) between two weight sets.
pub fn weight_error(a: &Weights, b: &Weights) -> f64 {
    a.iter().zip(b).map(|(x, y)| 0.5 * l1(x, y)).sum::<f64>() / a.len().max(1) as f64
}
