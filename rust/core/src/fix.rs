//! Weight repair. Builds candidate weight sets, scores each on the same pose
//! set, keeps the best candidate per broken region, blends at region seams,
//! and never returns anything that scores worse than the input in any
//! region (the input is always candidate 0).

use crate::math::Vec3;
use crate::metrics::{Ctx, Eval, evaluate, evaluate_partial, evaluate_with};
use crate::report::{Report, build, region_energy, region_energy_of};
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
    /// `auto`, one method, or a comma-separated list (`smooth,geodesic`).
    pub fn parse(s: &str) -> Option<Vec<Method>> {
        if s.contains(',') {
            let mut out = Vec::new();
            for part in s.split(',') {
                for m in Method::parse(part.trim())? {
                    if !out.contains(&m) {
                        out.push(m);
                    }
                }
            }
            return Some(out);
        }
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
    /// Extra candidates (e.g. SkinTokens output), scored like the rest.
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
    /// The region pick searched every plan (false: it hit its node budget
    /// and kept the best plan found; only on rigs with many failing regions).
    pub pick_exact: bool,
    /// False when the run without the external candidates scored better
    /// and was kept (`fix`).
    pub external_kept: bool,
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

/// Repair band widths, in feature sizes (`Ctx::feature`, the median limb
/// radius), never in edge rings: a ring-count band is half as wide on a
/// mesh twice as dense, and tears there. Calibrated on the test mannequin
/// the ring counts were tuned on (P2; mean edge 0.47 feature sizes, cape
/// edge 0.8), so there each band is as wide as before (old ring counts in
/// brackets).
pub mod widths {
    /// Margin around flagged vertices that a fix may change (6 rings).
    pub const MARGIN: f64 = 2.8;
    /// Seam blend between candidates (and the input): the Gaussian width
    /// of `blend` (what 3 smoothing passes spread on the test mannequin).
    pub const SEAM: f64 = 0.4;
    /// Clean-up: margin around tears and collapses (3 rings) and its
    /// smoothing width (12 passes).
    pub const TORN_MARGIN: f64 = 1.4;
    pub const TORN_SMOOTH: f64 = 0.8;
    /// Geodesic binding's smoothing (4 passes).
    pub const GEODESIC_SMOOTH: f64 = 0.5;
    /// Piece (cape, skirt) smoothing after transfer (2 passes).
    pub const PIECE_SMOOTH: f64 = 0.57;
    /// Margin that optimize may change around flagged vertices (4 rings).
    pub const OPTIMIZE_MARGIN: f64 = 1.9;
}

/// Vertices within rest-pose edge-path distance `dist` of `seeds`,
/// travelling only through `within` (everywhere when `None`). The metric
/// counterpart of `dilate`.
pub fn grow(model: &Model, seeds: &[bool], dist: f64, within: Option<&[bool]>) -> Vec<bool> {
    distances_within(model, seeds, dist, within).iter().map(|x| x.is_finite()).collect()
}

/// Rest-pose edge-path distance from `seeds`, infinite beyond `max`.
pub fn distances(model: &Model, seeds: &[bool], max: f64) -> Vec<f64> {
    distances_within(model, seeds, max, None)
}

fn distances_within(model: &Model, seeds: &[bool], max: f64, within: Option<&[bool]>) -> Vec<f64> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let n = model.nverts();
    let mut d = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();
    for v in 0..n {
        if seeds[v] {
            d[v] = 0.0;
            heap.push(Reverse((0u64, v)));
        }
    }
    // f64 >= 0 orders like its bit pattern.
    while let Some(Reverse((bits, v))) = heap.pop() {
        let dv = f64::from_bits(bits);
        if dv > d[v] {
            continue;
        }
        for &u in &model.adj[v] {
            let u = u as usize;
            if within.is_some_and(|m| !m[u]) {
                continue;
            }
            let du = dv + (model.rest[u] - model.rest[v]).len();
            if du < d[u] && du <= max {
                d[u] = du;
                heap.push(Reverse((du.to_bits(), u)));
            }
        }
    }
    d
}

/// Smoothing passes (`smooth` with lambda 0.5) that spread a weight like a
/// Gaussian of width `sigma`: each pass moves it a quarter of a squared edge
/// on average, so n passes spread `h * sqrt(n) / 2` (h = mean rest edge
/// length among `mask`ed vertices).
pub fn passes(model: &Model, mask: &[bool], sigma: f64) -> usize {
    let (mut sum, mut cnt) = (0.0, 0usize);
    for e in &model.edges {
        if mask[e[0] as usize] && mask[e[1] as usize] {
            sum += (model.rest[e[0] as usize] - model.rest[e[1] as usize]).len();
            cnt += 1;
        }
    }
    if cnt == 0 || sum <= 0.0 {
        return 0;
    }
    let h = sum / cnt as f64;
    ((4.0 * sigma * sigma / (h * h)).round() as usize).min(2000)
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
    let n = passes(model, &piece_mask, widths::PIECE_SMOOTH * ctx.feature);
    smooth(model, &out, &piece_mask, n, 0.5)
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
    let mask = grow(model, &torn, widths::TORN_MARGIN * ctx.feature, None);
    let n = passes(model, &mask, widths::TORN_SMOOTH * ctx.feature);
    prune_all(&smooth(model, &w, &mask, n, 0.5))
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
    let n = passes(model, &body, widths::GEODESIC_SMOOTH * ctx.feature);
    let w = prune_all(&smooth(model, &w, &body, n, 0.5));
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
    let active = grow(model, &flagged, widths::OPTIMIZE_MARGIN * ctx.feature, None);
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

/// External candidate (e.g. SkinTokens output) on a possibly different mesh
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

/// Standard normal CDF (Abramowitz and Stegun 7.1.26, error < 2e-7).
fn phi(x: f64) -> f64 {
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = 1.0 - poly * (-z * z).exp();
    if x >= 0.0 { 0.5 * (1.0 + erf) } else { 0.5 * (1.0 - erf) }
}

/// Blends per-region candidate choices across their seams like a Gaussian
/// blur of width `sigma` (metres), at any mesh density: a vertex keeps
/// `Phi(s / sigma)` of its own candidate, `s` being its rest-pose edge-path
/// distance to the nearest vertex assigned otherwise (less half an edge,
/// where the seam runs), and shares the rest among those other candidates
/// by the same rule. One bounded Dijkstra per candidate in use; vertices
/// more than 3 sigma from every seam keep their candidate's weights exactly.
fn blend(model: &Model, ctx: &Ctx, cands: &[Candidate], choice: &[usize], active: &[bool], sigma: f64) -> Weights {
    let nv = model.nverts();
    let assign: Vec<usize> = (0..nv).map(|v| if active[v] { choice[ctx.vregion[v] as usize] } else { 0 }).collect();
    let mut used: Vec<usize> = assign.clone();
    used.sort_unstable();
    used.dedup();
    if used == [0] {
        return cands[0].w.clone();
    }
    let reach = 3.0 * sigma;
    let dist: Vec<Vec<f64>> =
        used.par_iter().map(|&c| distances(model, &assign.iter().map(|&a| a == c).collect::<Vec<_>>(), reach)).collect();
    (0..nv)
        .into_par_iter()
        .map(|v| {
            let own = assign[v];
            let half = 0.5 * model.adj[v].iter().map(|&u| (model.rest[u as usize] - model.rest[v]).len()).fold(f64::MAX, f64::min);
            let half = if half.is_finite() { half } else { 0.0 };
            // Other candidates within reach and how much of each spills here.
            let mut others: Vec<(usize, f64)> = Vec::new();
            let mut nearest = f64::INFINITY;
            for (k, &c) in used.iter().enumerate() {
                let d = dist[k][v];
                if c != own && d.is_finite() {
                    nearest = nearest.min(d);
                    let x = if sigma > 0.0 { 1.0 - phi((d - half).max(0.0) / sigma) } else { 0.0 };
                    if x > 1e-6 {
                        others.push((c, x));
                    }
                }
            }
            if others.is_empty() {
                return cands[own].w[v].clone(); // away from seams: exact
            }
            let keep = if sigma > 0.0 { phi((nearest - half).max(0.0) / sigma) } else { 1.0 };
            let total: f64 = others.iter().map(|o| o.1).sum();
            let mut acc: VW = cands[own].w[v].iter().map(|&(j, y)| (j, y * keep)).collect();
            for &(c, x) in &others {
                acc.extend(cands[c].w[v].iter().map(|&(j, y)| (j, y * (1.0 - keep) * x / total)));
            }
            prune_vw(&normalize_vw(acc), 4, 0.01)
        })
        .collect()
}

/// Flagged-vertex counts per region and flag (`pick::NFLAG` slots in
/// `FLAG_NAMES` order; self-intersection, a warning, is not counted).
fn flag_counts(ctx: &Ctx, flags: &[u8], regions: Option<&[bool]>) -> Vec<[i64; crate::pick::NFLAG]> {
    let mut out = vec![[0i64; crate::pick::NFLAG]; ctx.regions.len()];
    for (v, &f) in flags.iter().enumerate() {
        let r = ctx.vregion[v] as usize;
        if f == 0 || regions.is_some_and(|d| !d[r]) {
            continue;
        }
        for (i, (bit, _)) in crate::metrics::FLAG_NAMES.iter().enumerate() {
            if f & bit != 0 && *bit != crate::metrics::F_INTERSECT {
                out[r][i] += 1;
            }
        }
    }
    out
}

/// Per region and flag: the flagged count at which `report::build` makes
/// it a failing finding, and the highest count allowed (one less, for
/// regions that do not fail yet; unlimited where the region already fails).
type Counts = Vec<[i64; crate::pick::NFLAG]>;
fn fail_limits(model: &Model, ctx: &Ctx, had_fail: &[bool]) -> (Counts, Counts) {
    use crate::metrics::{F_FOLLOW, F_INTERSECT, F_PIECE, F_UNWEIGHTED, FLAG_NAMES};
    let mut attached = vec![0i64; ctx.regions.len()];
    for m in ctx.piece_matches.iter().filter(|m| m.3) {
        attached[ctx.vregion[m.0 as usize] as usize] += 1;
    }
    let _ = model;
    let fail_at: Counts = (0..ctx.regions.len())
        .map(|r| {
            std::array::from_fn(|i| {
                let bit = FLAG_NAMES[i].0;
                let min = if bit == F_UNWEIGHTED || bit == F_PIECE { 1 } else { ctx.th.region_min_bad as i64 };
                match bit {
                    F_INTERSECT => i64::MAX,
                    // A piece straying in fewer than half its attached verts is a warning.
                    F_FOLLOW => min.max((attached[r] + 1) / 2),
                    _ => min,
                }
            })
        })
        .collect();
    let limit = (0..ctx.regions.len())
        .map(|r| if had_fail[r] { [i64::MAX; crate::pick::NFLAG] } else { fail_at[r].map(|x| if x == i64::MAX { x } else { x - 1 }) })
        .collect();
    (fail_at, limit)
}

/// Edit cost: score energy per unit of mean weight change (`pick::Problem::
/// edit_cost`). 0.01 weighs a plan that changes every vertex's weights by
/// 0.1 (L1) like 0.001 of energy, about 0.1 score points near 90: among
/// plans that score about the same, the smaller edit wins.
const EDIT_MU: f64 = 0.01;

fn summary(r: &Report) -> Summary {
    Summary { score: r.score, pass: r.pass, fails: r.findings.iter().filter(|f| f.severity == "fail").count() }
}

/// Repairs the weights (see the module docs). With external candidates
/// it also runs without them and keeps the better result (passes the gate
/// first, then score): an extra candidate never makes the fix worse than
/// not having it, even where the region pick runs out of search budget.
pub fn fix(file: &str, model: &Model, ctx: &Ctx, o: &FixOpts) -> FixResult {
    let with = fix_once(file, model, ctx, o);
    if o.external.is_empty() {
        return with;
    }
    let plain = FixOpts { methods: o.methods.clone(), source: o.source, external: Vec::new(), all_regions: o.all_regions };
    let mut without = fix_once(file, model, ctx, &plain);
    let key = |r: &FixResult| (r.report_after.pass, r.report_after.score);
    if key(&without) > key(&with) {
        without.fix.methods = with.fix.methods.clone();
        without.fix.external_kept = false;
        return without;
    }
    with
}

fn fix_once(file: &str, model: &Model, ctx: &Ctx, o: &FixOpts) -> FixResult {
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
    let active = grow(model, &flagged, widths::MARGIN * ctx.feature, None);
    let seam = widths::SEAM * ctx.feature;
    if std::env::var_os("WF_DEBUG").is_some() {
        eprintln!(
            "fix: {} verts, {} active, feature {:.4}, seam {seam:.4}",
            model.nverts(),
            active.iter().filter(|&&a| a).count(),
            ctx.feature
        );
    }
    // Incremental trial scoring: only regions touching a changed vertex (or
    // its one-ring) are re-evaluated; every other vertex keeps its cached
    // energy. Exact, because a vertex's energy depends only on its own
    // region and its neighbors.
    let score_trial = |wt: &Weights, base: &Weights, base_v: &[f64]| -> (Vec<f64>, Vec<u8>, Vec<bool>) {
        let mut dirty_r = vec![false; ctx.regions.len()];
        for v in 0..model.nverts() {
            if wt[v] != base[v] {
                dirty_r[ctx.vregion[v] as usize] = true;
                for &u in &model.adj[v] {
                    dirty_r[ctx.vregion[u as usize] as usize] = true;
                }
            }
        }
        let mask: Vec<bool> = (0..model.nverts()).map(|v| dirty_r[ctx.vregion[v] as usize]).collect();
        let part = evaluate_partial(model, ctx, wt, &mask);
        let e: Vec<f64> = (0..model.nverts()).map(|v| if mask[v] { part.energy[v] } else { base_v[v] }).collect();
        (region_energy_of(ctx, &e), part.flags, dirty_r)
    };
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
            blend(model, ctx, &pair, &vec![1; nr], &active, seam)
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
    // Twist/helper bones: generic candidates leave them unweighted (they
    // carry no weight in the input); the helper band weights them, alone
    // and refined by optimize (helpers.rs).
    if !model.skel.helpers.is_empty() {
        // How wide an armpit or groin band should be depends on the body
        // and the rig (wider trades stretch for collapse), so three widths
        // are scored and the pick keeps whichever repairs best.
        for (tag, width) in [("", 1.0), ("-narrow", 0.7), ("-wide", 1.4)] {
            let bw = crate::helpers::band(model, ctx, &model.weights, width);
            if o.methods.contains(&Method::Optimize) {
                let (ev_b, _) = score(&bw);
                push(format!("helper-band{tag}+optimize"), optimize(model, ctx, &bw, &ev_b), &mut cands);
            }
            push(format!("helper-band{tag}"), bw, &mut cands);
        }
    }
    for (name, other) in &o.external {
        push(format!("external:{name}"), external(model, ctx, other, match_dist), &mut cands);
    }

    // -------------------------------------------------------------- picking
    // Moves: one candidate on a set of regions, measured alone against the
    // input (pick.rs). The search over plans is exact, so the result never
    // depends on visiting order and a better candidate never scores worse.
    let orig_ev = evaluate_with(model, ctx, &model.weights, false);
    let orig_v = orig_ev.energy.clone();
    let orig_e = region_energy_of(ctx, &orig_v);
    let mut size = vec![0.0f64; nr];
    for &r in &ctx.vregion {
        size[r as usize] += 1.0;
    }
    let had_fail: Vec<bool> =
        ctx.regions.iter().map(|r| rep0.findings.iter().any(|f| f.severity == "fail" && f.region == r.name)).collect();
    let base_counts = flag_counts(ctx, &orig_ev.flags, None);
    let (fail_at, limit) = fail_limits(model, ctx, &had_fail);
    // Region adjacency (regions sharing a mesh edge).
    let mut adjr = vec![std::collections::BTreeSet::new(); nr];
    for e in &model.edges {
        let (a, b) = (ctx.vregion[e[0] as usize] as usize, ctx.vregion[e[1] as usize] as usize);
        if a != b {
            adjr[a].insert(b);
            adjr[b].insert(a);
        }
    }
    // Region sets a move may cover, per failing region: alone, with its
    // failing neighbors, with all neighbors (a fault across a joint spans
    // two regions), its group of failing regions (failing regions within
    // two steps of each other: both thighs across the hips), and that
    // group with all their neighbors. Only active vertices ever change.
    let mut sets: std::collections::BTreeSet<Vec<usize>> = std::collections::BTreeSet::new();
    let near_all = |r: usize| -> Vec<usize> { std::iter::once(r).chain(adjr[r].iter().copied()).collect() };
    for r in (0..nr).filter(|&r| failing[r]) {
        let near_fail: Vec<usize> = std::iter::once(r).chain(adjr[r].iter().copied().filter(|&k| failing[k])).collect();
        let mut group = vec![r];
        let mut i = 0;
        while i < group.len() {
            let around = near_all(group[i]);
            let join: Vec<usize> =
                (0..nr).filter(|&k| failing[k] && !group.contains(&k) && near_all(k).iter().any(|x| around.contains(x))).collect();
            group.extend(join);
            i += 1;
        }
        let mut wide: Vec<usize> = group.iter().flat_map(|&g| near_all(g)).collect();
        wide.sort_unstable();
        wide.dedup();
        for mut s in [vec![r], near_fail, near_all(r), group, wide] {
            s.sort_unstable();
            sets.insert(s);
        }
    }
    let debug = std::env::var_os("WF_DEBUG").is_some();
    // The state moves are measured against: the input at first, the
    // current plan in the refinement rounds.
    struct Base {
        choice: Vec<usize>,
        w: Weights,
        v: Vec<f64>,
        e: Vec<f64>,
        counts: Vec<[i64; crate::pick::NFLAG]>,
        change: f64,
    }
    let base_of = |choice: Vec<usize>, w: Weights| -> Base {
        let ev = evaluate_with(model, ctx, &w, false);
        let e = region_energy_of(ctx, &ev.energy);
        let counts = flag_counts(ctx, &ev.flags, None);
        let change = (0..model.nverts()).map(|v| l1(&w[v], &model.weights[v])).sum();
        Base { choice, w, v: ev.energy, e, counts, change }
    };
    let input_base = Base {
        choice: vec![0; nr],
        w: model.weights.clone(),
        v: orig_v.clone(),
        e: orig_e.clone(),
        counts: base_counts.clone(),
        change: 0.0,
    };
    let measure = |cands: &[Candidate], c: usize, base: &Base| -> Vec<crate::pick::Move> {
        sets.iter()
            .filter(|members| members.iter().any(|&k| base.choice[k] != c))
            .map(|members| {
                let mut trial = base.choice.clone();
                for &k in members {
                    trial[k] = c;
                }
                let wt = blend(model, ctx, cands, &trial, &active, seam);
                let (re, flags, dirty) = score_trial(&wt, &base.w, &base.v);
                let delta: Vec<(usize, f64)> = (0..nr).filter(|&k| dirty[k]).map(|k| (k, re[k] - base.e[k])).collect();
                let counts = flag_counts(ctx, &flags, Some(&dirty));
                let mut fl = Vec::new();
                for k in (0..nr).filter(|&k| dirty[k]) {
                    for f in 0..crate::pick::NFLAG {
                        let d = counts[k][f] - base.counts[k][f];
                        if d != 0 {
                            fl.push((k, f, d));
                        }
                    }
                }
                // Weight change vs the input added by this move (0 if it
                // brings the weights back toward the input).
                let change: f64 = (0..model.nverts()).map(|v| l1(&wt[v], &model.weights[v])).sum::<f64>() - base.change;
                crate::pick::Move { cand: c, members: members.clone(), delta, flags: fl, change: change.max(0.0) }
            })
            .collect()
    };
    // Search order: outward from the worst failing region along region
    // adjacency, so moves that conflict are decided next to each other.
    let mut by_e: Vec<usize> = (0..nr).filter(|&r| failing[r]).collect();
    by_e.sort_by(|&a, &b| orig_e[b].partial_cmp(&orig_e[a]).unwrap().then(a.cmp(&b)));
    let mut anchors: Vec<usize> = Vec::new();
    let mut seen = vec![false; nr];
    for &start in &by_e {
        if seen[start] {
            continue;
        }
        let mut queue = std::collections::VecDeque::from([start]);
        seen[start] = true;
        while let Some(r) = queue.pop_front() {
            if failing[r] {
                anchors.push(r);
            }
            for &k in &adjr[r] {
                if !seen[k] {
                    seen[k] = true;
                    queue.push_back(k);
                }
            }
        }
    }
    let problem_at = |base: &Base, moves: Vec<crate::pick::Move>, slack: f64| crate::pick::Problem {
        orig: base.e.clone(),
        size: size.clone(),
        // Rules stay relative to the input: caps on its energies, no
        // region gains a failing finding it did not have.
        cap: orig_e.iter().map(|e| e * 1.1 + slack).collect(),
        counts: base.counts.clone(),
        fail_at: fail_at.clone(),
        limit: limit.clone(),
        anchors: anchors.clone(),
        moves,
        gate_first: true,
        edit_cost: EDIT_MU / size.iter().sum::<f64>().max(1.0),
        min_gain: 0.2,
    };
    let problem = |moves: Vec<crate::pick::Move>, slack: f64| problem_at(&input_base, moves, slack);
    let apply = |cands: &[Candidate], p: &crate::pick::Problem, plan: &crate::pick::Plan, from: &[usize]| -> (Vec<usize>, Weights) {
        let mut choice = from.to_vec();
        for &mi in &plan.moves {
            for &k in &p.moves[mi].members {
                choice[k] = p.moves[mi].cand;
            }
        }
        let w = blend(model, ctx, cands, &choice, &active, seam);
        (choice, w)
    };
    // Moves measured apart can interact where they meet, so the few best
    // plans are scored on the mesh and the best of those is kept.
    let real = |w: &Weights, gate_first: bool| -> (i64, f64) {
        let ev = evaluate_with(model, ctx, w, false);
        let re = region_energy(ctx, &ev);
        let mut p = problem(Vec::new(), 0.0);
        p.gate_first = gate_first;
        let change: f64 = (0..model.nverts()).map(|v| l1(&w[v], &model.weights[v])).sum();
        (p.fails(&flag_counts(ctx, &ev.flags, None)), crate::pick::energy_of(&size, &re) + p.edit_cost * change)
    };
    let choose = |cands: &[Candidate], p: &crate::pick::Problem, tag: &str, from: &[usize]| -> (Vec<usize>, Weights, (i64, f64), bool) {
        let plans = crate::pick::pick_top(p, 4);
        let mut best: Option<(Vec<usize>, Weights, (i64, f64))> = None;
        for plan in &plans {
            let (c, wt) = apply(cands, p, plan, from);
            let r = real(&wt, p.gate_first);
            if debug {
                eprintln!(
                    "{tag}: {} moves, predicted ({}, {:.4}), real ({}, {:.4}), exact {}",
                    plan.moves.len(),
                    plan.fails,
                    plan.energy,
                    r.0,
                    r.1,
                    plan.exact
                );
            }
            if best.as_ref().is_none_or(|b| r.0 < b.2.0 || (r.0 == b.2.0 && r.1 < b.2.1 - 1e-12)) {
                best = Some((c, wt, r));
            }
        }
        let (c, w, r) = best.unwrap();
        (c, w, r, plans[0].exact)
    };
    // A candidate is measured only if, applied to the whole repair area, it
    // improves some failing region: the rest cannot repair anything, and
    // measuring costs one mesh evaluation per region set. Depends only on
    // the candidate's own numbers, so a better candidate is never dropped.
    let useful = |c: &Candidate| (0..nr).any(|r| failing[r] && c.region_e[r] < orig_e[r] - 1e-9);
    let mut moves: Vec<crate::pick::Move> =
        (1..cands.len()).filter(|&c| useful(&cands[c])).flat_map(|c| measure(&cands, c, &input_base)).collect();
    let p1 = problem(moves.clone(), 0.01);
    let (mut choice, mut w, mut best_real, mut exact) = choose(&cands, &p1, "pick 1", &input_base.choice);
    // Every stage's result is judged on the mesh; the fix is the best one
    // that breaks no rule (passes the gate first, then score), so a later
    // stage never undoes an earlier good one. Hard rule: no region may gain
    // a failing finding it did not have.
    let new_fails = |rep: &Report| {
        rep.findings.iter().any(|f| f.severity == "fail" && !rep0.findings.iter().any(|g| g.severity == "fail" && g.region == f.region))
    };
    type Kept = (Vec<usize>, Weights, Eval, Report);
    let mut kept: Option<Kept> = None;
    let consider = |kept: &mut Option<Kept>, choice: &[usize], w: &Weights| -> bool {
        let ev = evaluate(model, ctx, w);
        let rep = build(file, model, ctx, &ev);
        if std::env::var_os("WF_DEBUG").is_some() {
            let gained: Vec<String> = rep
                .findings
                .iter()
                .filter(|f| f.severity == "fail" && !rep0.findings.iter().any(|g| g.severity == "fail" && g.region == f.region))
                .map(|f| format!("{} {} {}", f.region, f.code, f.verts))
                .collect();
            eprintln!("stage: score {:.1}, pass {}, gains {:?}", rep.score, rep.pass, gained);
        }
        if new_fails(&rep) {
            return false;
        }
        if kept.as_ref().is_none_or(|k| (rep.pass, rep.score) > (k.3.pass, k.3.score)) {
            *kept = Some((choice.to_vec(), w.clone(), ev, rep));
        }
        true
    };
    let mut valid = consider(&mut kept, &choice, &w);

    // Method 4 starts from the best mixes so far: the plan picked for the
    // gate and the plan with the lowest score energy (it often clears the
    // last findings from there), then everything is picked again with them
    // as more candidates (the earlier plans stay possible).
    if o.methods.contains(&Method::Optimize) {
        let mut seeds = vec![w.clone()];
        // When nothing passes the gate, the score-only pick is the same
        // search; run it only when the gate pick passes.
        if best_real.0 == 0 {
            let mut by_energy = problem(moves.clone(), 0.01);
            by_energy.gate_first = false;
            let (_, we, _, e) = choose(&cands, &by_energy, "pick 1 (score only)", &input_base.choice);
            exact &= e;
            if we != w {
                seeds.push(we);
            }
        }
        for (k, seed) in seeds.iter().enumerate() {
            let (ev_mix, _) = score(seed);
            let opt = optimize(model, ctx, seed, &ev_mix);
            push(if k == 0 { "optimize".into() } else { "optimize(score)".into() }, opt, &mut cands);
            if useful(&cands[cands.len() - 1]) {
                moves.extend(measure(&cands, cands.len() - 1, &input_base));
            }
        }
        let p2 = problem(moves.clone(), 0.01);
        let (c2, w2, r2, e2) = choose(&cands, &p2, "pick 2", &input_base.choice);
        exact &= e2;
        let v2 = consider(&mut kept, &c2, &w2);
        if (v2 && !valid) || (v2 == valid && (r2.0 < best_real.0 || (r2.0 == best_real.0 && r2.1 < best_real.1 - 1e-12))) {
            choice = c2;
            w = w2;
            best_real = r2;
            valid = v2;
        }
    }

    // Refinement: moves measured apart miss how neighboring repairs meet
    // (seams between two candidates). Re-measure every move against the
    // current plan (including putting regions back to the input) and pick
    // again, exactly; go on while the result keeps the rules and scores
    // better on the mesh, or repairs a rule the current plan broke. Rules
    // stay relative to the input.
    for round in 0..4 {
        let base = base_of(choice.clone(), w.clone());
        let moves_r: Vec<crate::pick::Move> = (0..cands.len())
            .filter(|&c| c == 0 || useful(&cands[c]) || base.choice.contains(&c))
            .flat_map(|c| measure(&cands, c, &base))
            .collect();
        let mut p = problem_at(&base, moves_r, 0.01);
        p.min_gain = 0.0;
        let (c3, w3, r3, e3) = choose(&cands, &p, &format!("refine {round}"), &base.choice);
        if c3 == choice {
            break;
        }
        let v3 = consider(&mut kept, &c3, &w3);
        let gain = r3.0 < best_real.0 || (r3.0 == best_real.0 && r3.1 < best_real.1 - 1e-9);
        // Follow a better-scoring plan even if it breaks a rule (the next
        // round can repair it); the kept result only ever takes plans that
        // keep the rules.
        if !(gain || (v3 && !valid)) {
            break;
        }
        exact &= e3;
        choice = c3;
        w = w3;
        best_real = r3;
        valid = v3;
    }

    // The last plan scores best on the mesh but breaks a rule: measured
    // apart, its repairs interact (a region between two repairs gains a
    // finding). Drop repairs one at a time, each time the one whose removal
    // leaves the fewest such regions and then the lowest energy, and offer
    // the first rule-abiding result. A repair is a connected set of regions
    // on one candidate.
    let count_new = |rep: &Report| -> usize {
        let mut regions: Vec<&str> = rep
            .findings
            .iter()
            .filter(|f| f.severity == "fail" && !rep0.findings.iter().any(|g| g.severity == "fail" && g.region == f.region))
            .map(|f| f.region.as_str())
            .collect();
        regions.sort_unstable();
        regions.dedup();
        regions.len()
    };
    if new_fails(&build(file, model, ctx, &evaluate(model, ctx, &w))) {
        let mut cur = choice.clone();
        loop {
            let mut units: Vec<Vec<usize>> = Vec::new();
            let mut seen = vec![false; nr];
            for r in 0..nr {
                if cur[r] == 0 || seen[r] {
                    continue;
                }
                let mut unit = vec![r];
                seen[r] = true;
                let mut i = 0;
                while i < unit.len() {
                    for &k in &adjr[unit[i]] {
                        if !seen[k] && cur[k] == cur[r] {
                            seen[k] = true;
                            unit.push(k);
                        }
                    }
                    i += 1;
                }
                units.push(unit);
            }
            if units.is_empty() {
                break;
            }
            let mut best: Option<((usize, f64), Vec<usize>, Weights)> = None;
            for unit in &units {
                let mut trial = cur.clone();
                for &k in unit {
                    trial[k] = 0;
                }
                let wt = blend(model, ctx, &cands, &trial, &active, seam);
                let ev_t = evaluate_with(model, ctx, &wt, false);
                let rep_t = build(file, model, ctx, &ev_t);
                let key = (count_new(&rep_t), crate::pick::energy_of(&size, &region_energy(ctx, &ev_t)));
                if best.as_ref().is_none_or(|b| key.0 < b.0.0 || (key.0 == b.0.0 && key.1 < b.0.1)) {
                    best = Some((key, trial, wt));
                }
            }
            let (key, trial, wt) = best.unwrap();
            if debug {
                eprintln!("drop: {} regions gain a failing finding, energy {:.4}", key.0, key.1);
            }
            cur = trial;
            if key.0 == 0 {
                consider(&mut kept, &cur, &wt);
                break;
            }
        }
    }
    let (mut choice, mut w, mut ev, mut rep1) = match kept {
        Some(k) => k,
        None => (vec![0; nr], model.weights.clone(), evaluate(model, ctx, &model.weights), rep0.clone()),
    };
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
        pick_exact: exact,
        external_kept: !o.external.is_empty(),
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
