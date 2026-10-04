//! Closest-point weight transfer (what Blender's Data Transfer "nearest face
//! interpolated" does), plus the uniform-grid triangle index behind it.

use crate::math::{Vec3, v3};
use crate::scene::{VW, Weights, normalize_vw};
use std::collections::BTreeMap;

pub struct TriIndex<'a> {
    pos: &'a [Vec3],
    tris: Vec<[u32; 3]>,
    cell: f64,
    grid: BTreeMap<(i64, i64, i64), Vec<u32>>,
    lo: Vec3,
    hi: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub tri: u32,
    pub bary: [f64; 3],
    pub dist: f64,
    pub point: Vec3,
}

/// Closest point on triangle abc to p (Ericson, Real-Time Collision
/// Detection 5.1.5). Returns barycentrics.
pub fn closest_on_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> [f64; 3] {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [1.0 - v, v, 0.0];
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [1.0 - w, 0.0, w];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [0.0, 1.0 - w, w];
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    [1.0 - v - w, v, w]
}

impl<'a> TriIndex<'a> {
    pub fn new(pos: &'a [Vec3], tris: Vec<[u32; 3]>) -> TriIndex<'a> {
        let mut lo = v3(f64::MAX, f64::MAX, f64::MAX);
        let mut hi = -lo;
        let mut esum = 0.0;
        for t in &tris {
            for &i in t {
                lo = lo.min(pos[i as usize]);
                hi = hi.max(pos[i as usize]);
            }
            esum += (pos[t[0] as usize] - pos[t[1] as usize]).len();
        }
        let cell = if tris.is_empty() { 1.0 } else { (esum / tris.len() as f64 * 2.0).max((hi - lo).len() * 1e-3).max(1e-9) };
        let mut grid: BTreeMap<(i64, i64, i64), Vec<u32>> = BTreeMap::new();
        for (ti, t) in tris.iter().enumerate() {
            let p = t.map(|i| pos[i as usize]);
            let l = p[0].min(p[1]).min(p[2]);
            let h = p[0].max(p[1]).max(p[2]);
            let (a, b) = (Self::key(l, cell), Self::key(h, cell));
            for x in a.0..=b.0 {
                for y in a.1..=b.1 {
                    for z in a.2..=b.2 {
                        grid.entry((x, y, z)).or_default().push(ti as u32);
                    }
                }
            }
        }
        TriIndex { pos, tris, cell, grid, lo, hi }
    }

    fn key(p: Vec3, cell: f64) -> (i64, i64, i64) {
        ((p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64)
    }

    pub fn tri(&self, t: u32) -> [u32; 3] {
        self.tris[t as usize]
    }

    pub fn closest(&self, p: Vec3) -> Option<Hit> {
        if self.tris.is_empty() {
            return None;
        }
        let c = Self::key(p, self.cell);
        // Rings until the best hit is provably inside the searched cube.
        let span = ((self.hi - self.lo).len() / self.cell).ceil() as i64 + 2;
        let outside = (p.max(self.lo) - p).len().max((p - p.min(self.hi)).len());
        let start = ((outside / self.cell).floor() as i64 - 1).max(0);
        let mut best: Option<Hit> = None;
        let mut r = start;
        while r <= span + start {
            for x in c.0 - r..=c.0 + r {
                for y in c.1 - r..=c.1 + r {
                    for z in c.2 - r..=c.2 + r {
                        if (x - c.0).abs() != r && (y - c.1).abs() != r && (z - c.2).abs() != r {
                            continue; // interior cells were searched in earlier rings
                        }
                        let Some(list) = self.grid.get(&(x, y, z)) else { continue };
                        for &ti in list {
                            let t = self.tris[ti as usize];
                            let (a, b, cc) = (self.pos[t[0] as usize], self.pos[t[1] as usize], self.pos[t[2] as usize]);
                            let bary = closest_on_tri(p, a, b, cc);
                            let q = a * bary[0] + b * bary[1] + cc * bary[2];
                            let d = (q - p).len();
                            if best.is_none_or(|h| d < h.dist || (d == h.dist && ti < h.tri)) {
                                best = Some(Hit { tri: ti, bary, dist: d, point: q });
                            }
                        }
                    }
                }
            }
            if let Some(h) = best {
                if h.dist <= r as f64 * self.cell {
                    break;
                }
            }
            r += 1;
        }
        best
    }
}

pub fn blend(w: &Weights, tri: [u32; 3], bary: [f64; 3]) -> VW {
    let mut out: VW = Vec::new();
    for k in 0..3 {
        for &(j, x) in &w[tri[k] as usize] {
            out.push((j, x * bary[k]));
        }
    }
    normalize_vw(out)
}

/// Robust transfer onto a vertex subset (Abdrashitov et al. 2023, "Robust
/// Skin Weights Transfer via Weight Inpainting"): a target vertex takes the
/// source weights only where the closest source point is near (`max_dist`)
/// and the surfaces face the same way (two-sided, `min_cos`); every other
/// vertex is inpainted harmonically from the matched ones, so gaps (between
/// the legs, under a cape) get smooth weights instead of a nearest-surface
/// jump. Returns the match map: (closest source triangle, barycentrics,
/// matched?) per subset vertex.
pub fn match_subset(
    index: &TriIndex,
    src_normals: &[Vec3],
    targets: &[Vec3],
    target_normals: &[Vec3],
    subset: &[u32],
    max_dist: f64,
    min_cos: f64,
) -> Vec<(u32, [u32; 3], [f64; 3], bool)> {
    let mut out: Vec<(u32, [u32; 3], [f64; 3], bool, f64)> = subset
        .iter()
        .filter_map(|&v| {
            let h = index.closest(targets[v as usize])?;
            let t = index.tri(h.tri);
            let n =
                (src_normals[t[0] as usize] * h.bary[0] + src_normals[t[1] as usize] * h.bary[1] + src_normals[t[2] as usize] * h.bary[2])
                    .normalized();
            let ok = h.dist <= max_dist && n.dot(target_normals[v as usize]).abs() >= min_cos;
            Some((v, t, h.bary, ok, h.dist))
        })
        .collect();
    // Nothing matched (piece floats off the body): trust the closest few.
    if !out.is_empty() && !out.iter().any(|e| e.3) {
        let dmin = out.iter().map(|e| e.4).fold(f64::MAX, f64::min);
        for e in &mut out {
            e.3 = e.4 <= dmin * 1.5 + 1e-9;
        }
    }
    out.into_iter().map(|(v, t, b, ok, _)| (v, t, b, ok)).collect()
}

/// Weights for the subset from a match map: matched vertices blend the
/// source weights, the rest are inpainted.
pub fn inpaint_from_matches(adj: &[Vec<u32>], matches: &[(u32, [u32; 3], [f64; 3], bool)], src_w: &Weights) -> Weights {
    let nv = adj.len();
    let mut known: Vec<Option<VW>> = vec![None; nv];
    let mut nearest: Vec<Option<VW>> = vec![None; nv];
    let subset: Vec<u32> = matches.iter().map(|m| m.0).collect();
    for &(v, t, b, ok) in matches {
        let w = blend(src_w, t, b);
        if ok {
            known[v as usize] = Some(w.clone());
        }
        nearest[v as usize] = Some(w);
    }
    crate::solve::inpaint(adj, &subset, &known, &|v| nearest[v].clone().unwrap_or_default())
}

/// Published defaults for robust transfer (Abdrashitov et al. 2023): match
/// within 5% of the bounding-box diagonal and 30 degrees of facing.
pub const ROBUST_DIST: f64 = 0.05;
pub const ROBUST_DEG: f64 = 30.0;

/// Matches `targets` against the triangles whose vertices are all
/// `is_src`, on one mesh (pieces onto the body of the same character).
pub fn match_within(
    pos: &[Vec3],
    tris: &[[u32; 3]],
    is_src: &[bool],
    targets: &[u32],
    max_dist: f64,
    deg: f64,
) -> Vec<(u32, [u32; 3], [f64; 3], bool)> {
    let src_tris: Vec<[u32; 3]> = tris.iter().filter(|t| t.iter().all(|&i| is_src[i as usize])).copied().collect();
    if src_tris.is_empty() || targets.is_empty() {
        return Vec::new();
    }
    let normals = crate::scene::vertex_normals(pos, tris);
    let index = TriIndex::new(pos, src_tris);
    match_subset(&index, &normals, pos, &normals, targets, max_dist, deg.to_radians().cos())
}
