//! How close a rig's weights come to a professional rig's on the same mesh
//! and skeleton: the benchmark's main score (2026-11-13, the owner: "we
//! should have quality benchmarks").
//!
//! A professional rig's weights are taken as ground truth. The weights
//! under test (a generator's prediction for that mesh and skeleton, the
//! same after `weights fix`) are compared in two ways:
//! - **motion**: each vertex posed with the test weights against the same
//!   vertex posed with the pro weights, through the same poses (the
//!   range of motion, and the game's clips), on a body scaled to 1.75 m.
//!   Reported as the mean distance, the 95th percentile over the skin of
//!   each vertex's worst frame, and the share of the skin that is ever
//!   more than 2 cm off (what a person would see move differently);
//! - **weights**: half the L1 distance per vertex (0 same, 1 disjoint),
//!   mean over vertices.

use crate::math::Vec3;
use crate::metrics::Ctx;
use crate::scene::{Model, Weights};
use rayon::prelude::*;
use serde::Serialize;

/// Off by this much (m, on a 1.75 m body) counts as visibly different.
pub const SEEN_M: f64 = 0.02;

#[derive(Clone, Debug, Serialize)]
pub struct Motion {
    pub group: String,
    pub frames: usize,
    /// cm, mean over vertices and frames.
    pub mean_cm: f64,
    /// cm, 95th percentile over vertices of each vertex's worst frame.
    pub p95_cm: f64,
    /// % of the surface ever more than 2 cm off.
    pub off_pct: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Closeness {
    pub verts: usize,
    /// Mean half-L1 weight distance (0..1).
    pub weight_err: f64,
    pub motion: Vec<Motion>,
}

/// The test model's weights in the pro model's joint and vertex numbering.
/// Joints by name; vertices by index when the meshes match, else by the
/// nearest rest position.
pub fn align(pro: &Model, test: &Model) -> Result<Weights, String> {
    let jmap: Vec<Option<u32>> = test.skel.names.iter().map(|n| pro.skel.joint_by_name(n).map(|j| j as u32)).collect();
    let tol = 1e-5 * pro.scale;
    let same = pro.nverts() == test.nverts() && (0..pro.nverts()).all(|v| (pro.rest[v] - test.rest[v]).len() < tol);
    let vmap: Vec<usize> = if same {
        (0..pro.nverts()).collect()
    } else {
        let cell = pro.scale * 0.01;
        let key = |p: Vec3| ((p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64);
        let mut grid: std::collections::BTreeMap<(i64, i64, i64), Vec<usize>> = Default::default();
        for (i, p) in test.rest.iter().enumerate() {
            grid.entry(key(*p)).or_default().push(i);
        }
        (0..pro.nverts())
            .map(|v| {
                let p = pro.rest[v];
                let k = key(p);
                let mut best = (f64::MAX, usize::MAX);
                for r in 0..4i64 {
                    for dx in -r..=r {
                        for dy in -r..=r {
                            for dz in -r..=r {
                                if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                                    continue;
                                }
                                for &i in grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)).into_iter().flatten() {
                                    let d = (test.rest[i] - p).len();
                                    if d < best.0 || (d == best.0 && i < best.1) {
                                        best = (d, i);
                                    }
                                }
                            }
                        }
                    }
                    if best.1 != usize::MAX {
                        break;
                    }
                }
                best.1
            })
            .collect()
    };
    if vmap.iter().any(|&i| i == usize::MAX) {
        return Err("meshes do not match (a vertex has no counterpart within 3% of the body)".into());
    }
    let mut missing = 0usize;
    let w: Weights = vmap
        .iter()
        .map(|&i| {
            let vw: crate::scene::VW = test.weights[i]
                .iter()
                .filter_map(|&(j, x)| match jmap[j as usize] {
                    Some(p) => Some((p, x)),
                    None => {
                        missing += 1;
                        None
                    }
                })
                .collect();
            crate::scene::normalize_vw(vw)
        })
        .collect();
    let _ = missing;
    Ok(w)
}

/// Mean half-L1 distance between two weight sets on the same numbering.
pub fn weight_err(a: &Weights, b: &Weights) -> f64 {
    let n = a.len().max(1) as f64;
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let mut d = 0.0;
            for &(j, w) in x {
                d += (w - crate::scene::weight_of(y, j)).abs();
            }
            for &(j, w) in y {
                if crate::scene::weight_of(x, j) == 0.0 {
                    d += w;
                }
            }
            0.5 * d
        })
        .sum::<f64>()
        / n
}

/// Motion closeness of `w` to the pro model's own weights over a pose group.
pub fn motion(pro: &Model, ctx: &Ctx, w: &Weights, group: &str) -> Motion {
    let k = crate::owner::BODY_M / crate::owner::height(pro);
    let area = crate::owner::vertex_areas(pro);
    let total: f64 = area.iter().sum::<f64>().max(1e-12);
    let nv = pro.nverts();
    let per: Vec<Vec<f64>> = ctx
        .mats
        .par_iter()
        .map(|m| {
            let a = crate::skin::lbs(pro, m, &pro.weights);
            let b = crate::skin::lbs(pro, m, w);
            (0..nv).map(|v| (a[v] - b[v]).len() * k).collect()
        })
        .collect();
    let nf = per.len().max(1) as f64;
    let mut worst = vec![0.0f64; nv];
    let mut sum = 0.0;
    for f in &per {
        for v in 0..nv {
            worst[v] = worst[v].max(f[v]);
            sum += f[v];
        }
    }
    let mut sorted = worst.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let p95 = sorted.get(sorted.len() * 95 / 100).copied().unwrap_or(0.0);
    let off: f64 = (0..nv).filter(|&v| worst[v] > SEEN_M).map(|v| area[v]).sum();
    Motion {
        group: group.into(),
        frames: per.len(),
        mean_cm: (sum / nf / nv.max(1) as f64 * 1e4).round() / 100.0,
        p95_cm: (p95 * 1e4).round() / 100.0,
        off_pct: (off / total * 1e4).round() / 100.0,
    }
}

/// The whole comparison over the given pose groups (name, context).
pub fn closeness(pro: &Model, test: &Model, groups: &[(&str, &Ctx)]) -> Result<Closeness, String> {
    let w = align(pro, test)?;
    Ok(Closeness {
        verts: pro.nverts(),
        weight_err: (weight_err(&pro.weights, &w) * 1e4).round() / 1e4,
        motion: groups.iter().map(|(n, c)| motion(pro, c, &w, n)).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rig_is_closest_to_itself() {
        let m = Model::from_glb(crate::fixtures::to_glb(&crate::fixtures::build("clean"))).unwrap();
        let ctx = Ctx::new(&m, &crate::metrics::CtxOpts::default());
        let c = closeness(&m, &m, &[("poses", &ctx)]).unwrap();
        assert_eq!(c.weight_err, 0.0);
        assert_eq!(c.motion[0].mean_cm, 0.0);
        let bad = Model::from_glb(crate::fixtures::to_glb(&crate::fixtures::build("bleed"))).unwrap();
        let c = closeness(&m, &bad, &[("poses", &ctx)]).unwrap();
        assert!(c.weight_err > 0.0 && c.motion[0].off_pct > 0.0, "{c:?}");
    }
}
