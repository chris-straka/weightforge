//! The checker's headline: what a person watching the character notices,
//! measured, with where it is and how big an area.
//!
//! The 0-100 score is half the single worst region, so it counts faults
//! more than it measures quality (hand-made rigs scored 3 to 70). The
//! headline is instead **tears**: the share of the surface inside failing
//! findings, on the game's clips and on the range-of-motion poses apart.
//! Its findings are the owner's own words for what goes wrong
//! (2026-11-12): cloth moving with a bone it does not belong to, cloth not
//! following its limb, a stiff part (belt, buckle, pouch, plate) bending,
//! a limb passing through another, and stretch past 3.5x; plus the
//! checker's earlier ones (volume, bleed, visible speckle, pieces).
//!
//! Every amount is given for a body 1.75 m tall (the character scaled to
//! it), so rigs of any size read alike; areas in cm^2 and as a share of
//! the surface.

use crate::anatomy::Group;
use crate::math::{Quat, Vec3};
use crate::metrics::{Ctx, Eval};
use crate::report::Report;
use crate::scene::{Model, Weights};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeMap;

/// The body height every amount is scaled to.
pub const BODY_M: f64 = 1.75;

#[derive(Clone, Debug, Serialize)]
pub struct Thresholds {
    /// Wrong bone: a vertex this far (m, on a 1.75 m body) from where its
    /// own bones would put it.
    pub wrong_bone_m: f64,
    /// Not following: a mid-limb vertex this far off its limb's motion,
    /// in limb radii, and at least `lag_min_m`.
    pub lag_radii: f64,
    pub lag_min_m: f64,
    /// Stiff part: bends when some vertex is this far off the part's best
    /// rigid motion, as a share of the part's size, and at least
    /// `stiff_min_m`.
    pub stiff_share: f64,
    pub stiff_min_m: f64,
    /// A finding needs this many vertices and this much area (cm^2).
    pub min_verts: usize,
    pub min_area_cm2: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            // Set 2026-11-13 between the hand-made game rigs and the
            // generated cast on the hero's clips (bench/rig_score.py,
            // docs/rig-score.md): hand-made rigs reach 4-8 cm (wrong bone)
            // and 2-4 cm (not following), the cast 7-70 cm.
            wrong_bone_m: 0.06,
            lag_radii: 0.5,
            lag_min_m: 0.05,
            stiff_share: 0.08,
            stiff_min_m: 0.015,
            min_verts: 3,
            min_area_cm2: 2.0,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Finding {
    /// wrong_bone, not_following, stiff_bends, limb_through, stretch,
    /// volume, bleed, noise, unweighted, piece_bones, piece_follow
    pub kind: String,
    pub code: String,
    pub severity: String,
    /// The part, in words: "left thigh, front", "a band round the waist".
    pub part: String,
    pub area_cm2: f64,
    pub area_pct: f64,
    pub verts: usize,
    /// Where it is worst (a pose or a clip frame).
    pub worst: String,
    /// What happens, in words, with the amount.
    pub detail: String,
    #[serde(skip)]
    pub vertices: Vec<u32>,
    #[serde(skip)]
    pub worst_pose: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Section {
    /// "game clips" or "poses".
    pub name: String,
    pub frames: usize,
    pub clips: Vec<String>,
    pub source: String,
    /// Share of the surface (area) inside failing findings, %.
    pub tears_pct: f64,
    /// The same by vertex count, % (the measure before areas).
    pub tears_verts_pct: f64,
    /// Skin inside another part, cm^2 (limb through limb, worst pose per
    /// pair of parts).
    pub through_cm2: f64,
    pub findings: Vec<Finding>,
}

/// Rest-pose area per vertex (a third of each triangle round it), m^2.
pub fn vertex_areas(model: &Model) -> Vec<f64> {
    let mut a = vec![0.0; model.nverts()];
    for t in &model.tris {
        let (p, q, r) = (model.rest[t[0] as usize], model.rest[t[1] as usize], model.rest[t[2] as usize]);
        let ar = 0.5 * (q - p).cross(r - p).len() / 3.0;
        for &i in t {
            a[i as usize] += ar;
        }
    }
    a
}

/// Body height (rest, along +Y).
pub fn height(model: &Model) -> f64 {
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for p in &model.rest {
        lo = lo.min(p.y);
        hi = hi.max(p.y);
    }
    (hi - lo).max(1e-9)
}

fn pretty(role: &str) -> String {
    let (base, side) = match role.rsplit_once('.') {
        Some((b, "L")) => (b, "left "),
        Some((b, "R")) => (b, "right "),
        _ => (role, ""),
    };
    let base = base.trim_end_matches('*');
    let base = if base == "spine" { "back" } else { base };
    format!("{side}{}", base.replace('_', " "))
}

/// A bone in words: its role, else its name.
pub fn bone_label(model: &Model, j: usize) -> String {
    let j = crate::helpers::owner(&model.skel, j);
    let roles = &model.skel.anatomy.roles;
    // Single-bone roles first (a chain role names many).
    if let Some((r, _)) = roles.iter().filter(|(r, _)| !r.ends_with('*') && !r.starts_with("fingers")).find(|(_, js)| js.contains(&j)) {
        return pretty(r);
    }
    if let Some(sp) = roles.get("spine*").filter(|js| js.contains(&j)) {
        let i = sp.iter().position(|&x| x == j).unwrap();
        return if i + 1 == sp.len() {
            "chest".into()
        } else if i == 0 {
            "waist".into()
        } else {
            "middle of the back".into()
        };
    }
    if let Some((r, _)) = roles.iter().find(|(_, js)| js.contains(&j)) {
        return pretty(r);
    }
    model.skel.names[j].clone()
}

/// Joint hops between every pair the checks ask about (cached per start).
struct Hops<'a> {
    model: &'a Model,
    cache: BTreeMap<usize, Vec<u32>>,
}

impl Hops<'_> {
    fn from(&mut self, a: usize) -> &Vec<u32> {
        let m = self.model;
        self.cache.entry(a).or_insert_with(|| {
            let n = m.njoints();
            let mut d = vec![u32::MAX; n];
            let mut q = std::collections::VecDeque::new();
            d[a] = 0;
            q.push_back(a);
            while let Some(x) = q.pop_front() {
                let mut nb: Vec<usize> = m.skel.jchildren[x].clone();
                nb.extend(m.skel.jparent[x]);
                for y in nb {
                    // Helpers sit on their driver: no hop.
                    let step = if crate::helpers::helper_of(&m.skel, y).is_some() || crate::helpers::helper_of(&m.skel, x).is_some() {
                        0
                    } else {
                        1
                    };
                    if d[y] == u32::MAX || d[y] > d[x] + step {
                        d[y] = d[x] + step;
                        q.push_back(y);
                    }
                }
            }
            d
        })
    }
}

/// Horn's best rigid motion of `a` onto `b` (rotation, about the
/// centroids); returns the largest vertex distance off it.
fn rigid_residual(a: &[Vec3], b: &[Vec3]) -> f64 {
    let n = a.len().max(1) as f64;
    let ca = a.iter().fold(Vec3::ZERO, |s, p| s + *p) / n;
    let cb = b.iter().fold(Vec3::ZERO, |s, p| s + *p) / n;
    let mut s = [[0.0f64; 3]; 3];
    for (p, q) in a.iter().zip(b) {
        let (x, y) = (*p - ca, *q - cb);
        let (xa, ya) = ([x.x, x.y, x.z], [y.x, y.y, y.z]);
        for i in 0..3 {
            for k in 0..3 {
                s[i][k] += xa[i] * ya[k];
            }
        }
    }
    let (sxx, sxy, sxz, syx, syy, syz, szx, szy, szz) = (s[0][0], s[0][1], s[0][2], s[1][0], s[1][1], s[1][2], s[2][0], s[2][1], s[2][2]);
    let nmat = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let v = top_eigenvector(nmat);
    let q = Quat { w: v[0], x: v[1], y: v[2], z: v[3] }.normalized();
    a.iter().zip(b).map(|(p, t)| (q.rotate(*p - ca) + cb - *t).len()).fold(0.0, f64::max)
}

/// The eigenvector of the largest eigenvalue of a symmetric 4x4 matrix
/// (cyclic Jacobi).
fn top_eigenvector(m: [[f64; 4]; 4]) -> [f64; 4] {
    let mut a = m;
    let mut v = [[0.0f64; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..50 {
        let off: f64 = (0..4).flat_map(|i| (0..4).filter(move |&k| k != i).map(move |k| (i, k))).map(|(i, k)| a[i][k] * a[i][k]).sum();
        if off < 1e-24 {
            break;
        }
        for p in 0..3 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..4 {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..4 {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let best = (0..4).max_by(|&x, &y| a[x][x].partial_cmp(&a[y][y]).unwrap()).unwrap();
    [v[0][best], v[1][best], v[2][best], v[3][best]]
}

/// A part that should stay stiff: its vertices and what to call it.
pub struct Stiff {
    pub verts: Vec<u32>,
    pub name: String,
    pub size: f64,
    /// Judged on the game's clips only (the belt line: a waist bends in a
    /// range-of-motion spine bend by design).
    pub clips_only: bool,
}

const STIFF_WORDS: &[&str] = &[
    "belt",
    "buckle",
    "pouch",
    "plate",
    "armor",
    "armour",
    "bag",
    "satchel",
    "scabbard",
    "sheath",
    "badge",
    "brooch",
    "bracer",
    "pauldron",
    "gauntlet",
    "helmet",
    "shield",
    "holster",
    "vambrace",
    "greave",
    "cuirass",
    "breastplate",
];

fn words(name: &str) -> Vec<String> {
    name.split(|c: char| !c.is_alphanumeric()).filter(|s| !s.is_empty()).map(|s| s.to_lowercase()).collect()
}

/// A stiff word, its plural, or it with a number (`buckle2`).
fn stiff_word(w: &str) -> bool {
    STIFF_WORDS.iter().any(|s| {
        w.strip_prefix(s).is_some_and(|rest| rest.is_empty() || rest == "s" || rest == "es" || rest.bytes().all(|b| b.is_ascii_digit()))
    })
}

/// Stiff parts: pieces named for one (`belt`, `buckle`...) and small,
/// solid loose shells off the head (a buckle or a boot's heel cap
/// modelled apart). A fused body's belt is not found (see below).
pub fn stiff_parts(model: &Model, vgroup: &[Group], h: f64) -> Vec<Stiff> {
    let nv = model.nverts();
    let mut out: Vec<Stiff> = Vec::new();
    let size_of = |vs: &[u32]| {
        let (mut a, mut b) = (model.rest[vs[0] as usize], model.rest[vs[0] as usize]);
        for &v in vs {
            a = a.min(model.rest[v as usize]);
            b = b.max(model.rest[v as usize]);
        }
        (b - a).len()
    };
    // Named pieces.
    let mut taken = vec![false; nv];
    for (pi, part) in model.parts.iter().enumerate() {
        if part.rigid {
            continue;
        }
        if words(&part.name).iter().any(|w| stiff_word(w)) {
            let vs: Vec<u32> = (0..nv as u32).filter(|&v| model.vpart[v as usize] as usize == pi).collect();
            if vs.len() >= 4 {
                for &v in &vs {
                    taken[v as usize] = true;
                }
                out.push(Stiff { size: size_of(&vs), name: format!("the {}", part.name), verts: vs, clips_only: false });
            }
        }
    }
    // Small loose shells.
    let mut comp = vec![u32::MAX; nv];
    let mut shells: Vec<Vec<u32>> = Vec::new();
    for s in 0..nv {
        if comp[s] != u32::MAX {
            continue;
        }
        let id = shells.len() as u32;
        comp[s] = id;
        let mut st = vec![s];
        let mut vs = Vec::new();
        while let Some(v) = st.pop() {
            vs.push(v as u32);
            for &u in &model.adj[v] {
                if comp[u as usize] == u32::MAX {
                    comp[u as usize] = id;
                    st.push(u as usize);
                }
            }
        }
        shells.push(vs);
    }
    for vs in &shells {
        if vs.len() >= 8 && vs.len() * 33 < nv && !vs.iter().any(|&v| taken[v as usize] || model.rigid[v as usize]) {
            let sz = size_of(vs);
            // A buckle or a pouch: off the head (lashes, teeth, brows are
            // faces), and solid, not a strip or a card (hair, frills).
            let (mut a, mut b) = (model.rest[vs[0] as usize], model.rest[vs[0] as usize]);
            for &v in vs {
                a = a.min(model.rest[v as usize]);
                b = b.max(model.rest[v as usize]);
            }
            let e = b - a;
            let (thin, thick) = (e.x.min(e.y).min(e.z), e.x.max(e.y).max(e.z));
            let on_head = vs.iter().filter(|&&v| vgroup[v as usize] == Group::Head).count() * 2 > vs.len();
            if sz < 0.12 * h && thin > 0.35 * thick && !on_head {
                for &v in vs {
                    taken[v as usize] = true;
                }
                out.push(Stiff { size: sz, name: "a small loose part".into(), verts: vs.clone(), clips_only: false });
            }
        }
    }
    // Painted parts (a band of one colour round the waist, a patch unlike
    // what is round it) were tried on the fused generated bodies and left
    // out (2026-11-13): on 36 rigs they found one patch, the tip of a
    // beard; a belt the colour of the vest it sits on is not a patch
    // (docs/rig-score.md).
    // The belt line (the ring of torso skin between the hips joint and the
    // first spine joint) was tried as a stiff part on the clips and left
    // out (2026-11-13): hand-made rigs' waists bend as much in a run (3.3
    // cm on the Quaternius and MPFB bodies) as the game hero's (2.5 cm),
    // so it flagged good rigs and told nothing (docs/rig-score.md).
    out
}

/// Front, back, or a side of a group of vertices, seen from its bone.
fn side_word(model: &Model, vs: &[u32], j: usize) -> &'static str {
    let a = &model.skel.anatomy;
    let c = vs.iter().fold(Vec3::ZERO, |s, &v| s + model.rest[v as usize]) / vs.len().max(1) as f64;
    let (h, t) = (model.skel.head[j], model.skel.tail[j]);
    let d = (t - h).normalized();
    let mut r = c - h;
    r = r - d * r.dot(d);
    if r.len() < 1e-9 {
        return "";
    }
    let left = a.up.cross(a.forward).normalized();
    let (f, l) = (r.dot(a.forward), r.dot(left));
    if f.abs() >= l.abs() {
        if f > 0.0 { "front" } else { "back" }
    } else {
        let limb_side = match &a.group[j] {
            Group::Limb(s) if s.ends_with(".L") => 1.0,
            Group::Limb(s) if s.ends_with(".R") => -1.0,
            _ => 0.0,
        };
        if limb_side == 0.0 {
            if l > 0.0 { "left side" } else { "right side" }
        } else if l * limb_side > 0.0 {
            "outer side"
        } else {
            "inner side"
        }
    }
}

/// Splits flagged vertices into connected areas (by mesh edges).
fn areas(model: &Model, flagged: &[bool]) -> Vec<Vec<u32>> {
    let mut seen = vec![false; flagged.len()];
    let mut out = Vec::new();
    for s in 0..flagged.len() {
        if !flagged[s] || seen[s] {
            continue;
        }
        seen[s] = true;
        let mut st = vec![s];
        let mut vs = Vec::new();
        while let Some(v) = st.pop() {
            vs.push(v as u32);
            for &u in &model.adj[v] {
                if flagged[u as usize] && !seen[u as usize] {
                    seen[u as usize] = true;
                    st.push(u as usize);
                }
            }
        }
        out.push(vs);
    }
    out
}

/// The headline for one pose group. `motion`: these are the game's clips
/// (a limb through another fails there; in range-of-motion extremes it is
/// a warning, full bends poke). `rep`/`ev`: the check of the same poses.
#[allow(clippy::too_many_arguments)]
pub fn assess(
    model: &Model,
    ctx: &Ctx,
    ev: &Eval,
    rep: &Report,
    w: &Weights,
    motion: bool,
    name: &str,
    clips: Vec<String>,
    source: String,
) -> Section {
    let th = Thresholds::default();
    let nv = model.nverts();
    let h = height(model);
    let k = BODY_M / h; // metres on the model -> metres on a 1.75 m body
    let varea = vertex_areas(model);
    let total_area: f64 = varea.iter().sum::<f64>().max(1e-12);
    let cm2 = |vs: &[u32]| vs.iter().map(|&v| varea[v as usize]).sum::<f64>() * k * k * 1e4;
    let sk = &model.skel;
    let anat = &sk.anatomy;
    // Each vertex's bone: the nearest by distance through the body that
    // carries weight (an unweighted IK bone at a hoof owns nothing), else
    // its region's.
    let mut weighted = vec![false; sk.joints.len()];
    for vw in w {
        for &(j, x) in vw {
            if x > 0.0 {
                weighted[crate::helpers::owner(sk, j as usize)] = true;
            }
        }
    }
    let vbone: Vec<usize> = (0..nv)
        .map(|v| {
            let near = ctx.geo.near[v].iter().map(|e| crate::helpers::owner(sk, e.0 as usize)).find(|&j| weighted[j]);
            match (near, ctx.region_bone[ctx.vregion[v] as usize]) {
                (Some(j), _) => j,
                (None, Some(j)) => crate::helpers::owner(sk, j),
                (None, None) => crate::helpers::owner(sk, ctx.geo.nearest(v).map(|e| e.0 as usize).unwrap_or(0)),
            }
        })
        .collect();
    let vgroup: Vec<Group> = vbone.iter().map(|&j| anat.group.get(j).cloned().unwrap_or_default()).collect();
    let bgroup = |j: usize| anat.group.get(j).cloned().unwrap_or_default();

    // ---- candidates: wrong bone (weight on another limb).
    let mut hops = Hops { model, cache: BTreeMap::new() };
    // A limb's first bones (and any bone of its above them): where a limb
    // meets the torso.
    let mut proximal = vec![false; sk.joints.len()];
    for (role, js) in &anat.roles {
        let first = ["shoulder.", "upper_arm.", "thigh."].iter().any(|p| role.starts_with(p)) || role.contains("_upper.");
        if first {
            for &j in js {
                let mut x = Some(j);
                while let Some(q) = x {
                    if !matches!(bgroup(q), Group::Limb(_)) {
                        break;
                    }
                    proximal[q] = true;
                    x = sk.jparent[q];
                }
            }
        }
    }
    for h in &sk.helpers {
        proximal[h.joint] = proximal[h.driver];
    }
    let mut foreign: Vec<Option<(Vec<(u32, f64)>, u32)>> = vec![None; nv];
    for v in 0..nv {
        if model.rigid[v] {
            continue;
        }
        let r = vbone[v];
        let gr = &vgroup[v];
        let mut home = Vec::new();
        let mut far: Option<(u32, f64)> = None;
        for &(b, x) in &w[v] {
            let bo = crate::helpers::owner(sk, b as usize);
            let gb = bgroup(bo);
            // The same part, the torso and head together, and the torso
            // with a limb's first bones (armpit, groin, flank) blend; a
            // limb never follows another limb, nor the torso a forearm,
            // hand, shin or foot.
            let core = |g: &Group| matches!(g, Group::Torso | Group::Head);
            let ok = bo == r
                || gb == *gr
                || (core(&gb) && core(gr))
                || (core(gr) && proximal[bo])
                || (core(&gb) && proximal[r])
                || (core(&gb) && hops.from(r)[bo] <= 1);
            if ok {
                home.push((b, x));
            } else if far.is_none_or(|f| x > f.1) {
                far = Some((bo as u32, x));
            }
        }
        if let Some((fb, _)) = far {
            if home.iter().map(|x| x.1).sum::<f64>() < 0.05 {
                home = vec![(r as u32, 1.0)];
            }
            foreign[v] = Some((crate::scene::normalize_vw(home), fb));
        }
    }
    // ---- candidates: not following (mid-limb, round the limb).
    let limb_roles = ["upper_arm.L", "upper_arm.R", "forearm.L", "forearm.R", "thigh.L", "thigh.R", "shin.L", "shin.R"];
    let limb_bones: Vec<usize> = limb_roles.iter().filter_map(|r| anat.roles.get(*r).map(|v| v[0])).collect();
    let mut lag_ref: Vec<Option<Vec<(u32, f64)>>> = vec![None; nv];
    for v in 0..nv {
        let b = vbone[v];
        if model.rigid[v] || !limb_bones.contains(&b) || model.parts[model.vpart[v] as usize].piece {
            continue;
        }
        let (hd, tl) = (sk.head[b], sk.tail[b]);
        let len = (tl - hd).len().max(1e-9);
        let t = (model.rest[v] - hd).dot((tl - hd) / len) / len;
        let radius = ctx.region_radius[ctx.vregion[v] as usize];
        if !(0.3..=0.8).contains(&t) || radius <= 0.0 || sk.seg_dist(b, model.rest[v]) > 2.0 * radius {
            continue;
        }
        let chain: Vec<usize> = std::iter::once(b).chain(sk.jchildren[b].iter().copied()).collect();
        let own: Vec<(u32, f64)> = w[v].iter().copied().filter(|&(j, _)| chain.contains(&crate::helpers::owner(sk, j as usize))).collect();
        let s: f64 = own.iter().map(|x| x.1).sum();
        if s >= 0.2 && s < 0.999 {
            lag_ref[v] = Some(crate::scene::normalize_vw(own));
        }
    }
    let stiff = stiff_parts(model, &vgroup, h);
    let sep = ctx.th.intersect_sep * model.scale;

    struct PoseOut {
        wrong: Vec<(u32, f64)>,
        lag: Vec<(u32, f64)>,
        stiff: Vec<(usize, f64)>,
        through: Vec<(u32, u32)>,
    }
    let per: Vec<PoseOut> = (0..ctx.poses.len())
        .into_par_iter()
        .map(|p| {
            let m = &ctx.mats[p];
            let posed = crate::skin::lbs(model, m, w);
            let wrong = (0..nv)
                .filter_map(|v| {
                    foreign[v].as_ref().map(|(home, _)| (v as u32, (crate::skin::lbs_vertex(m, home, model.rest[v]) - posed[v]).len() * k))
                })
                .filter(|x| x.1 > th.wrong_bone_m)
                .collect();
            let lag = (0..nv)
                .filter_map(|v| {
                    let r = lag_ref[v].as_ref()?;
                    let d = (crate::skin::lbs_vertex(m, r, model.rest[v]) - posed[v]).len();
                    let radius = ctx.region_radius[ctx.vregion[v] as usize];
                    (d > th.lag_radii * radius && d * k > th.lag_min_m).then_some((v as u32, d * k))
                })
                .collect();
            let stiff = stiff
                .iter()
                .enumerate()
                .filter(|(_, s)| motion || !s.clips_only)
                .filter_map(|(si, s)| {
                    let a: Vec<Vec3> = s.verts.iter().map(|&v| model.rest[v as usize]).collect();
                    let b: Vec<Vec3> = s.verts.iter().map(|&v| posed[v as usize]).collect();
                    let r = rigid_residual(&a, &b);
                    if std::env::var_os("WF_DEBUG_STIFF").is_some() {
                        eprintln!(
                            "stiff {} pose {} residual {:.2} cm ({:.1}% of size)",
                            s.name,
                            ctx.poses[p].name,
                            r * k * 100.0,
                            100.0 * r / s.size
                        );
                    }
                    (r > th.stiff_share * s.size && r * k > th.stiff_min_m).then_some((si, r * k))
                })
                .collect();
            let through = crate::metrics::intersect_pairs(model, &posed, sep)
                .into_iter()
                .filter(|&(a, b)| {
                    let (ga, gb) = (&vgroup[model.tris[a as usize][0] as usize], &vgroup[model.tris[b as usize][0] as usize]);
                    ga != gb && !(matches!(ga, Group::Torso | Group::Head) && matches!(gb, Group::Torso | Group::Head))
                })
                .collect();
            PoseOut { wrong, lag, stiff, through }
        })
        .collect();

    let pose_name = |p: usize| ctx.poses.get(p).map(|x| x.name.clone()).unwrap_or_default();
    let mut findings: Vec<Finding> = Vec::new();
    let push =
        |findings: &mut Vec<Finding>, kind: &str, code: &str, sev: &str, part: String, vs: Vec<u32>, worst: usize, detail: String| {
            let a = cm2(&vs);
            if vs.len() < th.min_verts || a < th.min_area_cm2 {
                return;
            }
            findings.push(Finding {
                kind: kind.into(),
                code: code.into(),
                severity: sev.into(),
                part,
                area_cm2: (a * 10.0).round() / 10.0,
                area_pct: (vs.iter().map(|&v| varea[v as usize]).sum::<f64>() / total_area * 1e4).round() / 100.0,
                verts: vs.len(),
                worst: pose_name(worst),
                detail,
                vertices: vs,
                worst_pose: worst,
            });
        };
    // Per-vertex worst over poses, for the two vertex-wise checks.
    let worst_of = |get: &dyn Fn(&PoseOut) -> &Vec<(u32, f64)>| {
        let mut best = vec![(0.0f64, 0usize); nv];
        for (p, po) in per.iter().enumerate() {
            for &(v, d) in get(po) {
                if d > best[v as usize].0 {
                    best[v as usize] = (d, p);
                }
            }
        }
        best
    };
    let wrong = worst_of(&|po| &po.wrong);
    // Wrong bone: areas per (region bone, foreign bone).
    let mut by: BTreeMap<(usize, u32), Vec<u32>> = BTreeMap::new();
    for v in 0..nv {
        if wrong[v].0 > 0.0 {
            by.entry((vbone[v], foreign[v].as_ref().unwrap().1)).or_default().push(v as u32);
        }
    }
    for ((rb, fb), vs) in by {
        let mut flag = vec![false; nv];
        for &v in &vs {
            flag[v as usize] = true;
        }
        for area in areas(model, &flag) {
            let (amt, worst) = area.iter().map(|&v| wrong[v as usize]).fold((0.0, 0), |a, b| if b.0 > a.0 { b } else { a });
            let part = format!("{}, {}", bone_label(model, rb), side_word(model, &area, rb));
            let detail =
                format!("moves with the {} (up to {:.0} cm from where its own bones put it)", bone_label(model, fb as usize), amt * 100.0);
            push(&mut findings, "wrong_bone", "D_WRONG_BONE", "fail", part, area, worst, detail);
        }
    }
    let lag = worst_of(&|po| &po.lag);
    let mut by: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for v in 0..nv {
        if lag[v].0 > 0.0 {
            by.entry(vbone[v]).or_default().push(v as u32);
        }
    }
    for (b, vs) in by {
        let mut flag = vec![false; nv];
        for &v in &vs {
            flag[v as usize] = true;
        }
        for area in areas(model, &flag) {
            let (amt, worst) = area.iter().map(|&v| lag[v as usize]).fold((0.0, 0), |a, b| if b.0 > a.0 { b } else { a });
            // What pulls it: the largest weight off its limb.
            let mut pull: BTreeMap<usize, f64> = BTreeMap::new();
            let chain: Vec<usize> = std::iter::once(b).chain(sk.jchildren[b].iter().copied()).collect();
            for &v in &area {
                for &(j, x) in &w[v as usize] {
                    let o = crate::helpers::owner(sk, j as usize);
                    if !chain.contains(&o) {
                        *pull.entry(o).or_default() += x;
                    }
                }
            }
            let by_whom = pull.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(j, _)| bone_label(model, *j)).unwrap_or_default();
            let lab = bone_label(model, b);
            let part = format!("{lab}, {}", side_word(model, &area, b));
            let detail = format!("does not follow the {lab}: held back by the {by_whom}, up to {:.0} cm off", amt * 100.0);
            push(&mut findings, "not_following", "D_NOT_FOLLOWING", "fail", part, area, worst, detail);
        }
    }
    // Stiff parts.
    let mut sbest: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
    for (p, po) in per.iter().enumerate() {
        for &(si, r) in &po.stiff {
            let e = sbest.entry(si).or_insert((0.0, p));
            if r > e.0 {
                *e = (r, p);
            }
        }
    }
    for (si, (r, p)) in sbest {
        let s = &stiff[si];
        let detail = format!("bends: up to {:.1} cm off moving as one piece ({:.0}% of its size)", r * 100.0, 100.0 * r / k / s.size);
        push(&mut findings, "stiff_bends", "D_STIFF_BENDS", "fail", s.name.clone(), s.verts.clone(), p, detail);
    }
    // Limb through limb: per pair of parts, the pose with the most area.
    let mut tbest: BTreeMap<(String, String), (f64, usize, Vec<u32>, String)> = BTreeMap::new();
    for (p, po) in per.iter().enumerate() {
        let mut pairs: BTreeMap<(String, String), (Vec<u32>, BTreeMap<usize, usize>)> = BTreeMap::new();
        for &(a, b) in &po.through {
            let (va, vb) = (model.tris[a as usize][0] as usize, model.tris[b as usize][0] as usize);
            let (ga, gb) = (vgroup[va].name(), vgroup[vb].name());
            let (key, first, second, tb) = if ga <= gb { ((ga, gb), a, b, vb) } else { ((gb, ga), b, a, va) };
            let e = pairs.entry(key).or_default();
            e.0.extend(model.tris[first as usize]);
            e.0.extend(model.tris[second as usize]);
            *e.1.entry(vbone[tb]).or_default() += 1;
        }
        for (key, (mut vs, into)) in pairs {
            vs.sort_unstable();
            vs.dedup();
            let a = cm2(&vs);
            let into_bone = into.iter().max_by_key(|x| x.1).map(|x| *x.0).unwrap_or(0);
            let e = tbest.entry(key).or_insert((0.0, p, Vec::new(), String::new()));
            if a > e.0 {
                *e = (a, p, vs, bone_label(model, into_bone));
            }
        }
    }
    for ((ga, gb), (a, p, vs, into)) in tbest {
        let sev = if motion { "fail" } else { "warn" };
        let part = format!(
            "{} and {}",
            ga.replace(".L", " (left)").replace(".R", " (right)"),
            gb.replace(".L", " (left)").replace(".R", " (right)")
        );
        let detail = format!("pass through each other: {a:.0} cm\u{b2} of skin inside the other part (into the {into})");
        push(&mut findings, "limb_through", "D_LIMB_THROUGH", sev, part, vs, p, detail);
    }
    // The checker's own findings of these poses, as areas.
    let mask = crate::report::finding_mask(ctx, ev, rep);
    for f in &rep.findings {
        let (kind, bit) = match f.code.as_str() {
            "D_STRETCH" => ("stretch", crate::metrics::F_STRETCH),
            "D_VOLUME" => ("volume", crate::metrics::F_THIN),
            "D_BLEED" => ("bleed", crate::metrics::F_BLEED),
            "D_NOISE" => ("noise", crate::metrics::F_NOISE),
            "D_UNWEIGHTED" => ("unweighted", crate::metrics::F_UNWEIGHTED),
            "D_PIECE_FOLLOW" => ("piece_follow", crate::metrics::F_FOLLOW),
            _ => continue,
        };
        // Weight-only findings belong to the poses' section, where every
        // part of the body is moved; the clips' section shows motion.
        if motion && matches!(kind, "bleed" | "unweighted") {
            continue;
        }
        let vs: Vec<u32> = (0..nv as u32)
            .filter(|&v| ctx.regions[ctx.vregion[v as usize] as usize].name == f.region && mask[v as usize] & bit != 0)
            .collect();
        if vs.is_empty() {
            continue;
        }
        let mut votes: BTreeMap<u32, usize> = BTreeMap::new();
        for &v in &vs {
            *votes.entry(ev.worst_pose[v as usize]).or_default() += 1;
        }
        let worst = votes.into_iter().max_by_key(|x| x.1).map(|x| x.0 as usize).unwrap_or(0);
        let label = ctx.regions.iter().find(|r| r.name == f.region).map(|r| r.label.clone()).unwrap_or_else(|| f.region.clone());
        let j = ctx.regions.iter().position(|r| r.name == f.region).and_then(|r| ctx.region_bone[r]);
        let part = match j {
            Some(j) => format!("{label}, {}", side_word(model, &vs, j)),
            None => label,
        };
        let detail = match kind {
            "stretch" => format!("skin stretches past {:.1}x", ctx.th.stretch),
            "volume" => "the limb collapses to under half its thickness".into(),
            _ => f.detail.clone(),
        };
        push(&mut findings, kind, &f.code, &f.severity, part, vs, worst, detail);
    }
    // Tears: the skin inside failing findings. A limb inside another is
    // reported beside it (`through_cm2`), not as a tear: it is as often
    // the clip's doing (a hand swung into the chest, a body lying on its
    // arm) as the weights'.
    let mut torn = vec![false; nv];
    let mut through = vec![false; nv];
    for f in findings.iter().filter(|f| f.kind == "limb_through") {
        for &v in &f.vertices {
            through[v as usize] = true;
        }
    }
    for f in findings.iter().filter(|f| f.severity == "fail" && f.kind != "limb_through") {
        for &v in &f.vertices {
            torn[v as usize] = true;
        }
    }
    let tears_area: f64 = (0..nv).filter(|&v| torn[v]).map(|v| varea[v]).sum();
    let tears_verts = torn.iter().filter(|&&t| t).count();
    findings.sort_by(|a, b| (a.severity != "fail", -a.area_cm2).partial_cmp(&(b.severity != "fail", -b.area_cm2)).unwrap());
    Section {
        name: name.into(),
        frames: ctx.poses.len(),
        clips,
        source,
        tears_pct: (tears_area / total_area * 1e4).round() / 100.0 + 0.0,
        tears_verts_pct: (tears_verts as f64 / nv.max(1) as f64 * 1e4).round() / 100.0 + 0.0,
        through_cm2: ((0..nv).filter(|&v| through[v]).map(|v| varea[v]).sum::<f64>() * k * k * 1e4).round(),
        findings,
    }
}

/// The text a person reads first.
pub fn text(sections: &[Section]) -> String {
    let mut s = String::new();
    for sec in sections {
        let what = if sec.clips.is_empty() {
            format!("{} ({})", sec.name, sec.frames)
        } else {
            format!("{} ({}, {} frames, from {})", sec.name, sec.clips.join(" "), sec.frames, sec.source)
        };
        s += &format!("{what}: torn {:.2}% of the surface; {:.0} cm\u{b2} of limbs inside other parts\n", sec.tears_pct, sec.through_cm2);
        for f in &sec.findings {
            let mark = if f.severity == "fail" { "" } else { " (warning)" };
            s += &format!(
                "  {}{mark}: {}: {} - {:.0} cm\u{b2}, worst in {}\n",
                f.kind.replace('_', " "),
                f.part,
                f.detail,
                f.area_cm2,
                f.worst
            );
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::v3;

    #[test]
    fn rigid_residual_is_zero_for_a_rigid_motion() {
        let a: Vec<Vec3> = (0..20).map(|i| v3(i as f64 * 0.1, (i * i) as f64 * 0.01, (i % 3) as f64 * 0.2)).collect();
        let q = Quat::from_axis_angle(v3(0.3, 1.0, 0.2), 0.8);
        let b: Vec<Vec3> = a.iter().map(|p| q.rotate(*p) + v3(1.0, 2.0, 3.0)).collect();
        assert!(rigid_residual(&a, &b) < 1e-9);
        let mut c = b.clone();
        c[5].y += 0.1;
        let r = rigid_residual(&a, &c);
        assert!(r > 0.05 && r < 0.1, "{r}");
    }

    #[test]
    fn stiff_words_are_whole_words() {
        assert!(words("Belt_Buckle.001").iter().any(|w| stiff_word(w)));
        assert!(words("pouches_L").iter().any(|w| stiff_word(w)));
        assert!(!words("Beltane_hair").iter().any(|w| stiff_word(w)));
    }
}

/// Vertices inside failing findings (limbs inside other parts aside) of
/// one pose group under weights `w`.
pub fn torn(model: &Model, ctx: &Ctx, w: &Weights, motion: bool) -> Vec<bool> {
    let ev = crate::metrics::evaluate_with(model, ctx, w, false);
    let rep = crate::report::build("guard", model, ctx, &ev);
    let sec = assess(model, ctx, &ev, &rep, w, motion, "", Vec::new(), String::new());
    let mut t = vec![false; model.nverts()];
    for f in sec.findings.iter().filter(|f| f.severity == "fail" && f.kind != "limb_through") {
        for &v in &f.vertices {
            t[v as usize] = true;
        }
    }
    t
}

/// A repair judged by what is seen: regions where `output` tears more
/// skin than `input` (on any pose group: the poses, the game's clips) get
/// the input's weights back, round after round; if the whole is still
/// worse, the input comes back. Returns the weights and the regions put
/// back.
pub fn guard(model: &Model, groups: &[(&Ctx, bool)], input: &Weights, output: &Weights) -> (Weights, Vec<String>) {
    let Some(&(rctx, _)) = groups.first() else { return (output.clone(), Vec::new()) };
    let area = vertex_areas(model);
    let k = BODY_M / height(model);
    let nr = rctx.regions.len();
    let per_region = |t: &[Vec<bool>]| -> Vec<f64> {
        let mut a = vec![0.0; nr];
        for v in 0..model.nverts() {
            if t.iter().any(|x| x[v]) {
                a[rctx.vregion[v] as usize] += area[v] * k * k * 1e4;
            }
        }
        a
    };
    let tin: Vec<Vec<bool>> = groups.iter().map(|&(c, m)| torn(model, c, input, m)).collect();
    let ain = per_region(&tin);
    let mut cur = output.clone();
    let mut back = vec![false; nr];
    for _ in 0..6 {
        let tc: Vec<Vec<bool>> = groups.iter().map(|&(c, m)| torn(model, c, &cur, m)).collect();
        let ac = per_region(&tc);
        let worse: Vec<usize> = (0..nr).filter(|&r| !back[r] && ac[r] > ain[r] + (0.1 * ain[r]).max(2.0)).collect();
        if worse.is_empty() {
            let total = |a: &[f64]| a.iter().sum::<f64>();
            if total(&ac) > total(&ain) + 2.0 {
                return (input.clone(), vec!["everything (the whole tore more than the input)".into()]);
            }
            break;
        }
        for &r in &worse {
            back[r] = true;
        }
        for v in 0..model.nverts() {
            if back[rctx.vregion[v] as usize] {
                cur[v] = input[v].clone();
            }
        }
    }
    let names = (0..nr).filter(|&r| back[r]).map(|r| rctx.regions[r].label.clone()).collect();
    (cur, names)
}
