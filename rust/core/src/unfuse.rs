//! Legs that share one shell: a tunic or coat skirt, baggy trousers, or
//! thighs a generator fused into one lump. Which leg a piece of skin goes
//! round is read from its outline (a horizontal cut through a vertex gives
//! a loop; it goes round the left leg's axis, the right's, both, or
//! neither). Skin round one leg only has no business following the other
//! leg's bones: `own_legs` moves such weight to this leg's own bones. On a
//! body whose lump was cut in two (meshkit `tools/unfuse.py`) that is
//! what lets the halves part; on an uncut lump there is nothing round one
//! leg alone up there, and nothing changes.

use crate::math::Vec3;
use crate::poses::resolve_roles;
use crate::scene::{Model, VW, Weights, normalize_vw, prune_vw};
use std::collections::HashMap;

/// The legs of a humanoid skeleton, as the drape needs them.
pub struct Legs {
    /// Midpoint of the two hip joints.
    pub center: Vec3,
    /// Unit vector from the right hip joint to the left.
    pub lateral: Vec3,
    /// Per joint: 1 left leg chain, 2 right leg chain, 0 neither (helpers
    /// count with their driver).
    pub side: Vec<u8>,
    /// Per leg joint: its counterpart on the other leg.
    pub mirror: Vec<Option<usize>>,
    /// The thighs' parent joint (the hips), if they share one.
    pub hips: Option<usize>,
    /// Length of the left thigh bone.
    pub thigh_len: f64,
    /// Leg axis as a polyline from the hip joint down (thigh, shin), per side.
    axis: [Vec<Vec3>; 2],
}

impl Legs {
    pub fn of(model: &Model) -> Option<Legs> {
        let sk = &model.skel;
        let roles = resolve_roles(sk);
        let first = |r: &str| roles.get(r).and_then(|v| v.first().copied());
        let (tl, tr) = (first("thigh.L")?, first("thigh.R")?);
        let nj = model.njoints();
        let side: Vec<u8> = (0..nj)
            .map(|j| {
                let mut k = Some(crate::helpers::owner(sk, j));
                while let Some(b) = k {
                    if b == tl {
                        return 1;
                    }
                    if b == tr {
                        return 2;
                    }
                    k = sk.jparent[b];
                }
                0
            })
            .collect();
        let center = (sk.head[tl] + sk.head[tr]) * 0.5;
        let across = sk.head[tl] - sk.head[tr];
        let lateral = Vec3 { y: 0.0, ..across }.normalized();
        // Counterparts: the other leg's joint whose head is nearest this
        // one's head reflected in the body's middle plane, helpers with
        // helpers.
        let reflect = |p: Vec3| p - lateral * (2.0 * (p - center).dot(lateral));
        let is_helper = |j: usize| crate::helpers::helper_of(sk, j).is_some();
        let mirror: Vec<Option<usize>> = (0..nj)
            .map(|j| {
                if side[j] == 0 {
                    return None;
                }
                let want = reflect(sk.head[j]);
                (0..nj)
                    .filter(|&k| side[k] != 0 && side[k] != side[j] && is_helper(k) == is_helper(j))
                    .min_by(|&a, &b| (sk.head[a] - want).len().total_cmp(&(sk.head[b] - want).len()).then(a.cmp(&b)))
            })
            .collect();
        let axis_of = |thigh: usize, shin: Option<usize>| -> Vec<Vec3> {
            let mut pts = vec![sk.head[thigh], sk.tail[thigh]];
            if let Some(s) = shin {
                pts.push(sk.tail[s]);
            }
            pts
        };
        let axis = [axis_of(tl, first("shin.L")), axis_of(tr, first("shin.R"))];
        let hips = sk.jparent[tl].filter(|&p| sk.jparent[tr] == Some(p));
        Some(Legs { center, lateral, side, mirror, hips, thigh_len: sk.bone_len(tl), axis })
    }

    /// Where a leg's axis crosses height `y` (x, z): straight up from the
    /// hip joint above it, straight down from the ankle below it.
    pub fn axis_at(&self, leg: usize, y: f64) -> (f64, f64) {
        let pts = &self.axis[leg];
        if y >= pts[0].y {
            return (pts[0].x, pts[0].z);
        }
        for s in pts.windows(2) {
            if y >= s[1].y && s[0].y > s[1].y {
                let t = (s[0].y - y) / (s[0].y - s[1].y);
                let p = s[0].lerp(s[1], t);
                return (p.x, p.z);
            }
        }
        let p = pts[pts.len() - 1];
        (p.x, p.z)
    }

    /// Distance of a hip joint from the middle plane.
    pub fn half_width(&self) -> f64 {
        0.5 * (self.axis[0][0] - self.axis[1][0]).len()
    }

    fn lowest(&self) -> f64 {
        self.axis.iter().map(|a| a[a.len() - 1].y).fold(f64::MAX, f64::min)
    }
}

/// Outlines of a mesh: the loops a plane cuts from it, walked triangle to
/// triangle. Horizontal planes by default (`new`); `along` takes any
/// direction.
pub struct Slicer<'a> {
    model: &'a Model,
    /// Per vertex: height along the slicing direction, and its two
    /// coordinates in the slicing plane.
    h: Vec<f64>,
    c: Vec<(f64, f64)>,
    edge_tris: HashMap<(u32, u32), Vec<u32>>,
    vert_tris: Vec<Vec<u32>>,
    seen: Vec<u32>,
    stamp: u32,
}

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

impl<'a> Slicer<'a> {
    /// Horizontal outlines: height is y, the points are (x, z).
    pub fn new(model: &'a Model) -> Slicer<'a> {
        let (x, y, z) = (Vec3 { x: 1.0, y: 0.0, z: 0.0 }, Vec3 { x: 0.0, y: 1.0, z: 0.0 }, Vec3 { x: 0.0, y: 0.0, z: 1.0 });
        Slicer::along(model, y, x, z)
    }

    /// Outlines in planes across `dir`; points are (p . e1, p . e2).
    pub fn along(model: &'a Model, dir: Vec3, e1: Vec3, e2: Vec3) -> Slicer<'a> {
        let mut edge_tris: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        let mut vert_tris: Vec<Vec<u32>> = vec![Vec::new(); model.nverts()];
        for (ti, t) in model.tris.iter().enumerate() {
            for k in 0..3 {
                edge_tris.entry(edge_key(t[k], t[(k + 1) % 3])).or_default().push(ti as u32);
                vert_tris[t[k] as usize].push(ti as u32);
            }
        }
        let h = model.rest.iter().map(|p| p.dot(dir)).collect();
        let c = model.rest.iter().map(|p| (p.dot(e1), p.dot(e2))).collect();
        Slicer { model, h, c, edge_tris, vert_tris, seen: vec![0; model.tris.len()], stamp: 0 }
    }

    /// The two edges of a triangle that cross height y, if it crosses.
    fn crossing(&self, ti: u32, y: f64) -> Option<[(u32, u32); 2]> {
        let t = self.model.tris[ti as usize];
        let up = [self.h[t[0] as usize] > y, self.h[t[1] as usize] > y, self.h[t[2] as usize] > y];
        let mut out = [(0u32, 0u32); 2];
        let mut n = 0;
        for k in 0..3 {
            if up[k] != up[(k + 1) % 3] {
                if n == 2 {
                    return None;
                }
                out[n] = edge_key(t[k], t[(k + 1) % 3]);
                n += 1;
            }
        }
        (n == 2).then_some(out)
    }

    fn point(&self, e: (u32, u32), y: f64) -> (f64, f64) {
        let (ha, hb) = (self.h[e.0 as usize], self.h[e.1 as usize]);
        let (a, b) = (self.c[e.0 as usize], self.c[e.1 as usize]);
        let t = if (hb - ha).abs() > 1e-15 { ((y - ha) / (hb - ha)).clamp(0.0, 1.0) } else { 0.5 };
        (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
    }

    /// The outline through triangle `t0` at height `y` as (x, z) points
    /// and the triangles it crosses, and whether it closes. Marks its
    /// triangles seen under the current stamp.
    fn walk(&mut self, t0: u32, y: f64) -> (Vec<(f64, f64)>, Vec<u32>, bool) {
        let ends = self.crossing(t0, y).expect("a crossing triangle");
        self.seen[t0 as usize] = self.stamp;
        let mut pts: Vec<(f64, f64)> = Vec::new();
        let mut tris: Vec<u32> = vec![t0];
        let mut closed = false;
        for (dir, &first) in ends.iter().enumerate() {
            let mut run = Vec::new();
            let mut run_t = Vec::new();
            let (mut cur, mut e) = (t0, first);
            loop {
                run.push(self.point(e, y));
                let next = self.edge_tris[&e]
                    .iter()
                    .copied()
                    .find(|&n| n != cur && self.seen[n as usize] != self.stamp && self.crossing(n, y).is_some());
                let Some(n) = next else {
                    // Back at the start triangle: the loop is closed.
                    if dir == 0 && cur != t0 && self.edge_tris[&e].contains(&t0) {
                        closed = true;
                    }
                    break;
                };
                self.seen[n as usize] = self.stamp;
                run_t.push(n);
                let c = self.crossing(n, y).unwrap();
                e = if c[0] == e { c[1] } else { c[0] };
                cur = n;
            }
            if dir == 0 {
                pts = run;
                tris.extend(run_t);
            } else if !closed {
                run.reverse();
                run.extend(pts);
                pts = run;
                tris.extend(run_t);
            }
            if closed {
                break;
            }
        }
        (pts, tris, closed)
    }

    /// Every outline at height `y`: points, and whether it closes.
    pub fn outlines(&mut self, y: f64) -> Vec<(Vec<(f64, f64)>, bool)> {
        self.stamp += 1;
        let mut out = Vec::new();
        for ti in 0..self.model.tris.len() as u32 {
            if self.seen[ti as usize] != self.stamp && self.crossing(ti, y).is_some() {
                let (pts, _, closed) = self.walk(ti, y);
                out.push((pts, closed));
            }
        }
        out
    }

    /// The outline through vertex `v` (just above it, else just below).
    pub fn through(&mut self, v: usize) -> Option<(Vec<(f64, f64)>, bool, f64)> {
        let eps = 1e-5 * self.model.scale;
        let py = self.h[v];
        for y in [py + eps, py - eps] {
            if let Some(&ti) = self.vert_tris[v].iter().find(|&&ti| self.crossing(ti, y).is_some()) {
                self.stamp += 1;
                let (pts, _, closed) = self.walk(ti, y);
                return Some((pts, closed, y));
            }
        }
        None
    }
}

/// How far an outline turns round a point (radians, unsigned): about
/// 6.28 for a closed loop round it, 0 for one that leaves it outside.
pub fn winding(pts: &[(f64, f64)], closed: bool, c: (f64, f64)) -> f64 {
    let n = pts.len();
    if n < 2 {
        return 0.0;
    }
    let last = if closed { n } else { n - 1 };
    let mut sum = 0.0;
    for i in 0..last {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        let (ax, az, bx, bz) = (a.0 - c.0, a.1 - c.1, b.0 - c.0, b.1 - c.1);
        sum += (ax * bz - az * bx).atan2(ax * bx + az * bz);
    }
    sum.abs()
}

/// An outline counts as going round a point when it turns this far: an
/// open one (a hole in the mesh) still counts when it goes most of the way.
pub const ROUND: f64 = 1.5 * std::f64::consts::PI;

/// Which leg axes the body's outline through each vertex goes round: bit 1
/// the left, bit 2 the right (3: a shell round both legs). A vertex below
/// the ankles or with no outline through it gets 0.
pub fn outline_round(model: &Model, legs: &Legs) -> Vec<u8> {
    let mut sl = Slicer::new(model);
    let floor = legs.lowest();
    (0..model.nverts())
        .map(|v| {
            if model.rest[v].y < floor {
                return 0;
            }
            let Some((pts, closed, y)) = sl.through(v) else { return 0 };
            if pts.len() < 3 {
                return 0;
            }
            let mut bits = 0u8;
            for leg in 0..2 {
                if winding(&pts, closed, legs.axis_at(leg, y)) > ROUND {
                    bits |= 1 << leg;
                }
            }
            bits
        })
        .collect()
}

/// Which side of the body's middle plane the outline through each vertex
/// lies on: 1 wholly on the left leg's side, 2 wholly on the right's, 0
/// when it reaches across (the trunk, an uncut skirt), or for a vertex
/// above the hip joints or below the ankles. An outline may reach
/// `SIDE_SLACK` of the figure's height across the plane and still count:
/// a cut made beside the plane (meshkit `unfuse.py` slides it clear of
/// vertices) is still found.
pub fn outline_side(model: &Model, legs: &Legs) -> Vec<u8> {
    let mut sl = Slicer::new(model);
    let floor = legs.lowest();
    let (lo, hi) = model.rest.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.y), a.1.max(p.y)));
    let slack = SIDE_SLACK * (hi - lo);
    let lat = Vec3 { y: 0.0, ..legs.lateral }.normalized();
    let mut side: Vec<u8> = (0..model.nverts())
        .map(|v| {
            let y = model.rest[v].y;
            if y < floor || y > legs.center.y {
                return 0;
            }
            let Some((pts, _, _)) = sl.through(v) else { return 0 };
            let (mut min, mut max) = (f64::MAX, f64::MIN);
            for p in &pts {
                let x = (p.0 - legs.center.x) * lat.x + (p.1 - legs.center.z) * lat.z;
                min = min.min(x);
                max = max.max(x);
            }
            if min > -slack && max > slack {
                1
            } else if max < slack && min < -slack {
                2
            } else {
                0
            }
        })
        .collect();
    // A vertex on the cut whose own outline is too small to tell (a
    // corner of an inner wall) takes the side of the vertices it joins,
    // when they agree; the ends of a cut join both and stay 0.
    for _ in 0..4 {
        let prev = side.clone();
        for v in 0..model.nverts() {
            let q = model.rest[v] - legs.center;
            if prev[v] != 0 || q.y > 0.0 || model.rest[v].y < floor || q.dot(lat).abs() > slack {
                continue;
            }
            let (mut l, mut r) = (false, false);
            for &u in &model.adj[v] {
                l |= prev[u as usize] == 1;
                r |= prev[u as usize] == 2;
            }
            if l != r {
                side[v] = if l { 1 } else { 2 };
            }
        }
    }
    side
}

pub const SIDE_SLACK: f64 = 0.01;

/// Gives skin on one leg's side to that leg: where the outline through a
/// vertex lies wholly on one side of the middle plane (`outline_side`),
/// weight on the other leg's bones moves to their counterparts on this
/// one (thigh to thigh, shin to shin, helper to helper). Hips and spine
/// weights stay, and so does skin whose outline reaches across.
pub fn own_legs(model: &Model, w: &Weights) -> Weights {
    let Some(legs) = Legs::of(model) else { return w.clone() };
    let side = outline_side(model, &legs);
    own_legs_with(model, &legs, &side, w)
}

/// `own_legs` with the legs and sides already found.
pub fn own_legs_with(model: &Model, legs: &Legs, side: &[u8], w: &Weights) -> Weights {
    (0..model.nverts())
        .map(|v| {
            let other = match side[v] {
                1 => 2u8,
                2 => 1u8,
                _ => return w[v].clone(),
            };
            if !w[v].iter().any(|&(j, x)| x > 0.0 && legs.side[j as usize] == other) {
                return w[v].clone();
            }
            let mut out: VW = Vec::new();
            for &(j, x) in &w[v] {
                let to = match legs.mirror[j as usize] {
                    Some(m) if legs.side[j as usize] == other => m as u32,
                    _ => j,
                };
                match out.iter_mut().find(|e| e.0 == to) {
                    Some(e) => e.1 += x,
                    None => out.push((to, x)),
                }
            }
            prune_vw(&normalize_vw(out), 4, 0.01)
        })
        .collect()
}

/// How far below the crotch a leg's inner side takes to become all thigh,
/// in thigh lengths, and how far out from the middle plane skin above the
/// crotch does, in hip half-widths (the hip joint's distance from the
/// plane).
pub const BAND_DOWN: f64 = 0.3;
pub const BAND_OUT: f64 = 1.5;

/// The height of the crotch `unfuse.py` cut the legs apart up to, as it
/// records it in the mesh's `extras.unfuse.legs.crotch`. `None` on a body
/// it did not cut.
pub fn cut_crotch(model: &Model) -> Option<f64> {
    model.glb.arr("meshes").iter().find_map(|m| m.get("extras")?.get("unfuse")?.get("legs")?.get("crotch")?.as_f64())
}

/// The weights to start a repair from on a body `unfuse.py` cut: the
/// cut leaves each half with weights interpolated across it, naming both
/// thighs, and `own_legs` is what they should have been. `None` on a body
/// that was not cut, or when nothing changes.
pub fn cut_start(model: &Model) -> Option<Weights> {
    let mut w = if cut_crotch(model).is_some() { own_legs(model, &model.weights) } else { model.weights.clone() };
    for cut in arm_cuts(model) {
        w = arm_cut_weights(model, &cut, &w);
    }
    (w != model.weights).then_some(w)
}

/// Weights for the crotch of a body whose legs `unfuse.py` cut apart:
/// first `own_legs`; then skin near the middle plane follows the hips at
/// and above the crotch, and takes up its leg over `BAND_DOWN` thigh
/// lengths below it or `BAND_OUT` hip half-widths out from the plane,
/// whichever lets it have more. Leg weight is only ever capped, never
/// added, and what is taken goes to the thighs' parent. Without this the
/// skin above the cut still splits hard between the two thighs and tears
/// at the cut's top. A body that was not cut comes back unchanged: its
/// crotch is where its rig expects it.
pub fn crotch_band(model: &Model, w: &Weights) -> Weights {
    crotch_band_with(model, w, BAND_DOWN, BAND_OUT)
}

/// `crotch_band` with its two widths given.
pub fn crotch_band_with(model: &Model, w: &Weights, band_down: f64, band_out: f64) -> Weights {
    let (Some(legs), Some(crotch)) = (Legs::of(model), cut_crotch(model)) else { return w.clone() };
    let side = outline_side(model, &legs);
    let w = own_legs_with(model, &legs, &side, w);
    let Some(hips) = legs.hips else { return w };
    let lat = Vec3 { y: 0.0, ..legs.lateral }.normalized();
    let (down, out) = (band_down * legs.thigh_len, band_out * legs.half_width());
    let smooth = |x: f64| {
        let t = x.clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    (0..model.nverts())
        .map(|v| {
            let leg: f64 = w[v].iter().filter(|e| legs.side[e.0 as usize] != 0).map(|e| e.1).sum();
            if leg <= 0.0 {
                return w[v].clone();
            }
            let q = model.rest[v] - legs.center;
            let limit = smooth((crotch - model.rest[v].y) / down).max(smooth(q.dot(lat).abs() / out));
            if leg <= limit {
                return w[v].clone();
            }
            let k = limit / leg;
            let mut out: VW = Vec::new();
            let mut freed = 0.0;
            for &(j, x) in &w[v] {
                if legs.side[j as usize] != 0 {
                    freed += x * (1.0 - k);
                    if x * k > 0.0 {
                        out.push((j, x * k));
                    }
                } else {
                    out.push((j, x));
                }
            }
            match out.iter_mut().find(|e| e.0 == hips as u32) {
                Some(e) => e.1 += freed,
                None => out.push((hips as u32, freed)),
            }
            prune_vw(&normalize_vw(out), 4, 0.01)
        })
        .collect()
}

/// An arm that `unfuse.py` cut free of the body where a sleeve was joined
/// to it, as recorded in the mesh's `extras.unfuse` (`arm.l`, `arm.r`: the
/// facing of the cuts, from the body toward the arm).
pub struct ArmCut {
    /// Per joint: in this arm's chain (helpers count with their driver).
    pub chain: Vec<bool>,
    /// Unit vector across the cuts, from the body's side to the arm's.
    pub normal: Vec3,
    /// Length of the upper arm bone.
    pub len: f64,
    /// +1 for the left arm, -1 for the right.
    pub lateral_sign: f64,
}

pub fn arm_cuts(model: &Model) -> Vec<ArmCut> {
    let sk = &model.skel;
    let roles = resolve_roles(sk);
    let vec = |v: Option<&serde_json::Value>| -> Option<Vec3> {
        let a = v?.as_array()?;
        Some(Vec3 { x: a.first()?.as_f64()?, y: a.get(1)?.as_f64()?, z: a.get(2)?.as_f64()? })
    };
    let mut out = Vec::new();
    for mesh in model.glb.arr("meshes") {
        let Some(rec) = mesh.get("extras").and_then(|e| e.get("unfuse")) else { continue };
        for (key, role, sign) in [("arm.l", "upper_arm.L", 1.0), ("arm.r", "upper_arm.R", -1.0)] {
            let (Some(r), Some(&root)) = (rec.get(key), roles.get(role).and_then(|v| v.first())) else { continue };
            let Some(normal) = vec(r.get("normal")) else { continue };
            let chain: Vec<bool> = (0..model.njoints())
                .map(|j| {
                    let mut k = Some(crate::helpers::owner(sk, j));
                    while let Some(b) = k {
                        if b == root {
                            return true;
                        }
                        k = sk.jparent[b];
                    }
                    false
                })
                .collect();
            out.push(ArmCut { chain, normal: normal.normalized(), len: sk.bone_len(root), lateral_sign: sign });
        }
    }
    out
}

/// The two sides of every cut: pairs of vertices a hair apart (the two
/// rims `unfuse.py` leaves facing each other; nothing else in a welded
/// mesh sits that close), as (vertex, vertex).
pub fn cut_pairs(model: &Model) -> Vec<(usize, usize)> {
    let (lo, hi) = model.rest.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.y), a.1.max(p.y)));
    let reach = 1e-4 * (hi - lo);
    let cell = |p: Vec3| ((p.x / reach).floor() as i64, (p.y / reach).floor() as i64, (p.z / reach).floor() as i64);
    let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    for (v, p) in model.rest.iter().enumerate() {
        grid.entry(cell(*p)).or_default().push(v as u32);
    }
    let mut out = Vec::new();
    for (v, p) in model.rest.iter().enumerate() {
        let c = cell(*p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &u in grid.get(&(c.0 + dx, c.1 + dy, c.2 + dz)).into_iter().flatten() {
                        let u = u as usize;
                        if u > v && (model.rest[u] - *p).len() < reach && !model.adj[v].contains(&(u as u32)) {
                            out.push((v, u));
                        }
                    }
                }
            }
        }
    }
    out
}

/// How far from a sleeve's cut the weights are put right, in upper-arm
/// lengths along the skin.
pub const ARM_REACH: f64 = 0.6;

/// Weights round a cut that freed a sleeve from the body. A cuff pressed
/// into a coat was rigged as one skin: forearm weight runs onto the coat
/// and hips and thigh weight onto the sleeve. From the cut's two rims
/// outward along the skin, each side gives up the other's bones: the
/// sleeve keeps only its arm's, the coat none of them, in full at the cut
/// and fading out over `ARM_REACH`, and not at all near an end of a cut
/// that stops inside joined skin. A side with next to nothing of its own
/// takes the nearest bone of its own kind.
pub fn arm_cut_weights(model: &Model, cut: &ArmCut, w: &Weights) -> Weights {
    let sk = &model.skel;
    let legs = Legs::of(model);
    let nv = model.nverts();
    // Rims of this arm's cuts: pairs set apart along its facing, on its
    // side of the body, away from the legs' middle plane.
    let mut seeds = [vec![false; nv], vec![false; nv]]; // arm's side, body's side
    for (a, b) in cut_pairs(model) {
        let d = model.rest[a] - model.rest[b];
        let facing = d.normalized().dot(cut.normal);
        let side_x = legs.as_ref().map(|l| (model.rest[a] - l.center).dot(l.lateral)).unwrap_or(model.rest[a].x);
        if facing.abs() < 0.5 || side_x * cut.lateral_sign < 0.02 * model.scale {
            continue;
        }
        let (arm, body) = if facing > 0.0 { (a, b) } else { (b, a) };
        seeds[0][arm] = true;
        seeds[1][body] = true;
    }
    if !seeds[0].iter().any(|&s| s) {
        return w.clone();
    }
    let reach = ARM_REACH * cut.len;
    let dist = [crate::fix::distances(model, &seeds[0], reach), crate::fix::distances(model, &seeds[1], reach)];
    // Where a cut ends inside joined skin (a vertex that still joins a rim
    // of each side), the two sides are one skin again: near such an end
    // the weights are left as they were, so nothing switches hard across
    // skin that was not cut.
    let ends: Vec<bool> = (0..nv)
        .map(|v| {
            !seeds[0][v]
                && !seeds[1][v]
                && model.adj[v].iter().any(|&u| seeds[0][u as usize])
                && model.adj[v].iter().any(|&u| seeds[1][u as usize])
        })
        .collect();
    let to_end = crate::fix::distances(model, &ends, reach);
    let deform: Vec<bool> = {
        let mut used = vec![false; model.njoints()];
        for vw in w {
            for &(j, x) in vw {
                used[j as usize] |= x > 0.01;
            }
        }
        used
    };
    let nearest = |v: usize, arm: bool| -> Option<u32> {
        (0..model.njoints())
            .filter(|&j| deform[j] && cut.chain[j] == arm && crate::helpers::helper_of(sk, j).is_none())
            .min_by(|&a, &b| sk.seg_dist(a, model.rest[v]).total_cmp(&sk.seg_dist(b, model.rest[v])).then(a.cmp(&b)))
            .map(|j| j as u32)
    };
    let smooth = |x: f64| {
        let t = x.clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    (0..nv)
        .map(|v| {
            let (da, db) = (dist[0][v], dist[1][v]);
            if !da.is_finite() && !db.is_finite() {
                return w[v].clone();
            }
            let on_arm = da <= db;
            // How much of the other side's weight goes: all at the cut, none at the reach.
            let k = smooth(1.0 - da.min(db) / reach) * if to_end[v].is_finite() { smooth(to_end[v] / reach) } else { 1.0 };
            let mine = |j: u32| cut.chain[j as usize] == on_arm;
            let keep: f64 = w[v].iter().filter(|e| mine(e.0)).map(|e| e.1).sum();
            let lose: f64 = w[v].iter().filter(|e| !mine(e.0)).map(|e| e.1).sum();
            if lose <= 0.0 || k <= 0.0 {
                return w[v].clone();
            }
            let moved = lose * k;
            let mut out: VW = w[v].iter().map(|&(j, x)| if mine(j) { (j, x) } else { (j, x * (1.0 - k)) }).collect();
            // (A sliver of its own kind is no guide: scaled up it puts a
            // coat's flank on the collar bone.)
            if keep >= 0.15 {
                for e in out.iter_mut().filter(|e| mine(e.0)) {
                    e.1 += moved * e.1 / keep;
                }
            } else if let Some(j) = nearest(v, on_arm) {
                out.push((j, moved));
            } else {
                return w[v].clone();
            }
            prune_vw(&normalize_vw(out), 4, 0.01)
        })
        .collect()
}

/// Every limb `unfuse.py` cut apart: `crotch_band` for the legs,
/// `arm_cut_weights` for each arm. A body it did not cut comes back unchanged.
pub fn limbs_apart(model: &Model, w: &Weights) -> Weights {
    limbs_apart_with(model, w, BAND_DOWN, BAND_OUT)
}

pub fn limbs_apart_with(model: &Model, w: &Weights, band_down: f64, band_out: f64) -> Weights {
    let mut w = crotch_band_with(model, w, band_down, band_out);
    for cut in arm_cuts(model) {
        w = arm_cut_weights(model, &cut, &w);
    }
    w
}

/// An arm, for telling its sleeve from the body it hangs beside.
pub struct Arm {
    /// Per joint: in this arm's chain (helpers count with their driver).
    pub chain: Vec<bool>,
    /// Per joint: the forearm, the hand and what hangs from them.
    pub lower: Vec<bool>,
    /// Shoulder joint, elbow, wrist.
    pub axis: [Vec3; 3],
    /// Unit vectors: along the upper arm, across it toward the trunk, and
    /// the third (forward as near as the arm allows).
    pub dir: Vec3,
    pub toward: Vec3,
    pub fwd: Vec3,
    /// Length of the upper arm bone.
    pub len: f64,
}

pub fn arms_of(model: &Model) -> Vec<Arm> {
    let sk = &model.skel;
    let roles = resolve_roles(sk);
    let first = |r: &str| roles.get(r).and_then(|v| v.first().copied());
    let under = |root: usize| -> Vec<bool> {
        (0..model.njoints())
            .map(|j| {
                let mut k = Some(crate::helpers::owner(sk, j));
                while let Some(b) = k {
                    if b == root {
                        return true;
                    }
                    k = sk.jparent[b];
                }
                false
            })
            .collect()
    };
    let mid = model.rest.iter().fold(0.0, |a, p| a + p.x) / model.nverts().max(1) as f64;
    let mut out = Vec::new();
    for side in ["L", "R"] {
        let (Some(up), Some(fore)) = (first(&format!("upper_arm.{side}")), first(&format!("forearm.{side}"))) else { continue };
        let wrist = first(&format!("hand.{side}")).map(|h| sk.head[h]).unwrap_or(sk.tail[fore]);
        let axis = [sk.head[up], sk.head[fore], wrist];
        let dir = (axis[1] - axis[0]).normalized();
        let z = Vec3 { x: 0.0, y: 0.0, z: 1.0 };
        let fwd = (z - dir * z.dot(dir)).normalized();
        let mut toward = dir.cross(fwd).normalized();
        if toward.x * (mid - axis[0].x) < 0.0 {
            toward = -toward;
        }
        out.push(Arm { chain: under(up), lower: under(fore), axis, dir, toward, fwd, len: sk.bone_len(up) });
    }
    out
}

/// What a vertex beside or on an arm is, from the outline across the arm
/// through it. 1: sleeve (the outline goes round the arm's axis and is
/// the arm alone, or runs on round the trunk and the vertex is on the
/// arm's side of the waist between). 3: coat (the trunk's side of that
/// waist). 2: body (a closed outline that does not go round the arm).
/// 0: not beside the arm (above half way down the upper arm, past the
/// wrist), or no telling (the arm is sunk in the body with no waist;
/// with `by_weights` such a vertex goes with the bones it mostly follows).
pub fn sleeve_labels(model: &Model, arm: &Arm, w: &Weights, by_weights: bool) -> Vec<u8> {
    let mut sl = Slicer::along(model, arm.dir, arm.toward, arm.fwd);
    let h: Vec<f64> = arm.axis.iter().map(|p| p.dot(arm.dir)).collect();
    let axis_at = |y: f64| -> Option<(f64, f64)> {
        for k in 0..2 {
            if y >= h[k] && y <= h[k + 1] && h[k + 1] > h[k] {
                let p = arm.axis[k].lerp(arm.axis[k + 1], (y - h[k]) / (h[k + 1] - h[k]));
                return Some((p.dot(arm.toward), p.dot(arm.fwd)));
            }
        }
        None
    };
    let width = 0.04 * arm.len;
    (0..model.nverts())
        .map(|v| {
            let u = model.rest[v].dot(arm.dir);
            if u < h[0] + 0.5 * arm.len {
                return 0;
            }
            let Some((pts, closed, y)) = sl.through(v) else { return 0 };
            let Some(a) = axis_at(y) else { return 0 };
            if pts.len() < 8 {
                return 0;
            }
            if winding(&pts, closed, a) < ROUND {
                return if closed { 2 } else { 0 };
            }
            // Thickness (front to back) of the outline by distance from the
            // axis toward the trunk, in steps of `width`.
            let mut span: std::collections::BTreeMap<i64, (f64, f64)> = std::collections::BTreeMap::new();
            let n = pts.len();
            for i in 0..(if closed { n } else { n - 1 }) {
                let (p, q) = (pts[i], pts[(i + 1) % n]);
                for s in 0..8 {
                    let f = s as f64 / 8.0;
                    let (t, x) = (p.0 + (q.0 - p.0) * f - a.0, p.1 + (q.1 - p.1) * f);
                    let e = span.entry((t / width).floor() as i64).or_insert((x, x));
                    e.0 = e.0.min(x);
                    e.1 = e.1.max(x);
                }
            }
            let thick = |k: i64| span.get(&k).map_or(0.0, |e| e.1 - e.0);
            let far = *span.keys().next_back().unwrap();
            if (far as f64) * width < 0.9 * arm.len {
                return 1; // the arm alone
            }
            let lobe = thick(-1).max(thick(0)).max(thick(1));
            let mut best = 1i64;
            for k in 2..=far {
                if thick(k) < thick(best) {
                    best = k;
                } else if thick(k) > (1.3 * thick(best)).max(0.5 * lobe) {
                    break;
                }
            }
            if thick(best) < 0.6 * lobe {
                let t = model.rest[v].dot(arm.toward) - a.0;
                return if t < (best as f64 + 0.5) * width { 1 } else { 3 };
            }
            if by_weights {
                let on_arm: f64 = w[v].iter().filter(|e| arm.chain[e.0 as usize]).map(|e| e.1).sum();
                return if on_arm >= 0.5 { 1 } else { 3 };
            }
            0
        })
        .collect()
}

/// Re-weights a sleeve joined to the body, with no cut: sleeve skin
/// follows its arm's bones only, and the coat it is joined to gives up
/// what it carries of the forearm and hand (other skin beside the arm is
/// left alone: a hand's or a cuff's own outline does not go round the
/// arm's axis either). The join itself stays one
/// skin, so the edges across its waist take all the stretch.
pub fn sleeve_weights(model: &Model, w: &Weights, by_weights: bool) -> Weights {
    let sk = &model.skel;
    let mut w = w.clone();
    for arm in arms_of(model) {
        let labels = sleeve_labels(model, &arm, &w, by_weights);
        let deform: Vec<bool> = {
            let mut used = vec![false; model.njoints()];
            for vw in &w {
                for &(j, x) in vw {
                    used[j as usize] |= x > 0.01;
                }
            }
            used
        };
        let root = (0..model.njoints()).find(|&j| arm.chain[j] && sk.jparent[j].is_some_and(|p| !arm.chain[p]));
        let near_parents: Vec<usize> = {
            let p1 = root.and_then(|r| sk.jparent[r]);
            let p2 = p1.and_then(|p| sk.jparent[p]);
            p1.into_iter().chain(p2).collect()
        };
        w = (0..model.nverts())
            .map(|v| {
                // Which bones go: on a sleeve all but the arm's; on coat and
                // body the forearm's and hand's.
                // (The shoulder's blend into the chest is the rig's own: a
                // sleeve keeps its arm's two nearest ancestors, and nothing
                // changes above 0.8 of the way down the upper arm.)
                let goes = |j: u32| match labels[v] {
                    _ if (model.rest[v] - arm.axis[0]).dot(arm.dir) < 0.8 * arm.len => false,
                    1 => !arm.chain[j as usize] && !near_parents.contains(&(j as usize)),
                    3 => arm.lower[j as usize],
                    _ => false,
                };
                let lose: f64 = w[v].iter().filter(|e| goes(e.0)).map(|e| e.1).sum();
                if lose <= 0.0 {
                    return w[v].clone();
                }
                let keep = 1.0 - lose;
                let mut out: VW = w[v].iter().filter(|e| !goes(e.0)).copied().collect();
                if keep >= 0.15 {
                    out.iter_mut().for_each(|e| e.1 /= keep);
                } else {
                    let want_arm = labels[v] == 1;
                    let near = (0..model.njoints())
                        .filter(|&j| deform[j] && arm.chain[j] == want_arm && crate::helpers::helper_of(sk, j).is_none())
                        .min_by(|&a, &b| sk.seg_dist(a, model.rest[v]).total_cmp(&sk.seg_dist(b, model.rest[v])).then(a.cmp(&b)));
                    match near {
                        Some(j) => out.push((j as u32, lose)),
                        None => return w[v].clone(),
                    }
                }
                prune_vw(&normalize_vw(out), 4, 0.01)
            })
            .collect();
    }
    w
}

/// The coat under a sleeve follows the body: skin whose nearest bone
/// through the body (the checker's region) is not an arm's, nor one of
/// the arm's two nearest ancestors, gives up what it carries of that
/// arm's forearm and hand. A wide sleeve resting on a coat is rigged as
/// one skin with it, and forearm weight runs far down the coat's skirt;
/// that is cloth moving with the wrong bone.
pub fn coat_weights(model: &Model, ctx: &crate::metrics::Ctx, w: &Weights) -> Weights {
    let sk = &model.skel;
    let mut w = w.clone();
    for arm in arms_of(model) {
        let root = (0..model.njoints()).find(|&j| arm.chain[j] && sk.jparent[j].is_some_and(|p| !arm.chain[p]));
        let p1 = root.and_then(|r| sk.jparent[r]);
        let p2 = p1.and_then(|p| sk.jparent[p]);
        w = (0..model.nverts())
            .map(|v| {
                let Some(region) = ctx.region_bone[ctx.vregion[v] as usize] else { return w[v].clone() };
                if arm.chain[region] || Some(region) == p1 || Some(region) == p2 {
                    return w[v].clone();
                }
                let lose: f64 = w[v].iter().filter(|e| arm.lower[e.0 as usize]).map(|e| e.1).sum();
                if lose <= 0.0 {
                    return w[v].clone();
                }
                let mut out: VW = w[v].iter().filter(|e| !arm.lower[e.0 as usize]).copied().collect();
                if 1.0 - lose >= 0.15 {
                    out.iter_mut().for_each(|e| e.1 /= 1.0 - lose);
                } else {
                    out.push((region as u32, lose));
                }
                prune_vw(&normalize_vw(out), 4, 0.01)
            })
            .collect();
    }
    w
}
