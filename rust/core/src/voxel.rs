//! Geodesic distance from each bone to each vertex, measured through a
//! voxelization of the mesh (Dionne & de Lasa 2013, "Geodesic Voxel
//! Binding for Production Character Meshes"). Paths stay inside the body,
//! so a hand hanging next to a thigh is far from the thigh bone even though
//! it is close in a straight line. Works on multi-shell, non-manifold,
//! self-intersecting meshes because it only rasterizes triangles.
//!
//! The inside is what a flood from outside cannot reach, plus what a vote
//! of rays calls inside: generated meshes keep small openings, and one is
//! enough for the flood to fill the whole body, leaving only the shell
//! solid (a bone's distance then became a walk along the skin from one
//! seed). Six half-rays along the axes reach each voxel from outside the
//! grid; one says "inside" when it crossed some shell an odd number of
//! times. A small hole fools only the rays that pass through it, so four
//! of six decide. Shells count apart, so overlapping shells (an arm tube
//! through the torso) and flipped faces stay right.

use crate::math::{Vec3, segment_distance, v3};
use crate::scene::Model;
use rayon::prelude::*;
use serde::Serialize;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub struct Grid {
    pub n: [usize; 3],
    pub origin: Vec3,
    pub h: f64,
    /// 0 = empty, 1 = surface, 2 = interior.
    pub cell: Vec<u8>,
}

impl Grid {
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.n[1] + y) * self.n[0] + x
    }
    pub fn coord(&self, p: Vec3) -> [f64; 3] {
        [(p.x - self.origin.x) / self.h - 0.5, (p.y - self.origin.y) / self.h - 0.5, (p.z - self.origin.z) / self.h - 0.5]
    }
    pub fn cell_of(&self, p: Vec3) -> Option<[usize; 3]> {
        let c = self.coord(p);
        let mut o = [0usize; 3];
        for k in 0..3 {
            let v = c[k].round();
            if v < 0.0 || v as usize >= self.n[k] {
                return None;
            }
            o[k] = v as usize;
        }
        Some(o)
    }
    pub fn center(&self, x: usize, y: usize, z: usize) -> Vec3 {
        self.origin + v3(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5) * self.h
    }
    pub fn solid(&self, i: usize) -> bool {
        self.cell[i] != 0
    }
}

/// How sealed the body is, from the voxel fill (`voxelize`).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Solid {
    /// Holes in the body's surface (loops of edges with one face).
    pub openings: usize,
    /// The largest opening's extent, as a fraction of the body's size.
    pub largest_opening: f64,
    /// Share of the inside that the outside flood reached through the
    /// openings and only the ray vote put back.
    pub leaked: f64,
    /// Share of the body's inside (or near it) where the rays split three
    /// to three: the fill cannot tell inside from outside there.
    pub contested: f64,
    /// False when too much is contested (`MAX_CONTESTED`) for distances
    /// through the body to be trusted.
    pub trusted: bool,
}

/// More of the inside than this contested and the fill is not trusted.
pub const MAX_CONTESTED: f64 = 0.05;

/// Six-ray inside votes per voxel (see the module docs). `use_tri` picks
/// the triangles that count.
fn ray_votes(g: &Grid, pos: &[Vec3], tris: &[[u32; 3]], use_tri: &dyn Fn(usize) -> bool) -> Vec<u8> {
    // Shells: connected components over shared vertices.
    let mut parent: Vec<u32> = (0..pos.len() as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for t in tris {
        let a = find(&mut parent, t[0]);
        for &v in &t[1..] {
            let b = find(&mut parent, v);
            if a != b {
                parent[b as usize] = a;
            }
        }
    }
    let shell: Vec<u32> = tris.iter().map(|t| find(&mut parent, t[0])).collect();
    let mut votes = vec![0u8; g.cell.len()];
    let n = g.n;
    // Rays sit a hair off the voxel centres so they never run exactly
    // through a vertex or an edge of a grid-aligned mesh.
    let (eu, ev) = (0.013_7, 0.029_1);
    let mut odd = vec![false; pos.len()];
    for axis in 0..3 {
        let (ua, va) = ((axis + 1) % 3, (axis + 2) % 3);
        let (nu, nv) = (n[ua], n[va]);
        let mut hits: Vec<Vec<(f64, u32)>> = vec![Vec::new(); nu * nv];
        for (ti, t) in tris.iter().enumerate() {
            if !use_tri(ti) {
                continue;
            }
            let c: Vec<[f64; 3]> = t.iter().map(|&i| g.coord(pos[i as usize])).collect();
            let (a, b, cc) = ((c[0][ua], c[0][va]), (c[1][ua], c[1][va]), (c[2][ua], c[2][va]));
            let area = (b.0 - a.0) * (cc.1 - a.1) - (b.1 - a.1) * (cc.0 - a.0);
            if area.abs() < 1e-14 {
                continue;
            }
            let (lo_u, hi_u) = (a.0.min(b.0).min(cc.0), a.0.max(b.0).max(cc.0));
            let (lo_v, hi_v) = (a.1.min(b.1).min(cc.1), a.1.max(b.1).max(cc.1));
            let u0 = (lo_u - eu).ceil().max(0.0) as usize;
            let v0 = (lo_v - ev).ceil().max(0.0) as usize;
            let u1 = (hi_u - eu).floor();
            let v1 = (hi_v - ev).floor();
            if u1 < 0.0 || v1 < 0.0 {
                continue;
            }
            for iv in v0..=(v1 as usize).min(nv - 1) {
                for iu in u0..=(u1 as usize).min(nu - 1) {
                    let p = (iu as f64 + eu, iv as f64 + ev);
                    let e = |s: (f64, f64), t: (f64, f64)| (t.0 - s.0) * (p.1 - s.1) - (t.1 - s.1) * (p.0 - s.0);
                    let (e0, e1, e2) = (e(a, b), e(b, cc), e(cc, a));
                    if !((e0 > 0.0 && e1 > 0.0 && e2 > 0.0) || (e0 < 0.0 && e1 < 0.0 && e2 < 0.0)) {
                        continue;
                    }
                    let depth = (e1 * c[0][axis] + e2 * c[1][axis] + e0 * c[2][axis]) / area;
                    hits[iv * nu + iu].push((depth, shell[ti]));
                }
            }
        }
        for iv in 0..nv {
            for iu in 0..nu {
                let list = &mut hits[iv * nu + iu];
                if list.is_empty() {
                    continue;
                }
                list.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
                let cell = |k: usize| {
                    let mut q = [0usize; 3];
                    q[axis] = k;
                    q[ua] = iu;
                    q[va] = iv;
                    g.idx(q[0], q[1], q[2])
                };
                // From the low end, then from the high end.
                for from_low in [true, false] {
                    let mut count = 0usize;
                    let mut h = 0usize;
                    let order: Vec<usize> = if from_low { (0..n[axis]).collect() } else { (0..n[axis]).rev().collect() };
                    for k in order {
                        let kf = k as f64;
                        while h < list.len() {
                            let (d, sh) = if from_low { list[h] } else { list[list.len() - 1 - h] };
                            if (from_low && d >= kf) || (!from_low && d <= kf) {
                                break;
                            }
                            let o = &mut odd[sh as usize];
                            *o = !*o;
                            if *o {
                                count += 1;
                            } else {
                                count -= 1;
                            }
                            h += 1;
                        }
                        if count > 0 {
                            votes[cell(k)] += 1;
                        }
                    }
                    for &(_, sh) in list.iter() {
                        odd[sh as usize] = false;
                    }
                }
            }
        }
    }
    votes
}

/// Holes in a surface: loops of edges that have one face, as (count,
/// largest loop's bounding-box extent).
fn openings(pos: &[Vec3], tris: &[[u32; 3]], use_tri: &dyn Fn(usize) -> bool) -> (usize, f64) {
    let mut edges: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        if !use_tri(ti) {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut parent: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    fn find(p: &mut std::collections::HashMap<u32, u32>, x: u32) -> u32 {
        let mut r = x;
        while let Some(&q) = p.get(&r) {
            if q == r {
                break;
            }
            r = q;
        }
        p.insert(x, r);
        r
    }
    let mut open: Vec<(u32, u32)> = edges.iter().filter(|e| *e.1 == 1).map(|e| *e.0).collect();
    open.sort_unstable();
    for &(a, b) in &open {
        parent.entry(a).or_insert(a);
        parent.entry(b).or_insert(b);
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent.insert(rb, ra);
        }
    }
    let verts: Vec<u32> = parent.keys().copied().collect();
    let mut boxes: std::collections::BTreeMap<u32, (Vec3, Vec3)> = std::collections::BTreeMap::new();
    for v in verts {
        let r = find(&mut parent, v);
        let p = pos[v as usize];
        let e = boxes.entry(r).or_insert((p, p));
        e.0 = e.0.min(p);
        e.1 = e.1.max(p);
    }
    let largest = boxes.values().map(|(lo, hi)| (*hi - *lo).max_elem()).fold(0.0, f64::max);
    (boxes.len(), largest)
}

pub fn voxelize(pos: &[Vec3], tris: &[[u32; 3]], res: usize) -> Grid {
    voxelize_body(pos, tris, res, None).0
}

/// `voxelize`, and how sealed the body is (`body[t]` marks the triangles
/// of the body, as against open pieces such as a cape; all when `None`).
pub fn voxelize_body(pos: &[Vec3], tris: &[[u32; 3]], res: usize, body: Option<&[bool]>) -> (Grid, Solid) {
    let mut lo = v3(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = -lo;
    for p in pos {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let ext = hi - lo;
    let h = (ext.max_elem() / res as f64).max(1e-9);
    let pad = 2.0;
    let origin = lo - v3(pad, pad, pad) * h;
    let n = [
        ((ext.x / h).ceil() as usize + 2 * pad as usize + 1).max(1),
        ((ext.y / h).ceil() as usize + 2 * pad as usize + 1).max(1),
        ((ext.z / h).ceil() as usize + 2 * pad as usize + 1).max(1),
    ];
    let mut g = Grid { n, origin, h, cell: vec![0u8; n[0] * n[1] * n[2]] };
    // Surface: sample each triangle densely enough to hit every voxel it crosses.
    for t in tris {
        let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        let lmax = (b - a).len().max((c - a).len()).max((c - b).len());
        let k = ((lmax / (h * 0.5)).ceil() as usize).max(1);
        for i in 0..=k {
            for j in 0..=(k - i) {
                let (u, v) = (i as f64 / k as f64, j as f64 / k as f64);
                let p = a + (b - a) * u + (c - a) * v;
                if let Some([x, y, z]) = g.cell_of(p) {
                    let id = g.idx(x, y, z);
                    g.cell[id] = 1;
                }
            }
        }
    }
    for p in pos {
        if let Some([x, y, z]) = g.cell_of(*p) {
            let id = g.idx(x, y, z);
            g.cell[id] = 1;
        }
    }
    // Interior: everything the outside flood (6-connected) cannot reach.
    let total = g.cell.len();
    let mut outside = vec![false; total];
    let mut stack = vec![0usize];
    outside[0] = true;
    while let Some(i) = stack.pop() {
        let x = i % n[0];
        let y = (i / n[0]) % n[1];
        let z = i / (n[0] * n[1]);
        let mut push = |xx: isize, yy: isize, zz: isize| {
            if xx < 0 || yy < 0 || zz < 0 || xx as usize >= n[0] || yy as usize >= n[1] || zz as usize >= n[2] {
                return;
            }
            let j = (zz as usize * n[1] + yy as usize) * n[0] + xx as usize;
            if !outside[j] && g.cell[j] == 0 {
                outside[j] = true;
                stack.push(j);
            }
        };
        let (x, y, z) = (x as isize, y as isize, z as isize);
        push(x - 1, y, z);
        push(x + 1, y, z);
        push(x, y - 1, z);
        push(x, y + 1, z);
        push(x, y, z - 1);
        push(x, y, z + 1);
    }
    let all = |_: usize| true;
    let votes = ray_votes(&g, pos, tris, &all);
    let mut flooded = 0usize;
    for i in 0..total {
        if g.cell[i] == 0 && (!outside[i] || votes[i] >= 4) {
            if outside[i] {
                flooded += 1;
            }
            g.cell[i] = 2;
        }
    }
    // How sealed the body is: its own triangles only, so an open piece
    // (a cape is a sheet) is not counted as an opening.
    let is_body = |t: usize| body.is_none_or(|b| b[t]);
    let body_votes = if body.is_some_and(|b| b.iter().any(|x| !x)) { ray_votes(&g, pos, tris, &is_body) } else { votes };
    let (mut inside, mut tied) = (0usize, 0usize);
    for i in 0..total {
        if g.cell[i] == 1 {
            continue;
        }
        if body_votes[i] >= 4 || (g.cell[i] == 2 && !outside[i]) {
            inside += 1;
        } else if body_votes[i] == 3 {
            tied += 1;
        }
    }
    let (n_open, largest) = openings(pos, tris, &is_body);
    let interior = (0..total).filter(|&i| g.cell[i] == 2).count();
    let contested = tied as f64 / (inside + tied).max(1) as f64;
    let scale = {
        let mut lo = v3(f64::MAX, f64::MAX, f64::MAX);
        let mut hi = -lo;
        for p in pos {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        (hi - lo).max_elem().max(1e-9)
    };
    let solid = Solid {
        openings: n_open,
        largest_opening: (largest / scale * 1e4).round() / 1e4,
        leaked: (flooded as f64 / interior.max(1) as f64 * 1e4).round() / 1e4,
        contested: (contested * 1e4).round() / 1e4,
        trusted: contested <= MAX_CONTESTED,
    };
    (g, solid)
}

/// Per-vertex nearest bones by voxel geodesic distance.
pub struct Geo {
    /// Up to K (joint, distance) per vertex, ascending distance.
    pub near: Vec<Vec<(u32, f32)>>,
    /// Distances at or beyond `cap` mean "unreachable / far".
    pub cap: f32,
    pub res: usize,
    /// How sealed the body is (its openings, and whether the fill can be
    /// trusted).
    pub solid: Solid,
}

pub const K: usize = 8;

impl Geo {
    /// Distance from vertex v to bone j; bones outside the top-K report the
    /// K-th distance (a lower bound) or the cap.
    pub fn dist(&self, v: usize, j: u32) -> f32 {
        let near = &self.near[v];
        if let Some(e) = near.iter().find(|e| e.0 == j) {
            return e.1;
        }
        if near.len() < K { self.cap } else { near.last().map(|e| e.1).unwrap_or(self.cap) }
    }
    pub fn nearest(&self, v: usize) -> Option<(u32, f32)> {
        self.near[v].first().copied()
    }
}

fn bone_seeds(g: &Grid, a: Vec3, b: Vec3) -> Vec<(usize, f32)> {
    let len = (b - a).len();
    let steps = ((len / (g.h * 0.5)).ceil() as usize).max(1);
    let mut seeds = Vec::new();
    for s in 0..=steps {
        let p = a.lerp(b, s as f64 / steps as f64);
        if let Some([x, y, z]) = g.cell_of(p) {
            let i = g.idx(x, y, z);
            if g.solid(i) {
                seeds.push((i, 0.0));
            }
        }
    }
    if seeds.is_empty() {
        // Bone outside the mesh: seed the nearest solid voxel(s).
        let mut best = (f64::MAX, 0usize);
        for z in 0..g.n[2] {
            for y in 0..g.n[1] {
                for x in 0..g.n[0] {
                    let i = g.idx(x, y, z);
                    if g.solid(i) {
                        let d = segment_distance(g.center(x, y, z), a, b).1;
                        if d < best.0 {
                            best = (d, i);
                        }
                    }
                }
            }
        }
        if best.0 < f64::MAX {
            seeds.push((best.1, best.0 as f32));
        }
    }
    seeds.sort_unstable_by_key(|s| s.0);
    seeds.dedup_by_key(|s| s.0);
    seeds
}

fn dijkstra(g: &Grid, seeds: &[(usize, f32)], cap: f32) -> Vec<f32> {
    let mut dist = vec![f32::INFINITY; g.cell.len()];
    let mut heap = BinaryHeap::new();
    for &(i, d) in seeds {
        dist[i] = d;
        heap.push(Reverse((d.to_bits(), i as u32)));
    }
    let h = g.h as f32;
    let (nx, ny, nz) = (g.n[0] as isize, g.n[1] as isize, g.n[2] as isize);
    let mut offs = Vec::new();
    for dz in -1isize..=1 {
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let l = ((dx * dx + dy * dy + dz * dz) as f32).sqrt() * h;
                offs.push((dx, dy, dz, l));
            }
        }
    }
    while let Some(Reverse((db, i))) = heap.pop() {
        let d = f32::from_bits(db);
        let i = i as usize;
        if d > dist[i] || d > cap {
            continue;
        }
        let x = (i as isize) % nx;
        let y = ((i as isize) / nx) % ny;
        let z = (i as isize) / (nx * ny);
        for &(dx, dy, dz, l) in &offs {
            let (xx, yy, zz) = (x + dx, y + dy, z + dz);
            if xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz {
                continue;
            }
            let j = ((zz * ny + yy) * nx + xx) as usize;
            if g.cell[j] == 0 {
                continue;
            }
            let nd = d + l;
            if nd < dist[j] {
                dist[j] = nd;
                heap.push(Reverse((nd.to_bits(), j as u32)));
            }
        }
    }
    dist
}

/// Distance at p: trilinear over the solid voxel centers around it.
fn sample(g: &Grid, dist: &[f32], p: Vec3, cap: f32) -> f32 {
    let c = g.coord(p);
    let base = [c[0].floor(), c[1].floor(), c[2].floor()];
    let f = [c[0] - base[0], c[1] - base[1], c[2] - base[2]];
    let (mut acc, mut wsum) = (0.0f64, 0.0f64);
    let mut best = f32::INFINITY;
    for k in 0..8 {
        let o = [k & 1, (k >> 1) & 1, (k >> 2) & 1];
        let mut idx = [0usize; 3];
        let mut ok = true;
        let mut w = 1.0;
        for a in 0..3 {
            let v = base[a] as isize + o[a] as isize;
            if v < 0 || v as usize >= g.n[a] {
                ok = false;
                break;
            }
            idx[a] = v as usize;
            w *= if o[a] == 1 { f[a] } else { 1.0 - f[a] };
        }
        if !ok {
            continue;
        }
        let i = g.idx(idx[0], idx[1], idx[2]);
        if g.cell[i] == 0 || !dist[i].is_finite() {
            continue;
        }
        best = best.min(dist[i]);
        acc += dist[i] as f64 * w.max(1e-6);
        wsum += w.max(1e-6);
    }
    if wsum > 0.0 {
        (acc / wsum) as f32
    } else if best.is_finite() {
        best
    } else {
        cap
    }
}

/// Geodesic distances from every joint in `joints` to every vertex.
pub fn geodesics(model: &Model, res: usize, joints: &[usize]) -> Geo {
    let body: Vec<bool> = model.tris.iter().map(|t| !model.parts[model.vpart[t[0] as usize] as usize].piece).collect();
    let (g, solid) = voxelize_body(&model.rest, &model.tris, res, Some(&body));
    let cap = (model.scale * 0.75) as f32;
    let nv = model.nverts();
    let mut near: Vec<Vec<(u32, f32)>> = vec![Vec::new(); nv];
    for chunk in joints.chunks(16) {
        let cols: Vec<(u32, Vec<f32>)> = chunk
            .par_iter()
            .map(|&j| {
                let seeds = bone_seeds(&g, model.skel.head[j], model.skel.tail[j]);
                let dist = dijkstra(&g, &seeds, cap);
                let col: Vec<f32> = model.rest.iter().map(|p| sample(&g, &dist, *p, cap).min(cap)).collect();
                (j as u32, col)
            })
            .collect();
        near.par_iter_mut().enumerate().for_each(|(v, list)| {
            for (j, col) in &cols {
                list.push((*j, col[v]));
            }
            list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
            list.truncate(K);
        });
    }
    Geo { near, cap, res, solid }
}
