//! Geodesic distance from each bone to each vertex, measured through a
//! voxelization of the mesh (Dionne & de Lasa 2013, "Geodesic Voxel
//! Binding for Production Character Meshes"). Paths stay inside the body,
//! so a hand hanging next to a thigh is far from the thigh bone even though
//! it is close in a straight line. Works on multi-shell, non-manifold,
//! self-intersecting meshes because it only rasterizes triangles.

use crate::math::{Vec3, segment_distance, v3};
use crate::scene::Model;
use rayon::prelude::*;
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

pub fn voxelize(pos: &[Vec3], tris: &[[u32; 3]], res: usize) -> Grid {
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
    for i in 0..total {
        if g.cell[i] == 0 && !outside[i] {
            g.cell[i] = 2;
        }
    }
    g
}

/// Per-vertex nearest bones by voxel geodesic distance.
pub struct Geo {
    /// Up to K (joint, distance) per vertex, ascending distance.
    pub near: Vec<Vec<(u32, f32)>>,
    /// Distances at or beyond `cap` mean "unreachable / far".
    pub cap: f32,
    pub res: usize,
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
    let g = voxelize(&model.rest, &model.tris, res);
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
    Geo { near, cap, res }
}
