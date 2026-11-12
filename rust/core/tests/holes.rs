//! Distances "through the body" must not depend on whether the mesh is
//! sealed. Generated characters keep a handful of small openings; when the
//! voxel fill floods in through one, only the shell is left solid and the
//! distance from a bone becomes a walk along the skin from one seed.
//!
//! Known answer: inside a straight capped tube (convex) the shortest path
//! through the body from a bone to a vertex is the straight line, so the
//! geodesic distance is the Euclidean distance to the bone segment (up to
//! the voxel metric's error). The same tube with a small hole in its wall
//! must give the same distances and the same nearest bone per vertex.

use weightforge_core::fixtures::{Bone, Fixture, to_glb};
use weightforge_core::math::{Vec3, segment_distance, v3};
use weightforge_core::scene::Model;
use weightforge_core::voxel::geodesics;

const R: f64 = 0.1;
const RINGS: usize = 50; // y = 0 .. 1 in 0.02 steps
const SEG: usize = 24;

/// A capped tube along +Y, outward-facing triangles; `hole` removes the
/// wall quads (ring, segment) listed.
fn tube(hole: &[(usize, usize)]) -> Fixture {
    let mut pos = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for i in 0..=RINGS {
        let y = i as f64 / RINGS as f64;
        for k in 0..SEG {
            let a = std::f64::consts::TAU * k as f64 / SEG as f64;
            pos.push(v3(R * a.cos(), y, R * a.sin()));
        }
    }
    let at = |i: usize, k: usize| (i * SEG + k % SEG) as u32;
    for i in 0..RINGS {
        for k in 0..SEG {
            if hole.contains(&(i, k)) {
                continue;
            }
            let (a, b, c, d) = (at(i, k), at(i, k + 1), at(i + 1, k), at(i + 1, k + 1));
            tris.push([a, c, b]);
            tris.push([b, c, d]);
        }
    }
    let bottom = pos.len() as u32;
    pos.push(v3(0.0, 0.0, 0.0));
    let top = pos.len() as u32;
    pos.push(v3(0.0, 1.0, 0.0));
    for k in 0..SEG {
        tris.push([bottom, at(0, k), at(0, k + 1)]);
        tris.push([top, at(RINGS, k + 1), at(RINGS, k)]);
    }
    let bones = vec![
        Bone { name: "A", parent: None, head: v3(0.0, 0.15, 0.0), tail: v3(0.0, 0.5, 0.0) },
        Bone { name: "B", parent: Some(0), head: v3(0.0, 0.5, 0.0), tail: v3(0.0, 0.85, 0.0) },
    ];
    let w = pos.iter().map(|p: &Vec3| vec![(if p.y < 0.5 { 0 } else { 1 }, 1.0)]).collect();
    Fixture { bones, body_pos: pos, body_tris: tris, cape_pos: Vec::new(), cape_tris: Vec::new(), body_w: w, cape_w: Vec::new() }
}

fn distances(f: &Fixture) -> (Model, Vec<[f64; 2]>) {
    let m = Model::from_glb(to_glb(f)).unwrap();
    let geo = geodesics(&m, 128, &[0, 1]);
    let d = (0..m.nverts()).map(|v| [geo.dist(v, 0) as f64, geo.dist(v, 1) as f64]).collect();
    (m, d)
}

#[test]
fn a_small_hole_does_not_change_distances_or_regions() {
    let (m, closed) = distances(&tube(&[]));
    // A 2 x 2 quad hole (4 cm x 5 cm) in the side wall, high on bone B.
    let (mh, holed) = distances(&tube(&[(34, 5), (34, 6), (35, 5), (35, 6)]));
    assert_eq!(m.nverts(), mh.nverts());
    let h = 1.0 / 128.0;
    // The segments the distances are measured from (a leaf bone's tail is
    // estimated, so take them from the skeleton).
    let sk = &m.skel;
    let bone = [(sk.head[0], sk.tail[0]), (sk.head[1], sk.tail[1])];
    let (mut worst_c, mut worst_h, mut moved) = (f64::MIN, f64::MIN, 0);
    for v in 0..m.nverts() {
        let p = m.rest[v];
        let truth = [segment_distance(p, bone[0].0, bone[0].1).1, segment_distance(p, bone[1].0, bone[1].1).1];
        for b in 0..2 {
            // Voxel paths are at most ~8% longer than straight, plus a
            // couple of voxels at each end.
            let tol = 0.1 * truth[b] + 3.0 * h;
            worst_c = worst_c.max((closed[v][b] - truth[b]).abs() - tol);
            worst_h = worst_h.max((holed[v][b] - truth[b]).abs() - tol);
        }
        // Same nearest bone, except where the two are within a voxel.
        if (truth[0] - truth[1]).abs() > 2.0 * h {
            let want = if truth[0] < truth[1] { 0 } else { 1 };
            let got = if holed[v][0] < holed[v][1] { 0 } else { 1 };
            if got != want {
                moved += 1;
            }
        }
    }
    assert!(worst_c <= 0.0, "closed tube off the straight-line distance by {worst_c:.4} beyond tolerance");
    assert!(worst_h <= 0.0, "holed tube off the straight-line distance by {worst_h:.4} beyond tolerance");
    assert_eq!(moved, 0, "{moved} vertices changed region through the hole");
}
