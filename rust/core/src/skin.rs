//! Forward kinematics for a pose, then linear blend skinning (and dual
//! quaternion skinning, used as a volume-preserving target by `optimize`).
//! Every joint matrix is relative to rest: `M_j = G_j(pose) * G_j(rest)^-1`,
//! applied to the rest shape, so all candidates are measured on the same
//! reference geometry.

use crate::math::{Mat4, Quat, Vec3, v3};
use crate::poses::Pose;
use crate::scene::{Model, VW, Weights, globals};
use rayon::prelude::*;

pub fn joint_matrices(model: &Model, pose: &Pose) -> Vec<Mat4> {
    let sk = &model.skel;
    let mut nodes = sk.nodes.clone();
    for &(n, t, r, s) in &pose.locals {
        if let Some(nd) = nodes.get_mut(n) {
            if t.is_some() || r.is_some() || s.is_some() {
                if let Some(m) = nd.matrix.take() {
                    // Animated nodes must be TRS by spec; decompose defensively.
                    nd.t = m.translation();
                    nd.r = m.rotation();
                }
            }
            if let Some(t) = t {
                nd.t = t;
            }
            if let Some(r) = r {
                nd.r = r;
            }
            if let Some(s) = s {
                nd.s = s;
            }
        }
    }
    let g = if pose.moves.is_empty() {
        globals(&nodes, &sk.order)
    } else {
        let mut node_moves: Vec<Vec<(Vec3, f64)>> = vec![Vec::new(); nodes.len()];
        for &(j, axis, ang) in &pose.moves {
            node_moves[sk.joints[j]].push((axis, ang));
        }
        let mut g = vec![Mat4::IDENTITY; nodes.len()];
        let mut d = vec![Quat::IDENTITY; nodes.len()];
        for &i in &sk.order {
            let l = nodes[i].local();
            let (mut gi, mut di) = match nodes[i].parent {
                Some(p) => (g[p] * l, d[p]),
                None => (l, Quat::IDENTITY),
            };
            for &(axis, ang) in &node_moves[i] {
                let r = Quat::from_axis_angle(di.rotate(axis), ang);
                let h = gi.translation();
                let rm = Mat4::from_trs(Vec3::ZERO, r, v3(1.0, 1.0, 1.0));
                let about =
                    Mat4::from_trs(h, Quat::IDENTITY, v3(1.0, 1.0, 1.0)) * rm * Mat4::from_trs(-h, Quat::IDENTITY, v3(1.0, 1.0, 1.0));
                gi = about * gi;
                di = r * di;
            }
            g[i] = gi;
            d[i] = di;
        }
        g
    };
    sk.joints.iter().map(|&n| g[n] * sk.rest_global[n].inverse()).collect()
}

#[inline]
pub fn lbs_vertex(m: &[Mat4], vw: &VW, p: Vec3) -> Vec3 {
    if vw.is_empty() {
        return p;
    }
    let mut acc = Mat4::zero();
    for &(j, w) in vw {
        acc.scale_add(&m[j as usize], w);
    }
    acc.transform_point(p)
}

pub fn lbs(model: &Model, m: &[Mat4], w: &Weights) -> Vec<Vec3> {
    model.rest.par_iter().zip(w.par_iter()).map(|(p, vw)| lbs_vertex(m, vw, *p)).collect()
}

/// Dual quaternion (real, dual) for a rigid matrix (scale ignored).
#[derive(Clone, Copy)]
pub struct DualQuat {
    pub r: Quat,
    pub d: Quat,
}

pub fn to_dq(m: &Mat4) -> DualQuat {
    let r = m.rotation();
    let t = m.translation();
    let tq = Quat { x: t.x, y: t.y, z: t.z, w: 0.0 };
    let d = tq * r;
    DualQuat { r, d: Quat { x: d.x * 0.5, y: d.y * 0.5, z: d.z * 0.5, w: d.w * 0.5 } }
}

pub fn dqs_vertex(dq: &[DualQuat], vw: &VW, p: Vec3) -> Vec3 {
    if vw.is_empty() {
        return p;
    }
    let pivot = dq[vw[0].0 as usize].r;
    let (mut r, mut d) = (Quat { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }, Quat { x: 0.0, y: 0.0, z: 0.0, w: 0.0 });
    for &(j, w) in vw {
        let q = dq[j as usize];
        let s = if q.r.dot(pivot) < 0.0 { -w } else { w };
        r = Quat { x: r.x + q.r.x * s, y: r.y + q.r.y * s, z: r.z + q.r.z * s, w: r.w + q.r.w * s };
        d = Quat { x: d.x + q.d.x * s, y: d.y + q.d.y * s, z: d.z + q.d.z * s, w: d.w + q.d.w * s };
    }
    let n = (r.dot(r)).sqrt();
    if n < 1e-12 {
        return p;
    }
    let r = Quat { x: r.x / n, y: r.y / n, z: r.z / n, w: r.w / n };
    let d = Quat { x: d.x / n, y: d.y / n, z: d.z / n, w: d.w / n };
    let t = d * r.conj();
    r.rotate(p) + v3(t.x, t.y, t.z) * 2.0
}

pub fn dqs(model: &Model, m: &[Mat4], w: &Weights) -> Vec<Vec3> {
    let dq: Vec<DualQuat> = m.iter().map(to_dq).collect();
    model.rest.par_iter().zip(w.par_iter()).map(|(p, vw)| dqs_vertex(&dq, vw, *p)).collect()
}
