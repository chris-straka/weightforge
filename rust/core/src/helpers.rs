//! Twist/helper bones (the HLL humanoid's `DEF-upper_arm_twist.L`,
//! `DEF-thigh_twist.R`, ...).
//!
//! A helper sits on its driver's joint, under the driver's parent, and
//! turns by a share of the driver's rotation (motionforge
//! `core/src/helpers.rs` inserts them and bakes that rotation into every
//! clip). weightforge has to pose them the same way, or its ROM would
//! leave the helpers at rest and measure a rig the game never shows.
//!
//! Detection: node `extras.hll_helper = {"driver", "share"}`, else the
//! name convention `<driver base>_twist.<side>` with share 0.5. A helper
//! must hang from its driver's parent; anything else is a plain bone.
//!
//! Weights: helpers are not a region of their own (their vertices belong
//! to the driver's region), not a pose role, and every generated fix
//! candidate is re-blended onto them with the same rule motionforge uses
//! (`reblend`), so a candidate never drops the helpers.

use crate::math::{Mat4, Quat};
use crate::scene::{Model, Node, Skeleton, VW, Weights, normalize_vw, prune_vw};
use serde_json::Value;

pub const DEFAULT_SHARE: f64 = 0.5;

#[derive(Clone, Debug, PartialEq)]
pub struct Helper {
    /// Joint indices (skin order).
    pub joint: usize,
    pub driver: usize,
    pub share: f64,
}

/// Helpers among the skin's joints.
pub fn detect(raw_nodes: &[Value], nodes: &[Node], joints: &[usize], names: &[String]) -> Vec<Helper> {
    let mut out = Vec::new();
    for (j, &n) in joints.iter().enumerate() {
        let extras = raw_nodes.get(n).and_then(|x| x.get("extras")).and_then(|e| e.get("hll_helper"));
        let (driver_name, share) = match extras {
            Some(h) => (
                h.get("driver").and_then(Value::as_str).map(str::to_string),
                h.get("share").and_then(Value::as_f64).unwrap_or(DEFAULT_SHARE),
            ),
            None => (by_name(&names[j]), DEFAULT_SHARE),
        };
        let Some(driver_name) = driver_name else { continue };
        let Some(d) = names.iter().position(|x| *x == driver_name) else { continue };
        if d == j || nodes[joints[d]].parent != nodes[n].parent || !(0.0..=1.0).contains(&share) {
            continue;
        }
        out.push(Helper { joint: j, driver: d, share });
    }
    out
}

/// `DEF-upper_arm_twist.L` -> `DEF-upper_arm.L`.
fn by_name(name: &str) -> Option<String> {
    let (base, side) = name.rsplit_once('.')?;
    if side != "L" && side != "R" {
        return None;
    }
    let driver = base.strip_suffix("_twist")?;
    Some(format!("{driver}.{side}"))
}

pub fn helper_of(sk: &Skeleton, j: usize) -> Option<&Helper> {
    sk.helpers.iter().find(|h| h.joint == j)
}

/// The joint a vertex region or pose role should use for `j`.
pub fn owner(sk: &Skeleton, j: usize) -> usize {
    helper_of(sk, j).map(|h| h.driver).unwrap_or(j)
}

/// Pose the helpers in node globals `g` from their drivers. `keyed[node]`
/// marks nodes the pose sets explicitly (a baked clip), which win.
pub fn drive(sk: &Skeleton, g: &mut [Mat4], keyed: &[bool]) {
    for h in &sk.helpers {
        let (hn, dn) = (sk.joints[h.joint], sk.joints[h.driver]);
        if keyed.get(hn).copied().unwrap_or(false) {
            continue;
        }
        let parent = match sk.nodes[dn].parent {
            Some(p) => g[p],
            None => Mat4::IDENTITY,
        };
        let posed = parent.inverse() * g[dn];
        let rest = sk.nodes[dn].local();
        let delta = posed.rotation() * rest.rotation().conj();
        let part = Quat::IDENTITY.slerp(delta.normalized(), h.share).normalized();
        let hr = &sk.nodes[hn];
        let t = hr.t + (posed.translation() - rest.translation());
        g[hn] = parent * Mat4::from_trs(t, (part * hr.r).normalized(), hr.s);
    }
}

/// Re-blend one vertex over a helper (motionforge `helpers::reblend`).
/// Returns (ancestor scale, helper weight, driver weight).
pub fn reblend(limb: f64, prox: f64, share: f64) -> (f64, f64, f64) {
    let total = limb + prox;
    if limb <= 0.0 || prox <= 0.0 || total <= 0.0 {
        return (1.0, 0.0, limb);
    }
    let a = limb / total;
    if a <= share {
        let h = total * a / share;
        ((total - h) / prox, h, 0.0)
    } else {
        let h = total * (1.0 - a) / (1.0 - share);
        (0.0, h, total - h)
    }
}

fn is_ancestor(sk: &Skeleton, a: usize, mut j: usize) -> bool {
    let mut guard = 0;
    while let Some(p) = sk.jparent[j] {
        if p == a {
            return true;
        }
        j = p;
        guard += 1;
        if guard > sk.jparent.len() {
            break;
        }
    }
    false
}

/// Per helper: joints outside its driver's subtree (and not a helper).
/// In a pose that turns only the driver these all stay put, so their
/// weight is the "body" side of the joint (chest, hips, the other thigh).
fn outside(sk: &Skeleton) -> Vec<Vec<bool>> {
    sk.helpers
        .iter()
        .map(|h| (0..sk.joints.len()).map(|j| j != h.driver && !is_ancestor(sk, h.driver, j) && helper_of(sk, j).is_none()).collect())
        .collect()
}

/// Band widths `(rings, smoothing passes)`, measured on the SkinTokens
/// Andras game mesh (5.8k verts, README "Twist/helper bones"): wider arm bands
/// trade stretch for collapse, wider leg bands drag the shirt hem.
pub const ARM_BAND: (usize, usize) = (6, 20);
pub const LEG_BAND: (usize, usize) = (4, 30);

/// Fix candidate: widen each driver's joint band, then re-blend it onto
/// the helper. Around the joint, the driver's share `a = limb / (limb +
/// body)` is smoothed over the mesh (`rings` rings around the vertices
/// that mix the two or sit on a hard edge between them, then Laplacian
/// passes), so the armpit and groin turn gradually instead of tearing;
/// the helper then keeps that wider blend from collapsing (`reassign`).
/// Identity without helpers.
pub fn band(model: &Model, w: &Weights) -> Weights {
    let sk = &model.skel;
    if sk.helpers.is_empty() {
        return w.clone();
    }
    let out = outside(sk);
    // Fold helpers into their drivers first.
    let mut cur: Weights = w
        .iter()
        .map(|vw| {
            let mut acc: VW = Vec::new();
            for &(j, x) in vw {
                let j = owner(sk, j as usize) as u32;
                match acc.iter_mut().find(|e| e.0 == j) {
                    Some(e) => e.1 += x,
                    None => acc.push((j, x)),
                }
            }
            acc
        })
        .collect();
    let nv = model.nverts();
    for (k, h) in sk.helpers.iter().enumerate() {
        let d = h.driver as u32;
        // Legs: the body side is the hips chain only, so the band never
        // spreads one thigh into the other across the crotch.
        let is_leg = sk.names[h.driver].contains("thigh");
        let (rings, iters) = if is_leg { LEG_BAND } else { ARM_BAND };
        let body_set: Vec<bool> =
            if is_leg { (0..sk.joints.len()).map(|j| j != h.driver && is_ancestor(sk, j, h.driver)).collect() } else { out[k].clone() };
        let parent = sk.jparent[h.driver].unwrap_or(h.driver) as u32;
        let limb: Vec<f64> = cur.iter().map(|vw| vw.iter().filter(|e| e.0 == d).map(|e| e.1).sum()).collect();
        let body: Vec<f64> = cur.iter().map(|vw| vw.iter().filter(|e| body_set[e.0 as usize]).map(|e| e.1).sum()).collect();
        let joint: Vec<bool> = (0..nv).map(|v| limb[v] + body[v] >= 0.5).collect();
        let share_of = |v: usize| limb[v] / (limb[v] + body[v]).max(1e-12);
        // Seeds: vertices that mix the two, and both ends of a hard edge
        // (all-limb next to all-body).
        let mixed: Vec<bool> = (0..nv)
            .map(|v| {
                joint[v]
                    && ((limb[v] > 0.02 && body[v] > 0.02)
                        || model.adj[v].iter().any(|&u| joint[u as usize] && (share_of(v) - share_of(u as usize)).abs() > 0.5))
            })
            .collect();
        let mut zone = mixed.clone();
        for _ in 0..rings {
            let prev = zone.clone();
            for v in 0..nv {
                if !prev[v] && joint[v] && model.adj[v].iter().any(|&u| prev[u as usize]) {
                    zone[v] = true;
                }
            }
        }
        let mut a: Vec<f64> = (0..nv).map(|v| if joint[v] { limb[v] / (limb[v] + body[v]) } else { 0.0 }).collect();
        for _ in 0..iters {
            let prev = a.clone();
            for v in 0..nv {
                if !zone[v] {
                    continue;
                }
                let ns: Vec<f64> = model.adj[v].iter().filter(|&&u| joint[u as usize]).map(|&u| prev[u as usize]).collect();
                if ns.is_empty() {
                    continue;
                }
                let mean = ns.iter().sum::<f64>() / ns.len() as f64;
                a[v] = 0.5 * prev[v] + 0.5 * mean;
            }
        }
        for v in 0..nv {
            if !zone[v] {
                continue;
            }
            let total = limb[v] + body[v];
            let (nl, nb) = (a[v] * total, (1.0 - a[v]) * total);
            let vw = &mut cur[v];
            if body[v] > 1e-9 {
                let f = nb / body[v];
                for e in vw.iter_mut() {
                    if body_set[e.0 as usize] {
                        e.1 *= f;
                    }
                }
            } else if nb > 0.0 {
                vw.push((parent, nb));
            }
            match vw.iter_mut().find(|e| e.0 == d) {
                Some(e) => e.1 = nl,
                None => vw.push((d, nl)),
            }
            // Merge duplicates (the parent may already be listed).
            let mut m: VW = Vec::new();
            for &(j, x) in vw.iter() {
                match m.iter_mut().find(|e| e.0 == j) {
                    Some(e) => e.1 += x,
                    None => m.push((j, x)),
                }
            }
            *vw = m;
        }
    }
    reassign(sk, &cur)
}

/// Fold helper weight back into the drivers, then re-blend every vertex
/// that mixes a driver with its ancestors. Identity without helpers.
pub fn reassign(sk: &Skeleton, w: &Weights) -> Weights {
    if sk.helpers.is_empty() {
        return w.clone();
    }
    let anc = outside(sk);
    w.iter()
        .map(|vw| {
            let mut acc: VW = Vec::new();
            for &(j, x) in vw {
                let j = owner(sk, j as usize) as u32;
                match acc.iter_mut().find(|e| e.0 == j) {
                    Some(e) => e.1 += x,
                    None => acc.push((j, x)),
                }
            }
            for (k, h) in sk.helpers.iter().enumerate() {
                let limb: f64 = acc.iter().filter(|e| e.0 as usize == h.driver).map(|e| e.1).sum();
                let prox: f64 = acc.iter().filter(|e| anc[k][e.0 as usize]).map(|e| e.1).sum();
                let (scale, hw, dw) = reblend(limb, prox, h.share);
                if hw <= 0.0 {
                    continue;
                }
                for e in acc.iter_mut() {
                    if anc[k][e.0 as usize] {
                        e.1 *= scale;
                    } else if e.0 as usize == h.driver {
                        e.1 = dw;
                    }
                }
                acc.push((h.joint as u32, hw));
            }
            acc.retain(|e| e.1 > 0.0);
            normalize_vw(prune_vw(&acc, 4, 0.0))
        })
        .collect()
}

/// Weights as motion: each helper weight `x` becomes `share * x` on the
/// driver and the rest on the helper's parent joint (they turn the vertex
/// the same way). Identity without helpers.
pub fn equivalent<'a>(sk: &Skeleton, w: &'a Weights) -> std::borrow::Cow<'a, Weights> {
    if sk.helpers.is_empty() {
        return std::borrow::Cow::Borrowed(w);
    }
    let mut split: Vec<Option<(u32, u32, f64)>> = vec![None; sk.joints.len()];
    for h in &sk.helpers {
        let parent = sk.jparent[h.joint].unwrap_or(h.driver);
        split[h.joint] = Some((h.driver as u32, parent as u32, h.share));
    }
    std::borrow::Cow::Owned(
        w.iter()
            .map(|vw| {
                let mut acc: VW = Vec::with_capacity(vw.len() + 1);
                let mut add = |j: u32, x: f64| match acc.iter_mut().find(|e| e.0 == j) {
                    Some(e) => e.1 += x,
                    None => acc.push((j, x)),
                };
                for &(j, x) in vw {
                    match split[j as usize] {
                        Some((d, p, s)) => {
                            add(d, x * s);
                            add(p, x * (1.0 - s));
                        }
                        None => add(j, x),
                    }
                }
                acc
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{Bone, build, to_glb};
    use crate::metrics::{Ctx, CtxOpts, evaluate};

    /// The clean mannequin plus the HLL arm and thigh helpers.
    fn with_helpers() -> Model {
        let mut f = build("clean");
        for side in ["L", "R"] {
            for (helper, driver) in [("DEF-upper_arm_twist", "DEF-upper_arm"), ("DEF-thigh_twist", "DEF-thigh")] {
                let name: &'static str = Box::leak(format!("{helper}.{side}").into_boxed_str());
                let d = f.bones.iter().position(|b| b.name == format!("{driver}.{side}")).unwrap();
                let b = &f.bones[d];
                f.bones.push(Bone { name, parent: b.parent, head: b.head, tail: b.tail });
            }
        }
        Model::from_glb(to_glb(&f)).unwrap()
    }

    #[test]
    fn helpers_are_found_and_turn_half_their_driver() {
        let m = with_helpers();
        assert_eq!(m.skel.helpers.len(), 4);
        let set = crate::poses::parse_pose_set(
            "PoseSet(name: \"t\", poses: [PoseSpec(name: \"up\", moves: [Move(bone: \"upper_arm.L\", motion: Bend(Up, 80))])])",
        )
        .unwrap();
        let poses = crate::poses::resolve(&set, &m.skel);
        // The helper is not a pose role: the move lands on the real upper arm.
        assert_eq!(m.skel.names[poses[0].moves[0].0], "DEF-upper_arm.L");
        let mats = crate::skin::joint_matrices(&m, &poses[0]);
        let h = m.skel.helpers.iter().find(|h| m.skel.names[h.joint] == "DEF-upper_arm_twist.L").unwrap();
        let angle = |q: Quat| 2.0 * q.w.abs().min(1.0).acos().to_degrees();
        let (ad, ah) = (angle(mats[h.driver].rotation()), angle(mats[h.joint].rotation()));
        assert!((ad - 80.0).abs() < 1e-6, "driver {ad}");
        assert!((ah - 40.0).abs() < 1e-6, "helper {ah}");
        // Same pivot: the joint head does not move.
        let head = m.skel.head[h.joint];
        assert!((mats[h.joint].transform_point(head) - head).len() < 1e-9);
    }

    #[test]
    fn band_weights_helpers_and_regions_ignore_them() {
        let m = with_helpers();
        let ctx = Ctx::new(&m, &CtxOpts::default());
        // No region is named after a helper; it belongs to its driver.
        assert!(!ctx.regions.iter().any(|r| r.name.contains("_twist")));
        let w = band(&m, &m.weights);
        for h in &m.skel.helpers {
            let total: f64 = w.iter().flat_map(|vw| vw.iter()).filter(|e| e.0 as usize == h.joint).map(|e| e.1).sum();
            assert!(total > 1.0, "{} got no weight", m.skel.names[h.joint]);
        }
        for vw in &w {
            assert!(vw.len() <= 4);
            assert!((vw.iter().map(|e| e.1).sum::<f64>() - 1.0).abs() < 1e-9);
        }
        // Unweighted helpers do not change the check (noise is judged on motion).
        let plain = Model::from_glb(to_glb(&build("clean"))).unwrap();
        let a = evaluate(&m, &ctx, &m.weights);
        let b = evaluate(&plain, &Ctx::new(&plain, &CtxOpts::default()), &plain.weights);
        assert_eq!(a.flags, b.flags);
    }

    #[test]
    fn names() {
        assert_eq!(by_name("DEF-upper_arm_twist.L").as_deref(), Some("DEF-upper_arm.L"));
        assert_eq!(by_name("DEF-thigh_twist.R").as_deref(), Some("DEF-thigh.R"));
        assert_eq!(by_name("DEF-upper_arm.L"), None);
        assert_eq!(by_name("upperarm_twist_01_l"), None);
    }

    #[test]
    fn reblend_keeps_mean_rotation() {
        for &(l, p) in &[(0.5, 0.5), (0.1, 0.9), (0.8, 0.2)] {
            let (s, h, d) = reblend(l, p, 0.5);
            let t = p * s + h + d;
            assert!((t - (l + p)).abs() < 1e-12);
            assert!(((h * 0.5 + d) / t - l / (l + p)).abs() < 1e-12);
        }
    }
}
