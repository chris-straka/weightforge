//! What is fused on a generated body: the triangles that join two limbs
//! (a corner nearest a left-leg bone and a corner nearest a right-leg
//! bone; an arm corner and a trunk corner well down the arm; an arm corner
//! and a leg corner), how far the join runs, and how many of the vertices
//! that fail in the hip and arm poses sit on it. One JSON line, and with
//! `--out` a picture: limbs in colour, joins magenta, failing skin red.
//!
//!     cargo run --release --example fused_probe -- rig.glb [--out fused.png] [--slices slices.png]

use std::path::Path;
use weightforge_core as wf;
use wf::math::{Vec3, v3};
use wf::render::{Fit, Image, View, draw_mesh};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Limb {
    Trunk,
    LegL,
    LegR,
    ArmL,
    ArmR,
    Head,
}

fn limb_color(l: Limb) -> [f64; 3] {
    match l {
        Limb::Trunk => [0.62, 0.62, 0.62],
        Limb::LegL => [0.25, 0.45, 0.85],
        Limb::LegR => [0.90, 0.60, 0.20],
        Limb::ArmL => [0.30, 0.70, 0.40],
        Limb::ArmR => [0.60, 0.40, 0.75],
        Limb::Head => [0.75, 0.75, 0.55],
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: fused_probe <rig.glb> [--out fused.png] [--slices slices.png]");
        std::process::exit(2);
    };
    let (mut out, mut slices) = (None, None);
    while let Some(a) = args.next() {
        if a == "--out" {
            out = args.next();
        } else if a == "--slices" {
            slices = args.next();
        }
    }
    let opts = wf::metrics::CtxOpts::default();
    let (model, ctx, ev, rep) = wf::check(Path::new(&path), &opts).expect("rig");
    let sk = &model.skel;
    let roles = wf::poses::resolve_roles(sk);
    let role = |name: &str| -> Option<usize> { roles.get(name).and_then(|v| v.first().copied()) };
    let nj = model.njoints();
    let limb_of: Vec<Limb> = (0..nj)
        .map(|j| {
            let mut k = Some(wf::helpers::owner(sk, j));
            while let Some(b) = k {
                for (r, l) in [
                    ("thigh.L", Limb::LegL),
                    ("thigh.R", Limb::LegR),
                    ("upper_arm.L", Limb::ArmL),
                    ("upper_arm.R", Limb::ArmR),
                    ("neck", Limb::Head),
                    ("head", Limb::Head),
                ] {
                    if role(r) == Some(b) {
                        return l;
                    }
                }
                k = sk.jparent[b];
            }
            Limb::Trunk
        })
        .collect();
    // Deform bones: those that carry weight, and helpers' drivers.
    let mut deform = vec![false; nj];
    for vw in &model.weights {
        for &(j, x) in vw {
            if x > 0.01 {
                deform[wf::helpers::owner(sk, j as usize)] = true;
            }
        }
    }
    let nv = model.nverts();
    let nearest: Vec<usize> = (0..nv)
        .map(|v| {
            (0..nj)
                .filter(|&j| deform[j])
                .min_by(|&a, &b| sk.seg_dist(a, model.rest[v]).total_cmp(&sk.seg_dist(b, model.rest[v])))
                .unwrap_or(0)
        })
        .collect();
    let limb: Vec<Limb> = nearest.iter().map(|&j| limb_of[j]).collect();

    let median = |mut d: Vec<f64>| -> f64 {
        if d.is_empty() {
            return 0.0;
        }
        d.sort_by(|a, b| a.total_cmp(b));
        d[d.len() / 2]
    };
    let radius_of = |j: Option<usize>| -> f64 {
        let Some(j) = j else { return 0.0 };
        median((0..nv).filter(|&v| nearest[v] == j).map(|v| sk.seg_dist(j, model.rest[v])).collect())
    };
    let (thl, thr) = (role("thigh.L"), role("thigh.R"));
    let thigh_r = 0.5 * (radius_of(thl) + radius_of(thr));
    let hip_y = match (thl, thr) {
        (Some(a), Some(b)) => 0.5 * (sk.head[a].y + sk.head[b].y),
        _ => 0.0,
    };
    let thigh_len = thl.map(|j| sk.bone_len(j)).unwrap_or(0.0);

    // Joins.
    let mut tri_kind = vec![0u8; model.tris.len()]; // 1 leg-leg, 2 arm-trunk (web), 3 arm-leg
    let mut join_v = [vec![false; nv], vec![false; nv], vec![false; nv]];
    let arm_t = |v: usize| -> f64 {
        // Distance along the arm from the shoulder joint, in arm radii.
        let side = if limb[v] == Limb::ArmL { "upper_arm.L" } else { "upper_arm.R" };
        let Some(j) = role(side) else { return 0.0 };
        let dir = (sk.tail[j] - sk.head[j]).normalized();
        let r = radius_of(Some(j)).max(1e-6);
        (model.rest[v] - sk.head[j]).dot(dir) / r
    };
    let (mut ll_low, mut ll_high) = (f64::MAX, f64::MIN);
    let mut at_max = [0.0f64; 2];
    let mut counts = [0usize; 3];
    for (ti, t) in model.tris.iter().enumerate() {
        let ls: Vec<Limb> = t.iter().map(|&i| limb[i as usize]).collect();
        let has = |l: Limb| ls.contains(&l);
        let arm = has(Limb::ArmL) || has(Limb::ArmR);
        let leg = has(Limb::LegL) || has(Limb::LegR);
        let kind = if has(Limb::LegL) && has(Limb::LegR) {
            for &i in t {
                ll_low = ll_low.min(model.rest[i as usize].y);
                ll_high = ll_high.max(model.rest[i as usize].y);
            }
            1
        } else if arm && leg {
            3
        } else if arm && has(Limb::Trunk) {
            let far = t
                .iter()
                .filter(|&&i| matches!(limb[i as usize], Limb::ArmL | Limb::ArmR))
                .map(|&i| (arm_t(i as usize), limb[i as usize]))
                .fold((0.0f64, Limb::ArmL), |a, b| if b.0 > a.0 { b } else { a });
            if far.0 > 1.5 {
                let s = (far.1 == Limb::ArmR) as usize;
                at_max[s] = at_max[s].max(far.0);
                2
            } else {
                0
            }
        } else {
            0
        };
        tri_kind[ti] = kind;
        if kind > 0 {
            counts[kind as usize - 1] += 1;
            for &i in t {
                join_v[kind as usize - 1][i as usize] = true;
            }
        }
    }
    // Width of the leg-leg join: mean length of its edges that cross it.
    let cross: Vec<f64> = model
        .edges
        .iter()
        .filter(|e| {
            let (a, b) = (limb[e[0] as usize], limb[e[1] as usize]);
            (a == Limb::LegL && b == Limb::LegR) || (a == Limb::LegR && b == Limb::LegL)
        })
        .map(|e| (model.rest[e[0] as usize] - model.rest[e[1] as usize]).len())
        .collect();
    let cross_mean = if cross.is_empty() { 0.0 } else { cross.iter().sum::<f64>() / cross.len() as f64 };

    // Failing vertices per pose (as `weights probe` counts them).
    let mask = wf::report::finding_mask(&ctx, &ev, &rep);
    let near = |set: &[bool], v: usize| set[v] || model.adj[v].iter().any(|&u| set[u as usize]);
    let mut poses = serde_json::Map::new();
    let mut bad_leg = vec![false; nv];
    let mut bad_arm = vec![false; nv];
    let (mut crotch, mut crotch_on, mut armpit, mut armpit_on) = (0usize, 0usize, 0usize, 0usize);
    for (p, pose) in ctx.poses.iter().enumerate() {
        let bad: Vec<usize> = ev.bad_in_pose[p]
            .iter()
            .filter(|(v, f)| f & mask[*v as usize] & !wf::metrics::F_INTERSECT != 0)
            .map(|x| x.0 as usize)
            .collect();
        if !bad.is_empty() {
            poses.insert(pose.name.clone(), serde_json::json!(bad.len()));
        }
        if pose.name.starts_with("hip_forward") {
            crotch += bad.len();
            crotch_on += bad.iter().filter(|&&v| near(&join_v[0], v)).count();
            bad.iter().for_each(|&v| bad_leg[v] = true);
        }
        if pose.name.starts_with("arm_") {
            armpit += bad.len();
            armpit_on += bad.iter().filter(|&&v| near(&join_v[1], v) || near(&join_v[2], v)).count();
            bad.iter().for_each(|&v| bad_arm[v] = true);
        }
    }
    let cm = |x: f64| (x * 1000.0).round() / 10.0;
    println!(
        "{}",
        serde_json::json!({
            "rig": rep.file, "score": rep.score, "fails": rep.findings.iter().filter(|f| f.severity == "fail").count(),
            "verts": nv, "tris": model.tris.len(), "height_cm": cm(model.rest.iter().map(|p| p.y).fold(f64::MIN, f64::max) - model.rest.iter().map(|p| p.y).fold(f64::MAX, f64::min)),
            "thigh_radius_cm": cm(thigh_r), "thigh_len_cm": cm(thigh_len),
            "leg_leg_tris": counts[0],
            "leg_join_below_hip_cm": if counts[0] > 0 { cm(hip_y - ll_low) } else { 0.0 },
            "leg_join_top_below_hip_cm": if counts[0] > 0 { cm(hip_y - ll_high) } else { 0.0 },
            "leg_join_edge_cm": cm(cross_mean),
            "arm_trunk_web_tris": counts[1], "arm_web_reach_radii": [(at_max[0] * 10.0).round() / 10.0, (at_max[1] * 10.0).round() / 10.0],
            "arm_leg_tris": counts[2],
            "crotch_bad": crotch, "crotch_bad_on_join": crotch_on,
            "armpit_bad": armpit, "armpit_bad_on_join": armpit_on,
            "poses": poses,
        })
    );

    if let (Some(path), Some(legs)) = (slices, wf::unfuse::Legs::of(&model)) {
        // Cross-sections from the hip joints down to below the knees, seen
        // from above: outlines round both legs magenta, round one blue or
        // orange, round neither grey; a white dot where each leg's axis is.
        let n = 8usize;
        let cell = 300usize;
        let mut img = Image::new(cell * n, cell + 24, wf::render::BG);
        let mut sl = wf::unfuse::Slicer::new(&model);
        let half =
            0.26 * (model.rest.iter().map(|p| p.y).fold(f64::MIN, f64::max) - model.rest.iter().map(|p| p.y).fold(f64::MAX, f64::min));
        let ppu = cell as f64 * 0.5 / half;
        let c = legs.center;
        for k in 0..n {
            let below = thigh_len * (-0.1 + 1.6 * k as f64 / (n - 1) as f64);
            let y = hip_y - below;
            let put = |img: &mut Image, p: (f64, f64), col: [u8; 3], s: usize| {
                let px = (k * cell) as f64 + cell as f64 * 0.5 + (p.0 - c.x) * ppu;
                let py = 24.0 + cell as f64 * 0.5 - (p.1 - c.z) * ppu;
                if px >= (k * cell) as f64 && px < ((k + 1) * cell - s) as f64 && py >= 24.0 && py < (24 + cell - s) as f64 {
                    img.rect(px as usize, py as usize, s, s, col);
                }
            };
            for (pts, closed) in sl.outlines(y) {
                let (wl, wr) = (
                    wf::unfuse::winding(&pts, closed, legs.axis_at(0, y)) > wf::unfuse::ROUND,
                    wf::unfuse::winding(&pts, closed, legs.axis_at(1, y)) > wf::unfuse::ROUND,
                );
                let col = match (wl, wr) {
                    (true, true) => [240, 40, 215],
                    (true, false) => [70, 120, 220],
                    (false, true) => [230, 150, 50],
                    _ => [120, 120, 120],
                };
                let m = pts.len();
                for i in 0..(if closed { m } else { m - 1 }) {
                    let (a, b) = (pts[i], pts[(i + 1) % m]);
                    let steps = (((a.0 - b.0).hypot(a.1 - b.1) * ppu).ceil() as usize).max(1);
                    for s in 0..=steps {
                        let t = s as f64 / steps as f64;
                        put(&mut img, (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t), col, 2);
                    }
                }
            }
            for leg in 0..2 {
                put(&mut img, legs.axis_at(leg, y), [255, 255, 255], 5);
            }
            img.text(k * cell + 6, 4, &format!("{:.0} cm below hips", below * 100.0), 2, [220, 220, 220]);
        }
        std::fs::write(&path, img.png_bytes()).expect("write png");
    }
    let Some(out) = out else { return };
    // Flat-shaded copy: a vertex per corner, so a join is a solid colour.
    let draw = |img: &mut Image, x0: usize, y0: usize, w: usize, h: usize, view: View, keep: &dyn Fn(Vec3) -> bool, bad: &[bool]| {
        let mut pos = Vec::new();
        let mut tris = Vec::new();
        let mut col = Vec::new();
        for (ti, t) in model.tris.iter().enumerate() {
            if !t.iter().all(|&i| keep(model.rest[i as usize])) {
                continue;
            }
            let base = pos.len() as u32;
            for &i in t {
                let i = i as usize;
                pos.push(model.rest[i]);
                let c = match tri_kind[ti] {
                    1 | 2 | 3 => [0.95, 0.10, 0.85],
                    _ => limb_color(limb[i]),
                };
                col.push(if bad[i] { [1.0, 0.1, 0.1] } else { c });
            }
            tris.push([base, base + 1, base + 2]);
        }
        if pos.is_empty() {
            return;
        }
        let f: Fit = wf::render::fit(&view, &[&pos], w, h);
        draw_mesh(img, x0, y0, w, h, &pos, &tris, &col, &view, f);
    };
    let (cw, ch) = (420usize, 420usize);
    let mut img = Image::new(cw * 4, ch * 2 + 24, wf::render::BG);
    let none = vec![false; nv];
    let top = model.rest.iter().map(|p| p.y).fold(f64::MIN, f64::max);
    let leg_band = |p: Vec3| p.y < hip_y + 0.25 * thigh_len && p.y > hip_y - 0.75 * thigh_len;
    let sh_y = role("upper_arm.L").map(|j| sk.head[j].y).unwrap_or(top * 0.8);
    let arm_band = |p: Vec3| p.y < sh_y + 0.06 * top && p.y > sh_y - 0.28 * top;
    let views = [
        (View { yaw_deg: 0.0, pitch_deg: -35.0 }, "front, from below"),
        (View { yaw_deg: 180.0, pitch_deg: -35.0 }, "back, from below"),
        (View { yaw_deg: 0.0, pitch_deg: 0.0 }, "front"),
        (View { yaw_deg: 180.0, pitch_deg: 0.0 }, "back"),
    ];
    for (k, (view, label)) in views.iter().enumerate() {
        let legs = k < 2;
        let keep: &dyn Fn(Vec3) -> bool = if legs { &leg_band } else { &arm_band };
        draw(&mut img, k * cw, 24, cw, ch, *view, keep, &none);
        draw(&mut img, k * cw, 24 + ch, cw, ch, *view, keep, if legs { &bad_leg } else { &bad_arm });
        img.text(k * cw + 6, 4, label, 2, [220, 220, 220]);
    }
    let _ = v3(0.0, 0.0, 0.0);
    std::fs::write(&out, img.png_bytes()).expect("write png");
}
