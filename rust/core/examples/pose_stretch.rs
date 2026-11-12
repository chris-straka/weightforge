//! Every pose's stretch and failing-vertex count, as `weights probe`
//! measures them (the third-largest vertex stretch in the pose; vertices
//! of a reported finding that fail in it), one JSON line a pose. For laying
//! a set of judged cases against a new build of the checker.
//!
//!     cargo run --release --example pose_stretch -- rig.glb [--poses rom.ron] [pose ...]

use std::path::Path;
use weightforge_core as wf;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: pose_stretch <rig.glb> [--poses rom.ron] [pose ...]");
        std::process::exit(2);
    };
    let (mut ron, mut only) = (None, Vec::new());
    while let Some(a) = args.next() {
        if a == "--poses" {
            ron = args.next();
        } else {
            only.push(a);
        }
    }
    let set = ron.map(|p| wf::poses::parse_pose_set(&std::fs::read_to_string(&p).expect("pose file")).expect("pose set"));
    let opts = wf::metrics::CtxOpts { poses: set.as_ref(), ..Default::default() };
    let (model, ctx, ev, rep) = wf::check(Path::new(&path), &opts).expect("rig");
    let mask = wf::report::finding_mask(&ctx, &ev, &rep);
    for (p, pose) in ctx.poses.iter().enumerate() {
        if !only.is_empty() && !only.contains(&pose.name) {
            continue;
        }
        let pos = wf::skin::lbs(&model, &ctx.mats[p], &model.weights);
        let mut st = vec![1.0f64; model.nverts()];
        for (e, r0) in model.edges.iter().zip(&ctx.rest_edge) {
            let (a, b) = (e[0] as usize, e[1] as usize);
            let s = (pos[a] - pos[b]).len() / r0.max(1e-12);
            st[a] = st[a].max(s);
            st[b] = st[b].max(s);
        }
        st.sort_by(|a, b| b.total_cmp(a));
        let stretch = st.get(2).copied().unwrap_or(1.0);
        let bad = ev.bad_in_pose[p].iter().filter(|(v, f)| f & mask[*v as usize] & !wf::metrics::F_INTERSECT != 0).count();
        println!(
            "{}",
            serde_json::json!({"rig": rep.file, "pose": pose.name, "stretch": (stretch * 100.0).round() / 100.0, "bad": bad, "score": rep.score})
        );
    }
}
