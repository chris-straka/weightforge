//! The edges that stretch most in one pose: where they are at rest, how
//! long, and what their ends are weighted to. For reading a tear.
//!
//!     cargo run --release --example worst_edges -- rig.glb hip_forward.L [count]

use std::path::Path;
use weightforge_core as wf;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        eprintln!("usage: worst_edges <rig.glb> <pose> [count]");
        std::process::exit(2);
    }
    let count: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);
    let opts = wf::metrics::CtxOpts { intersect: false, ..Default::default() };
    let (model, ctx, _, _) = wf::check(Path::new(&a[0]), &opts).expect("rig");
    let Some(p) = ctx.poses.iter().position(|p| p.name == a[1]) else {
        eprintln!("no pose {}", a[1]);
        std::process::exit(2);
    };
    let pos = wf::skin::lbs(&model, &ctx.mats[p], &model.weights);
    let mut es: Vec<(f64, usize)> = model
        .edges
        .iter()
        .enumerate()
        .map(|(i, e)| ((pos[e[0] as usize] - pos[e[1] as usize]).len() / ctx.rest_edge[i].max(1e-12), i))
        .collect();
    es.sort_by(|x, y| y.0.total_cmp(&x.0));
    let sk = &model.skel;
    let nm = |v: usize| {
        model.weights[v]
            .iter()
            .map(|&(j, x)| format!("{}:{:.2}", sk.names[j as usize].trim_start_matches("DEF-"), x))
            .collect::<Vec<_>>()
            .join(" ")
    };
    for &(s, i) in es.iter().take(count) {
        let e = model.edges[i];
        let (u, v) = (e[0] as usize, e[1] as usize);
        let (pu, pv) = (model.rest[u], model.rest[v]);
        println!(
            "x{:.1} len {:.1} mm | ({:+.3},{:.3},{:+.3}) [{}] -- ({:+.3},{:.3},{:+.3}) [{}]",
            s,
            ctx.rest_edge[i] * 1000.0,
            pu.x,
            pu.y,
            pu.z,
            nm(u),
            pv.x,
            pv.y,
            pv.z,
            nm(v)
        );
    }
}
