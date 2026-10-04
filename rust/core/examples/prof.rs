use std::time::Instant;
use weightforge_core::{fix::*, metrics::*, scene::Model};
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let t = Instant::now();
    let m = Model::load(std::path::Path::new(&p)).unwrap();
    println!("load {:.2}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let ctx = Ctx::new(&m, &CtxOpts::default());
    println!("ctx {:.2}s ({} poses)", t.elapsed().as_secs_f64(), ctx.poses.len());
    let t = Instant::now();
    let _ = evaluate(&m, &ctx, &m.weights);
    println!("evaluate full {:.2}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let _ = evaluate_with(&m, &ctx, &m.weights, false);
    println!("evaluate fast {:.2}s", t.elapsed().as_secs_f64());
    let ev = evaluate_with(&m, &ctx, &m.weights, false);
    for (n, f) in [("despeckle", 0), ("smooth", 1), ("geodesic", 2), ("bands", 3)] {
        let t = Instant::now();
        let _ = match f { 0 => cleanup(&m, &ctx, &ev, false), 1 => cleanup(&m, &ctx, &ev, true), 2 => geodesic(&m, &ctx), _ => joint_bands(&m, &ctx) };
        println!("{n} {:.2}s", t.elapsed().as_secs_f64());
    }
    let t = Instant::now();
    let _ = optimize(&m, &ctx, &m.weights, &ev);
    println!("optimize {:.2}s", t.elapsed().as_secs_f64());
}
