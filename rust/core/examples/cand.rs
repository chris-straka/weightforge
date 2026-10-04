use weightforge_core::{fix::*, fixtures, glb::Glb, metrics::*, scene::*};
fn main() {
    let fault = std::env::args().nth(1).unwrap();
    let bytes = fixtures::to_glb(&fixtures::build(&fault)).to_bytes();
    let m = Model::from_glb(Glb::parse(&bytes).unwrap()).unwrap();
    let (b, c) = fixtures::ground_truth();
    let gt: Weights = b.into_iter().chain(c).collect();
    let ctx = Ctx::new(&m, &CtxOpts::default());
    let ev = evaluate(&m, &ctx, &m.weights);
    let noisy: Vec<usize> = (0..m.nverts()).filter(|&v| ev.flags[v] & F_NOISE != 0).collect();
    println!("orig {:.5}  noise-flagged verts {}", weight_error(&m.weights, &gt), noisy.len());
    let bad: Vec<usize> = (0..m.nverts())
        .filter(|&v| {
            let a = &m.weights[v];
            let b = &gt[v];
            a.iter().map(|&(j, x)| (x - weight_of(b, j)).abs()).sum::<f64>()
                + b.iter().filter(|e| weight_of(a, e.0) == 0.0).map(|e| e.1).sum::<f64>()
                > 0.1
        })
        .collect();
    let unk = bad.iter().filter(|&&v| noise_of(&m, &m.weights, v) > 0.25).count();
    let nz: Vec<f64> = bad.iter().map(|&v| noise_of(&m, &m.weights, v)).collect();
    let mut nzs = nz.clone();
    nzs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "verts off GT >0.1: {}  of which noise>0.25: {}  noise quartiles {:.2} {:.2} {:.2}",
        bad.len(),
        unk,
        nzs[nzs.len() / 4],
        nzs[nzs.len() / 2],
        nzs[3 * nzs.len() / 4]
    );
    let cw = cleanup(&m, &ctx, &ev, true);
    let l1 = |a: &VW, b: &VW| {
        a.iter().map(|&(j, x)| (x - weight_of(b, j)).abs()).sum::<f64>()
            + b.iter().filter(|e| weight_of(a, e.0) == 0.0).map(|e| e.1).sum::<f64>()
    };
    let e_in: f64 = bad.iter().map(|&v| l1(&m.weights[v], &gt[v])).sum();
    let e_out: f64 = bad.iter().map(|&v| l1(&cw[v], &gt[v])).sum();
    println!("on bad verts: {e_in:.2} -> {e_out:.2}");
    let dw = cleanup(&m, &ctx, &ev, false);
    let mut res: Vec<(f64, usize)> = (0..m.nverts()).map(|v| (l1(&dw[v], &gt[v]), v)).filter(|x| x.0 > 0.02).collect();
    res.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!(
        "residual verts >0.02: {} sum {:.2} (of which originally bad: {})",
        res.len(),
        res.iter().map(|x| x.0).sum::<f64>(),
        res.iter().filter(|x| bad.contains(&x.1)).count()
    );
    for &(e, v) in res.iter().take(4) {
        println!("  v{v} err {e:.2} noise_flag {} out {:?} gt {:?} orig {:?}", ev.flags[v] & F_NOISE != 0, dw[v], gt[v], m.weights[v]);
    }
    for &v in bad.iter().take(0) {
        println!("  v{v} in {:?}\n     out {:?}\n     gt {:?}", m.weights[v], cw[v], gt[v]);
    }
    for (n, w) in [
        ("despeckle", cleanup(&m, &ctx, &ev, false)),
        ("cleanup", cleanup(&m, &ctx, &ev, true)),
        ("geodesic", geodesic(&m, &ctx)),
        ("bands", joint_bands(&m, &ctx)),
    ] {
        println!("{n:<10} {:.5}", weight_error(&w, &gt));
    }
    let res = fix(&fault, &m, &ctx, &FixOpts::default());
    println!("fix {:.5}", weight_error(&res.weights, &gt));
    let mut rr: Vec<(f64, usize)> = (0..m.nverts()).map(|v| (l1(&res.weights[v], &gt[v]), v)).filter(|x| x.0 > 0.02).collect();
    rr.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("fix residual verts >0.02: {} sum {:.2}", rr.len(), rr.iter().map(|x| x.0).sum::<f64>());
    for &(e, v) in rr.iter().take(6) {
        println!("  v{v} y {:.3} err {e:.2} out {:?} gt {:?}", m.rest[v].y, res.weights[v], gt[v]);
    }
    for r in &res.fix.regions {
        println!("  {} {} -> {} via {}", r.name, r.before, r.after, r.chosen);
    }
}
