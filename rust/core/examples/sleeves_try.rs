//! Applies `unfuse::sleeve_weights` (a sleeve joined to the body
//! re-weighted, with no cut) to a rig and writes it; with `--labels` also
//! writes what each vertex beside an arm was taken for, with its
//! position, for a cut along the sleeve's edge (meshkit `unfuse.py
//! --labels`). `--by-weights` also tells sleeve from coat where the arm is
//! sunk in the body with no waist, by the bones the skin mostly follows.
//!
//!     cargo run --release --example sleeves_try -- rig.glb out.glb [--labels labels.json] [--by-weights] [--coat | --sleeve]

use std::path::Path;
use weightforge_core as wf;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        eprintln!("usage: sleeves_try <rig.glb> <out.glb> [--labels labels.json] [--by-weights]");
        std::process::exit(2);
    }
    let by_weights = a.iter().any(|x| x == "--by-weights");
    let labels_path = a.iter().position(|x| x == "--labels").and_then(|i| a.get(i + 1));
    let mut model = wf::scene::Model::load(Path::new(&a[0])).expect("rig");
    if let Some(w) = wf::unfuse::cut_start(&model) {
        model.weights = w;
    }
    let file = wf::scene::Model::load(Path::new(&a[0])).expect("rig");
    // --coat: only the coat's half (skin of the body gives up the forearm's
    // bones); --sleeve: only the sleeve's; default both.
    let only = |x: &str| a.iter().any(|y| y == x);
    let mut w = model.weights.clone();
    if !only("--sleeve") {
        let ctx = wf::metrics::Ctx::new(&model, &wf::metrics::CtxOpts { intersect: false, ..Default::default() });
        w = wf::unfuse::coat_weights(&model, &ctx, &w);
    }
    if !only("--coat") {
        w = wf::unfuse::sleeve_weights(&model, &w, by_weights);
    }
    let changed = (0..model.nverts()).filter(|&v| w[v] != file.weights[v]).count();
    let mut doc = Vec::new();
    for (k, arm) in wf::unfuse::arms_of(&model).iter().enumerate() {
        let labels = wf::unfuse::sleeve_labels(&model, arm, &model.weights, by_weights);
        let count = |x: u8| labels.iter().filter(|&&l| l == x).count();
        eprintln!("arm {k}: sleeve {}, coat {}, body {}", count(1), count(3), count(2));
        for v in (0..model.nverts()).filter(|&v| labels[v] == 1 || labels[v] == 3) {
            let p = model.rest[v];
            doc.push(serde_json::json!([k, p.x, p.y, p.z, labels[v]]));
        }
    }
    eprintln!("{}: {} verts, {} changed", a[0], model.nverts(), changed);
    if let Some(p) = labels_path {
        std::fs::write(p, serde_json::to_vec(&doc).unwrap()).expect("write labels");
    }
    let glb = file.write_weights(&w).expect("write");
    std::fs::write(&a[1], glb.to_bytes()).expect("write file");
}
