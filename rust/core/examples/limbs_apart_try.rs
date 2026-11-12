//! Applies `unfuse::limbs_apart` (the weights for limbs that meshkit's
//! `unfuse.py` cut apart) to a rig and writes it, for measuring that step
//! alone. The two band widths can be given, to tune them.
//!
//!     cargo run --release --example limbs_apart_try -- cut.glb out.glb [band_down band_out]

use std::path::Path;
use weightforge_core as wf;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        eprintln!("usage: limbs_apart_try <cut.glb> <out.glb> [band_down band_out]");
        std::process::exit(2);
    }
    let num = |i: usize, d: f64| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let model = wf::scene::Model::load(Path::new(&a[0])).expect("rig");
    let w = wf::unfuse::limbs_apart_with(&model, &model.weights, num(2, wf::unfuse::BAND_DOWN), num(3, wf::unfuse::BAND_OUT));
    let changed = (0..model.nverts()).filter(|&v| w[v] != model.weights[v]).count();
    eprintln!("{}: {} verts, {} changed", a[0], model.nverts(), changed);
    let glb = model.write_weights(&w).expect("write");
    std::fs::write(&a[1], glb.to_bytes()).expect("write file");
}
