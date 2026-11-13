//! Joint heads of a rig (rest, world), with each joint's parent:
//!     cargo run --release --example bone_heads -- a.glb [filter]
use std::path::Path;
use weightforge_core as wf;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let m = wf::scene::Model::load(Path::new(&a[0])).unwrap();
    let sk = &m.skel;
    for j in 0..sk.joints.len() {
        if a.len() > 1 && !sk.names[j].contains(&a[1]) {
            continue;
        }
        let h = sk.head[j];
        let p = sk.jparent[j].map(|p| sk.names[p].clone()).unwrap_or_default();
        println!("{:<28} {:>7.3} {:>7.3} {:>7.3}  <- {p}", sk.names[j], h.x, h.y, h.z);
    }
}
