//! The stiff parts the checker finds on a rig (named pieces, small loose
//! shells, painted bands and patches), with their size and vertex count.
//!     cargo run --release --example stiff_probe -- a.glb
use std::path::Path;
use weightforge_core as wf;
fn main() {
    for p in std::env::args().skip(1) {
        let m = wf::scene::Model::load(Path::new(&p)).unwrap();
        let ctx = wf::metrics::Ctx::new(&m, &wf::metrics::CtxOpts { clips: false, ..Default::default() });
        let g: Vec<wf::anatomy::Group> = (0..m.nverts())
            .map(|v| {
                let j = ctx.region_bone[ctx.vregion[v] as usize].unwrap_or(0);
                m.skel.anatomy.group.get(j).cloned().unwrap_or_default()
            })
            .collect();
        let paint = wf::paint::read(&m);
        let h = wf::owner::height(&m);
        println!("== {} (textured {})", p, paint.textured);
        for s in wf::owner::stiff_parts(&m, &g, h) {
            println!("  {:<40} {:>5} verts, size {:.1}% of height", s.name, s.verts.len(), 100.0 * s.size / h);
        }
    }
}
