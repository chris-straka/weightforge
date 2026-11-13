//! Bone roles of each rig from its structure (what the checker poses by)
//! beside the roles its bone names alone would give, and the body class.
//!
//!     cargo run --release --example roles_probe -- a.glb [b.glb ...]
use std::path::Path;
use weightforge_core as wf;

fn main() {
    for p in std::env::args().skip(1) {
        let m = match wf::scene::Model::load(Path::new(&p)) {
            Ok(m) => m,
            Err(e) => {
                println!("{p}: {e}");
                continue;
            }
        };
        let sk = &m.skel;
        let a = &sk.anatomy;
        let names = |js: &[usize]| js.iter().map(|&j| sk.names[j].clone()).collect::<Vec<_>>().join(" ");
        println!(
            "== {} : {} ({} joints, {} skins, {} rigid verts, facing {:?}) {:?}",
            Path::new(&p).file_name().unwrap().to_string_lossy(),
            a.body.as_str(),
            sk.joints.len(),
            m.skins.len(),
            m.rigid.iter().filter(|&&r| r).count(),
            (a.forward.x, a.forward.z),
            a.notes
        );
        let old = wf::poses::name_roles(sk);
        let mut keys: Vec<&String> = a.roles.keys().chain(old.keys()).collect();
        keys.sort();
        keys.dedup();
        for k in keys {
            let new = a.roles.get(k).map(|v| names(v)).unwrap_or_default();
            let was = old.get(k).map(|v| names(v)).unwrap_or_default();
            if k.starts_with("fingers") {
                println!(
                    "  {k:<14} {} joints (names: {})",
                    a.roles.get(k).map(|v| v.len()).unwrap_or(0),
                    old.get(k).map(|v| v.len()).unwrap_or(0)
                );
            } else if new == was {
                println!("  {k:<14} {new}");
            } else {
                println!("  {k:<14} {new:<40} | names: {was}");
            }
        }
    }
}
