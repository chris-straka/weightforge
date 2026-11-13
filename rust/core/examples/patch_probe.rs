//! Paint patches of a rig in a height band (fractions of the height):
//! size, torso share, height span, angular cover, colour.
//!     cargo run --release --example patch_probe -- a.glb [lo hi tol]
use std::path::Path;
use weightforge_core as wf;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let m = wf::scene::Model::load(Path::new(&a[0])).unwrap();
    let lo: f64 = a.get(1).and_then(|x| x.parse().ok()).unwrap_or(0.45);
    let hi: f64 = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(0.65);
    let tol: f32 = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(0.09);
    let paint = wf::paint::read(&m);
    let patch = wf::paint::patches(&m, &paint, tol);
    let (mut y0, mut y1) = (f64::MAX, f64::MIN);
    for p in &m.rest {
        y0 = y0.min(p.y);
        y1 = y1.max(p.y);
    }
    let h = y1 - y0;
    let mut by: std::collections::BTreeMap<u32, Vec<usize>> = Default::default();
    for v in 0..m.nverts() {
        by.entry(patch[v]).or_default().push(v);
    }
    println!("{} patches", by.len());
    let mut rows: Vec<(usize, String)> = Vec::new();
    for (_, vs) in by {
        let c = vs.iter().fold(wf::math::Vec3::ZERO, |s, &v| s + m.rest[v]) / vs.len() as f64;
        let f = (c.y - y0) / h;
        if vs.len() < 12 || f < lo || f > hi {
            continue;
        }
        let (mut a0, mut a1) = (f64::MAX, f64::MIN);
        let mut bins = [false; 12];
        let mut col = [0f64; 3];
        for &v in &vs {
            a0 = a0.min(m.rest[v].y);
            a1 = a1.max(m.rest[v].y);
            let ang = (m.rest[v].z - c.z).atan2(m.rest[v].x - c.x);
            bins[(((ang + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * 12.0) as usize).min(11)] = true;
            for k in 0..3 {
                col[k] += paint.rgb[v][k] as f64 / vs.len() as f64;
            }
        }
        rows.push((
            vs.len(),
            format!(
                "{:>5} verts at {:.2} span {:.3} bins {:>2} rgb {:.2},{:.2},{:.2}",
                vs.len(),
                f,
                (a1 - a0) / h,
                bins.iter().filter(|&&b| b).count(),
                col[0],
                col[1],
                col[2]
            ),
        ));
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    for r in rows.iter().take(25) {
        println!("  {}", r.1);
    }
}
