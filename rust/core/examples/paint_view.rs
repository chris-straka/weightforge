//! A rig at rest in its paint, front and side (to see what the paint
//! pass reads): paint_view a.glb out.png [patch_tol]
use std::path::Path;
use weightforge_core as wf;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let m = wf::scene::Model::load(Path::new(&a[0])).unwrap();
    let paint = wf::paint::read(&m);
    let cols: Vec<[f64; 3]> = if let Some(t) = a.get(2).and_then(|x| x.parse::<f32>().ok()) {
        let p = wf::paint::patches(&m, &paint, t);
        p.iter()
            .map(|&i| {
                let h = (i as u64).wrapping_mul(2654435761) % 1000;
                [(h % 10) as f64 / 10.0, ((h / 10) % 10) as f64 / 10.0, ((h / 100) % 10) as f64 / 10.0]
            })
            .collect()
    } else {
        paint.rgb.iter().map(|c| [c[0] as f64, c[1] as f64, c[2] as f64]).collect()
    };
    let mut img = wf::render::Image::new(1000, 900, wf::render::BG);
    for (i, yaw) in [0.0, 90.0].iter().enumerate() {
        let view = wf::render::View { yaw_deg: *yaw, pitch_deg: 5.0 };
        let f = wf::render::fit(&view, &[&m.rest], 500, 900);
        wf::render::draw_mesh(&mut img, i * 500, 0, 500, 900, &m.rest, &m.tris, &cols, &view, f);
    }
    std::fs::write(&a[1], img.png_bytes()).unwrap();
}
