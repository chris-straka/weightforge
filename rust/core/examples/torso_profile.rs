//! Torso radius by height (median distance of torso vertices from the
//! vertical line through the spine), 1% bins, and the paint's mean
//! brightness there: torso_profile a.glb
use std::path::Path;
use weightforge_core as wf;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let m = wf::scene::Model::load(Path::new(&a[0])).unwrap();
    let ctx = wf::metrics::Ctx::new(&m, &wf::metrics::CtxOpts { clips: false, ..Default::default() });
    let paint = wf::paint::read(&m);
    let (mut y0, mut y1) = (f64::MAX, f64::MIN);
    for p in &m.rest {
        y0 = y0.min(p.y);
        y1 = y1.max(p.y);
    }
    let h = y1 - y0;
    let hips = m.skel.anatomy.roles["hips"][0];
    let c = m.skel.head[hips];
    let mut bins: Vec<Vec<(f64, f64)>> = vec![Vec::new(); 100];
    for v in 0..m.nverts() {
        let j = ctx.region_bone[ctx.vregion[v] as usize].unwrap_or(0);
        if m.skel.anatomy.group[j] != wf::anatomy::Group::Torso {
            continue;
        }
        let p = m.rest[v];
        let b = (((p.y - y0) / h) * 100.0) as usize;
        let r = ((p.x - c.x).powi(2) + (p.z - c.z).powi(2)).sqrt() / h;
        let l = (paint.rgb[v][0] + paint.rgb[v][1] + paint.rgb[v][2]) as f64 / 3.0;
        if b < 100 {
            bins[b].push((r, l));
        }
    }
    for (b, xs) in bins.iter_mut().enumerate() {
        if xs.len() < 5 {
            continue;
        }
        xs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let r = xs[xs.len() / 2].0;
        let p90 = xs[xs.len() * 9 / 10].0;
        let l = xs.iter().map(|x| x.1).sum::<f64>() / xs.len() as f64;
        println!("{b:>3}% n={:>4} r_med {:.4} r_p90 {:.4} light {:.2}", xs.len(), r, p90, l);
    }
}
