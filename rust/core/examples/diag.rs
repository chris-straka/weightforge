//! Per-region metric distributions for threshold tuning.
use weightforge_core::{math::segment_distance, metrics::*, scene::Model};
fn main() {
    for p in std::env::args().skip(1) {
        let m = Model::load(std::path::Path::new(&p)).unwrap();
        let ctx = Ctx::new(&m, &CtxOpts { intersect: false, ..Default::default() });
        let ev = evaluate(&m, &ctx, &m.weights);
        println!("== {p}");
        for (ri, r) in ctx.regions.iter().enumerate() {
            let vs: Vec<usize> = (0..m.nverts()).filter(|&v| ctx.vregion[v] as usize == ri).collect();
            let mut st: Vec<f64> = vs.iter().map(|&v| ev.max_stretch[v]).collect();
            st.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let mut th: Vec<f64> = vs.iter().map(|&v| ev.min_thin[v]).collect();
            th.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // thin extent along region bone
            let j = m.skel.joint_by_name(&r.name);
            let ext = j
                .map(|j| {
                    let ts: Vec<f64> = vs
                        .iter()
                        .filter(|&&v| ev.min_thin[v] < 0.5)
                        .map(|&v| segment_distance(m.rest[v], m.skel.head[j], m.skel.tail[j]).0)
                        .collect();
                    if ts.is_empty() {
                        0.0
                    } else {
                        ts.iter().cloned().fold(f64::MIN, f64::max) - ts.iter().cloned().fold(f64::MAX, f64::min)
                    }
                })
                .unwrap_or(-1.0);
            let c2 = st.iter().filter(|&&x| x > 2.0).count();
            let c25 = st.iter().filter(|&&x| x > 2.5).count();
            let c3 = st.iter().filter(|&&x| x > 3.0).count();
            let wv = *vs.iter().max_by(|&&a, &&b| ev.max_stretch[a].partial_cmp(&ev.max_stretch[b]).unwrap()).unwrap();
            // region volume ratio per pose (min over poses)
            let jj = j.unwrap_or(0);
            let tris: Vec<[u32; 3]> =
                m.tris.iter().filter(|t| t.iter().all(|&i| ctx.vregion[i as usize] as usize == ri)).cloned().collect();
            let vol = |pos: &dyn Fn(usize) -> weightforge_core::math::Vec3, c: weightforge_core::math::Vec3| -> f64 {
                tris.iter()
                    .map(|t| {
                        let a = pos(t[0] as usize) - c;
                        let b = pos(t[1] as usize) - c;
                        let d = pos(t[2] as usize) - c;
                        a.dot(b.cross(d)) / 6.0
                    })
                    .sum()
            };
            let c0 = (m.skel.head[jj] + m.skel.tail[jj]) / 2.0;
            let v0 = vol(&|i| m.rest[i], c0);
            let mut vmin = 9.0f64;
            let mut vpose = String::new();
            for (pi, mats) in ctx.mats.iter().enumerate() {
                let posed = weightforge_core::skin::lbs(&m, mats, &m.weights);
                let c = mats[jj].transform_point(c0);
                let r = vol(&|i| posed[i], c) / v0;
                if r < vmin {
                    vmin = r;
                    vpose = ctx.poses[pi].name.clone();
                }
            }
            println!(
                "   >2:{c2} >2.5:{c25} >3:{c3} worst v{wv} at {:?} pose {} | vol min {:.2} in {}",
                m.rest[wv], ctx.poses[ev.worst_pose[wv] as usize].name, vmin, vpose
            );
            let noise = vs.iter().filter(|&&v| ev.flags[v] & F_NOISE != 0).count();
            let bleed = vs.iter().filter(|&&v| ev.flags[v] & F_BLEED != 0).count();
            println!(
                "{:<18} n{:>4} stretch max {:.2} p5 {:.2} p10 {:.2} | thin min {:.2} p5 {:.2} ext {:.2} | noise {} bleed {}",
                r.name,
                vs.len(),
                st[0],
                st[(vs.len() / 20).min(vs.len() - 1)],
                st[(vs.len() / 10).min(vs.len() - 1)],
                th[0],
                th[(vs.len() / 20).min(vs.len() - 1)],
                ext,
                noise,
                bleed
            );
        }
    }
}
