//! How far weightforge's role transfer of a source's clips lands from the
//! clips already in an animated file of the same rig (motionforge
//! `animate`'s output): per clip, the largest joint-head distance (cm on a
//! 1.75 m body) and rotation difference (degrees), relative to the hips.
//!     cargo run --release --example transfer_vs_file -- animated.glb source.glb [clip ...]
use std::path::Path;
use weightforge_core as wf;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dst = wf::scene::Model::load(Path::new(&a[0])).unwrap();
    let src = wf::scene::Model::load(Path::new(&a[1])).unwrap();
    let clips: Vec<String> = if a.len() > 2 { a[2..].to_vec() } else { wf::motion::GAME_CLIPS.iter().map(|s| s.to_string()).collect() };
    let own = wf::poses::clip_frames(&dst, wf::motion::FPS);
    let t = wf::motion::transfer(&src, "src", &dst, &clips, wf::motion::FPS).unwrap();
    let k = wf::owner::BODY_M / wf::owner::height(&dst);
    let hips = dst.skel.anatomy.roles["hips"][0];
    let mut i = 0;
    for c in &t.clips {
        let Some((_, frames)) = own.iter().find(|x| &x.0 == c) else { continue };
        let (mut dmax, mut amax, mut worst) = (0.0f64, 0.0f64, String::new());
        for f in frames {
            let tp = &t.poses[i];
            i += 1;
            let ma = wf::skin::joint_matrices(&dst, f);
            let mb = wf::skin::joint_matrices(&dst, tp);
            let (ha, hb) = (ma[hips].transform_point(dst.skel.head[hips]), mb[hips].transform_point(dst.skel.head[hips]));
            for j in 0..dst.njoints() {
                let pa = ma[j].transform_point(dst.skel.head[j]) - ha;
                let pb = mb[j].transform_point(dst.skel.head[j]) - hb;
                let d = (pa - pb).len() * k * 100.0;
                let ang = 2.0 * ma[j].rotation().dot(mb[j].rotation()).abs().min(1.0).acos().to_degrees();
                if d > dmax {
                    dmax = d;
                    worst = format!("{} {}", dst.skel.names[j], f.name);
                }
                amax = amax.max(ang);
            }
        }
        println!("{c}: {} frames, largest head gap {dmax:.1} cm ({worst}), largest turn gap {amax:.1} deg", frames.len());
        if std::env::var_os("PER_BONE").is_some() {
            let f = &frames[frames.len() / 2];
            let tp = &t.poses[i - frames.len() + frames.len() / 2];
            let ma = wf::skin::joint_matrices(&dst, f);
            let mb = wf::skin::joint_matrices(&dst, tp);
            for j in 0..dst.njoints() {
                let ang = 2.0 * ma[j].rotation().dot(mb[j].rotation()).abs().min(1.0).acos().to_degrees();
                println!("   {:<24} {ang:>6.1} deg", dst.skel.names[j]);
            }
        }
    }
}
