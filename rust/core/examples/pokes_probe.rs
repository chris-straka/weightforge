//! Where a rig pokes through itself: vertices flagged `intersect`, in all
//! and away from the cut between its legs, and per pose. The cut's plane
//! is read from the file (`extras.unfuse.legs`, meshkit's unfuse), or from
//! a second file, so an uncut rig can be counted against its cut copy's
//! plane; with neither, the plane through the hip joints. "Away" is more
//! than 5 mm from the plane, or above the hip joints: everything but the
//! cut's inner walls and the rim they hang from. One JSON line.
//!
//!     cargo run --release --example pokes_probe -- rig.glb [plane-from.glb]

use std::path::Path;
use weightforge_core as wf;
use wf::math::Vec3;

fn plane_of(model: &wf::scene::Model) -> Option<(Vec3, Vec3)> {
    let vec = |v: Option<&serde_json::Value>| -> Option<Vec3> {
        let a = v?.as_array()?;
        Some(Vec3 { x: a.first()?.as_f64()?, y: a.get(1)?.as_f64()?, z: a.get(2)?.as_f64()? })
    };
    model.glb.arr("meshes").iter().find_map(|m| {
        let l = m.get("extras")?.get("unfuse")?.get("legs")?;
        Some((vec(l.get("point"))?, vec(l.get("normal"))?.normalized()))
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: pokes_probe <rig.glb> [plane-from.glb]");
        std::process::exit(2);
    };
    let (model, ctx, ev, rep) = wf::check(Path::new(path), &wf::metrics::CtxOpts::default()).expect("rig");
    let legs = wf::unfuse::Legs::of(&model);
    let other = args.get(1).map(|p| wf::scene::Model::load(Path::new(p)).expect("plane-from rig"));
    let plane = plane_of(&model).or_else(|| other.as_ref().and_then(plane_of)).or_else(|| legs.as_ref().map(|l| (l.center, l.lateral)));
    let hips = legs.as_ref().map(|l| l.center.y).unwrap_or(f64::MAX);
    let near_cut = |v: usize| -> bool {
        let Some((p, n)) = plane else { return false };
        model.rest[v].y < hips && (model.rest[v] - p).dot(n).abs() < 0.005
    };
    let flagged: Vec<usize> = (0..model.nverts()).filter(|&v| ev.flags[v] & wf::metrics::F_INTERSECT != 0).collect();
    let mut poses = serde_json::Map::new();
    for (p, pose) in ctx.poses.iter().enumerate() {
        let hit: Vec<usize> = ev.bad_in_pose[p].iter().filter(|x| x.1 & wf::metrics::F_INTERSECT != 0).map(|x| x.0 as usize).collect();
        if !hit.is_empty() {
            poses.insert(pose.name.clone(), serde_json::json!([hit.len(), hit.iter().filter(|&&v| !near_cut(v)).count()]));
        }
    }
    let away = flagged.iter().filter(|&&v| !near_cut(v)).count();
    let verts_away = (0..model.nverts()).filter(|&v| !near_cut(v)).count();
    println!(
        "{}",
        serde_json::json!({
            "rig": rep.file, "verts": model.nverts(), "pokes": flagged.len(), "pokes_away_from_cut": away,
            "verts_away_from_cut": verts_away,
            "pokes_pct": (1e4 * flagged.len() as f64 / model.nverts().max(1) as f64).round() / 100.0,
            "pokes_away_pct": (1e4 * away as f64 / verts_away.max(1) as f64).round() / 100.0,
            "poses": poses,
        })
    );
}
