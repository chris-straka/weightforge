//! The standard humanoid skeleton (motionforge `standardize`) names the
//! neck `DEF-spine.004` (and `.005`) and the head `DEF-spine.006`, Rigify's
//! numbering. The checker must find them as the neck and the head, test
//! the head poses on them, and bend only the real spine in the spine poses.
//! The mannequin names them `DEF-neck` and `DEF-head`; renamed, it is the
//! same character, so every number must come out the same.

use weightforge_core::fixtures::{self, Fixture};
use weightforge_core::metrics::{Ctx, CtxOpts, evaluate};
use weightforge_core::poses::{HUMANOID_RON, parse_pose_set, resolve};
use weightforge_core::report::{Report, build};
use weightforge_core::scene::Model;

/// The mannequin with the standard skeleton's neck and head names.
fn standard(mut f: Fixture) -> Fixture {
    for b in f.bones.iter_mut() {
        b.name = match b.name {
            "DEF-neck" => "DEF-spine.004",
            "DEF-head" => "DEF-spine.006",
            n => n,
        };
    }
    f
}

fn model(f: &Fixture) -> Model {
    Model::from_glb(fixtures::to_glb(f)).unwrap()
}

fn check(m: &Model) -> Report {
    let ctx = Ctx::new(m, &CtxOpts::default());
    let ev = evaluate(m, &ctx, &m.weights);
    build("t", m, &ctx, &ev)
}

fn moved(m: &Model, pose: &weightforge_core::poses::Pose) -> Vec<String> {
    pose.moves.iter().map(|mv| m.skel.names[mv.0].clone()).collect()
}

#[test]
fn standard_names_get_the_head_poses_and_a_three_bone_spine() {
    let m = model(&standard(fixtures::build("clean")));
    let poses = resolve(&parse_pose_set(HUMANOID_RON).unwrap(), &m.skel);
    let names: Vec<&str> = poses.iter().map(|p| p.name.as_str()).collect();
    for want in ["head_turn.L", "head_turn.R", "head_nod", "head_back"] {
        assert!(names.contains(&want), "{want} missing from {names:?}");
    }
    let nod = poses.iter().find(|p| p.name == "head_nod").unwrap();
    assert_eq!(moved(&m, nod), ["DEF-spine.004", "DEF-spine.006"]);
    // 45 degrees forward over the three spine bones: 15 each.
    let fwd = poses.iter().find(|p| p.name == "spine_forward").unwrap();
    assert_eq!(moved(&m, fwd), ["DEF-spine.001", "DEF-spine.002", "DEF-spine.003"]);
    for mv in &fwd.moves {
        assert!((mv.2.to_degrees() - 15.0).abs() < 1e-9, "{}", mv.2.to_degrees());
    }
}

#[test]
fn renaming_the_neck_and_head_changes_nothing() {
    let plain = check(&model(&fixtures::build("clean")));
    let named = check(&model(&standard(fixtures::build("clean"))));
    assert_eq!(plain.poses, named.poses);
    assert_eq!(plain.score, named.score);
    assert_eq!(plain.pass, named.pass);
    let key = |r: &Report| {
        let mut k: Vec<_> = r.findings.iter().map(|f| (f.code.clone(), f.severity.clone(), f.verts)).collect();
        k.sort();
        k
    };
    assert_eq!(key(&plain), key(&named));
}

/// A hard cut across the neck (the lower neck on the chest, the rest on
/// the head) tears when the head turns: 70 degrees between the two sides
/// of a 5 cm neck is a 5.7 cm jump across 1 cm edges. Built at twice the
/// mannequin's density so the tear is well past the 3.5x line (at 1x it
/// is 3.0x). Its clean twin has no failing finding at the neck or head.
#[test]
fn a_torn_neck_fails_in_a_head_pose() {
    let torn = check(&model(&standard(fixtures::build_at("neck_tear", 2.0))));
    let neck = torn.findings.iter().find(|f| f.severity == "fail" && f.code == "D_STRETCH" && f.region == "DEF-spine.004");
    assert!(neck.is_some_and(|f| f.detail.contains("head_")), "{:#?}", torn.findings);
    let clean = check(&model(&standard(fixtures::build_at("clean", 2.0))));
    let at_head = |r: &Report| {
        r.findings
            .iter()
            .filter(|f| f.severity == "fail" && f.region.starts_with("DEF-spine.00") && f.region.as_str() > "DEF-spine.003")
            .count()
    };
    assert_eq!(at_head(&clean), 0, "{:#?}", clean.findings);
}

/// Names play no part: a neck and head named like nothing the checker
/// knows still get the head poses (the structure says where they are).
#[test]
fn renamed_neck_and_head_still_get_the_head_poses() {
    let mut f = fixtures::build("clean");
    for b in f.bones.iter_mut() {
        b.name = match b.name {
            "DEF-neck" => "DEF-jaw_base",
            "DEF-head" => "DEF-cranium",
            n => n,
        };
    }
    let rep = check(&model(&f));
    assert!(rep.skipped_poses.iter().all(|s| !s.starts_with("head_")), "{:?}", rep.skipped_poses);
    let m = model(&f);
    let poses = resolve(&parse_pose_set(HUMANOID_RON).unwrap(), &m.skel);
    let nod = poses.iter().find(|p| p.name == "head_nod").unwrap();
    assert_eq!(moved(&m, nod), ["DEF-jaw_base", "DEF-cranium"]);
}

/// A pose that cannot run says so: a humanoid has no tail, so a set that
/// wags one lists the pose as not tested (it used to vanish).
#[test]
fn a_pose_that_cannot_run_is_reported() {
    let m = model(&fixtures::build("clean"));
    let set = parse_pose_set(
        r#"PoseSet(name: "t", poses: [
            PoseSpec(name: "wag", moves: [Move(bone: "tail*", motion: Bend(Left, 30))]),
            PoseSpec(name: "nod", moves: [Move(bone: "head", motion: Bend(Forward, 20))]),
        ])"#,
    )
    .unwrap();
    let (poses, skipped) = weightforge_core::poses::resolve_report(&set, &m.skel);
    assert_eq!(poses.len(), 1);
    assert_eq!(skipped, ["wag: no bone for tail*"]);
}
