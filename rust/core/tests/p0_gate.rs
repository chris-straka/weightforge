//! P0 gate: every injected fault is found, in the right region, and the
//! clean twin raises no failing finding. Faulted fixtures may only fail in
//! regions the fault touches.

use weightforge_core::fixtures;
use weightforge_core::glb::Glb;
use weightforge_core::metrics::{Ctx, CtxOpts, evaluate};
use weightforge_core::report::{Report, build};
use weightforge_core::scene::Model;

pub fn fixture_model(fault: &str) -> Model {
    let bytes = fixtures::to_glb(&fixtures::build(fault)).to_bytes();
    Model::from_glb(Glb::parse(&bytes).unwrap()).unwrap()
}

fn report(fault: &str) -> Report {
    let m = fixture_model(fault);
    let ctx = Ctx::new(&m, &CtxOpts::default());
    let ev = evaluate(&m, &ctx, &m.weights);
    build(fault, &m, &ctx, &ev)
}

fn fails(r: &Report) -> Vec<(String, String)> {
    r.findings.iter().filter(|f| f.severity == "fail").map(|f| (f.code.clone(), f.region.clone())).collect()
}

/// (fault, must find (code, region) — any of, allowed failing regions)
const CASES: &[(&str, &[(&str, &str)], &[&str])] = &[
    (
        "noise",
        &[("D_NOISE", "DEF-spine.001"), ("D_NOISE", "DEF-spine.002")],
        &["DEF-spine.001", "DEF-spine.002", "DEF-spine.003", "DEF-upper_arm.L", "DEF-upper_arm.R", "DEF-spine"],
    ),
    ("bleed", &[("D_BLEED", "DEF-hand.L")], &["DEF-hand.L", "DEF-forearm.L"]),
    ("no_elbow_falloff", &[("D_STRETCH", "DEF-forearm.L"), ("D_STRETCH", "DEF-upper_arm.L")], &["DEF-forearm.L", "DEF-upper_arm.L"]),
    ("cape_wrong_bone", &[("D_PIECE_FOLLOW", "piece:Cape")], &["piece:Cape"]),
    (
        "elbow_collapse",
        &[("D_STRETCH", "DEF-forearm.R"), ("D_VOLUME", "DEF-forearm.R"), ("D_VOLUME", "DEF-upper_arm.R")],
        &["DEF-forearm.R", "DEF-upper_arm.R"],
    ),
    ("knee_wide_falloff", &[("D_VOLUME", "DEF-shin.R"), ("D_VOLUME", "DEF-thigh.R")], &["DEF-shin.R", "DEF-thigh.R"]),
];

#[test]
fn clean_twin_has_no_failing_findings() {
    let r = report("clean");
    assert!(r.pass, "clean twin failed: {:?}", fails(&r));
}

#[test]
fn every_fault_found_in_its_region_only() {
    for (fault, want, allowed) in CASES {
        let r = report(fault);
        let f = fails(&r);
        assert!(!r.pass, "{fault}: not detected");
        assert!(want.iter().any(|(c, reg)| f.iter().any(|(fc, fr)| fc == c && fr == reg)), "{fault}: expected one of {want:?}, got {f:?}");
        for (c, reg) in &f {
            assert!(allowed.contains(&reg.as_str()), "{fault}: false alarm {c} in {reg}");
        }
    }
}

#[test]
fn every_fixture_listed_is_tested() {
    for f in fixtures::FAULTS {
        assert!(*f == "clean" || CASES.iter().any(|c| c.0 == *f), "fixture {f} has no gate case");
    }
}

#[test]
fn every_fault_scores_below_the_clean_twin() {
    let clean = report("clean").score;
    for (fault, _, _) in CASES {
        let s = report(fault).score;
        assert!(s < clean - 5.0, "{fault}: score {s} not clearly below clean {clean}");
    }
}
