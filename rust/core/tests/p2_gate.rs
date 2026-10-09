//! P2/P3 gate on the fixtures: weight error vs ground truth drops >= 80%,
//! every faulted region scores better, the output is rfcheck-clean
//! (<= 4 influences, unit sums in the stored type), and nothing but the
//! skinning data changes.

use weightforge_core::fix::{FixOpts, fix, weight_error};
use weightforge_core::fixtures;
use weightforge_core::glb::Glb;
use weightforge_core::metrics::{Ctx, CtxOpts};
use weightforge_core::scene::{Model, Weights};

fn model_of(fault: &str) -> (Model, Vec<u8>) {
    let bytes = fixtures::to_glb(&fixtures::build(fault)).to_bytes();
    (Model::from_glb(Glb::parse(&bytes).unwrap()).unwrap(), bytes)
}

fn truth() -> Weights {
    let (b, c) = fixtures::ground_truth();
    b.into_iter().chain(c).collect()
}

const FAULTS: &[&str] = &["noise", "bleed", "no_elbow_falloff", "cape_wrong_bone", "elbow_collapse", "knee_wide_falloff"];

#[test]
fn fix_recovers_ground_truth_and_scores() {
    let gt = truth();
    let mut lines = Vec::new();
    for fault in FAULTS {
        let (m, _) = model_of(fault);
        assert_eq!(m.nverts(), gt.len(), "welding must keep fixture vertex order");
        let ctx = Ctx::new(&m, &CtxOpts::default());
        let res = fix(fault, &m, &ctx, &FixOpts::default());
        let e0 = weight_error(&m.weights, &gt);
        let e1 = weight_error(&res.weights, &gt);
        let drop = 1.0 - e1 / e0;
        lines.push(format!(
            "{fault}: weight error {e0:.5} -> {e1:.5} ({:.0}% drop), score {:.1} -> {:.1}",
            drop * 100.0,
            res.fix.before.score,
            res.fix.after.score
        ));
        eprintln!("{}", lines.last().unwrap());
        assert!(
            res.report_after.pass,
            "{fault}: still failing: {:?}",
            res.report_after.findings.iter().filter(|f| f.severity == "fail").collect::<Vec<_>>()
        );
        assert!(drop >= 0.8, "{fault}: weight error drop {:.0}% < 80%", drop * 100.0);
        for r in &res.fix.regions {
            if res.report_before.findings.iter().any(|f| f.severity == "fail" && f.region == r.name) {
                assert!(r.after > r.before, "{fault}: region {} not better ({} -> {})", r.name, r.before, r.after);
            }
        }
    }
}

#[test]
fn clean_input_is_left_alone() {
    let (m, bytes) = model_of("clean");
    let ctx = Ctx::new(&m, &CtxOpts::default());
    let res = fix("clean", &m, &ctx, &FixOpts::default());
    assert_eq!(res.fix.verts_changed, 0);
    // Writing unchanged weights back reproduces the file byte for byte.
    assert_eq!(m.write_weights(&res.weights).unwrap().to_bytes(), bytes);
}

#[test]
fn output_is_rig_contract_clean_and_only_weights_change() {
    for fault in FAULTS {
        let (m, bytes) = model_of(fault);
        let ctx = Ctx::new(&m, &CtxOpts::default());
        let res = fix(fault, &m, &ctx, &FixOpts::default());
        let out = m.write_weights(&res.weights).unwrap().to_bytes();
        let back = Model::from_glb(Glb::parse(&out).unwrap()).unwrap();
        assert_eq!(back.bind, m.bind, "{fault}: vertices moved");
        assert_eq!(back.skel.names, m.skel.names, "{fault}: bones changed");
        for (v, vw) in back.weights.iter().enumerate() {
            assert!(vw.len() <= 4, "{fault}: vertex {v} has {} influences", vw.len());
        }
        // Raw stored sums (rfcheck reads the accessors, not our model).
        let g = Glb::parse(&out).unwrap();
        for mesh in g.arr("meshes") {
            for p in mesh["primitives"].as_array().unwrap() {
                let wa = p["attributes"]["WEIGHTS_0"].as_u64().unwrap() as usize;
                let (_, w) = g.read_f64(wa).unwrap();
                for row in w.chunks(4) {
                    let s: f64 = row.iter().sum();
                    assert!((s - 1.0).abs() <= 1e-3, "{fault}: weight sum {s}");
                }
                assert!(p["attributes"].get("JOINTS_1").is_none());
            }
        }
        // Same accessors, same layout: only BIN bytes inside skinning views differ.
        assert_eq!(out.len(), bytes.len(), "{fault}: in-place write changed file size");
        if let Some(rf) = rfcheck() {
            let path = std::env::temp_dir().join(format!("wf_p2_{fault}.glb"));
            std::fs::write(&path, &out).unwrap();
            let st = std::process::Command::new(rf).arg(&path).status().unwrap();
            assert!(st.success(), "{fault}: rfcheck failed");
            let _ = std::fs::remove_file(&path);
        }
    }
}

fn rfcheck() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let p = std::path::PathBuf::from(home).join("Games/_blender/rfcheck/target/release/rfcheck");
    p.exists()
        .then_some(p)
        .or_else(|| std::process::Command::new("rfcheck").arg("--help").output().ok().map(|_| std::path::PathBuf::from("rfcheck")))
}

#[test]
fn partial_evaluation_matches_full() {
    use weightforge_core::metrics::{evaluate_partial, evaluate_with};
    let (m, _) = model_of("bleed");
    let ctx = Ctx::new(&m, &CtxOpts::default());
    // Perturb the left arm and score its regions partially.
    let mut w = m.weights.clone();
    let regions: Vec<u32> =
        ctx.regions.iter().enumerate().filter(|(_, r)| r.name.ends_with(".L") && r.name.contains("arm")).map(|(i, _)| i as u32).collect();
    let mask: Vec<bool> = (0..m.nverts()).map(|v| regions.contains(&ctx.vregion[v])).collect();
    for v in 0..m.nverts() {
        if mask[v] && v % 7 == 0 {
            w[v] = vec![(0, 1.0)];
        }
    }
    let full = evaluate_with(&m, &ctx, &w, false);
    let part = evaluate_partial(&m, &ctx, &w, &mask);
    for v in 0..m.nverts() {
        if mask[v] {
            assert!((full.energy[v] - part.energy[v]).abs() < 1e-12, "vertex {v}: {} vs {}", full.energy[v], part.energy[v]);
            assert_eq!(full.flags[v], part.flags[v]);
        }
    }
}
