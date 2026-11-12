//! Legs apart (`unfuse.rs`): which side of the body an outline lies on,
//! weight on the wrong leg's bones going home, and nothing at all
//! happening to a body that meshkit's `unfuse.py` did not cut.

use weightforge_core::fixtures;
use weightforge_core::scene::{Model, VW, normalize_vw};
use weightforge_core::unfuse::{Legs, crotch_band, limbs_apart, outline_side, own_legs};

fn mannequin() -> Model {
    Model::from_glb(fixtures::to_glb(&fixtures::build("clean"))).unwrap()
}

/// A shin vertex per leg, well below the knee.
fn shin_vertices(m: &Model, legs: &Legs) -> (usize, usize) {
    let find = |sign: f64| {
        (0..m.nverts())
            .filter(|&v| (m.rest[v].x - legs.center.x) * sign > 0.03 && m.rest[v].y < 0.35 && m.rest[v].y > 0.2)
            .min_by(|&a, &b| (m.rest[a].y - 0.28).abs().total_cmp(&(m.rest[b].y - 0.28).abs()))
            .expect("a shin vertex")
    };
    (find(1.0), find(-1.0))
}

#[test]
fn outlines_know_their_leg() {
    let m = mannequin();
    let legs = Legs::of(&m).expect("a humanoid");
    let side = outline_side(&m, &legs);
    let (l, r) = shin_vertices(&m, &legs);
    assert_eq!((side[l], side[r]), (1, 2), "a shin's outline lies on its own side of the middle plane");
    // The trunk's outline reaches across: neither leg's.
    let belly = (0..m.nverts()).find(|&v| m.rest[v].y > legs.center.y + 0.1 && m.rest[v].y < legs.center.y + 0.2).unwrap();
    assert_eq!(side[belly], 0);
}

#[test]
fn weight_on_the_wrong_leg_goes_home() {
    let mut m = mannequin();
    let legs = Legs::of(&m).unwrap();
    let (l, _) = shin_vertices(&m, &legs);
    let clean = m.weights.clone();
    // A clean body: nothing to move.
    assert!(own_legs(&m, &clean) == clean, "a clean mannequin has no weight on the wrong leg");
    // Plant the fault: half of a left shin vertex's weight on the right leg's bones.
    let moved: VW = clean[l].iter().map(|&(j, x)| (legs.mirror[j as usize].map_or(j, |k| k as u32), 0.5 * x)).collect();
    let mut planted: VW = clean[l].iter().map(|&(j, x)| (j, 0.5 * x)).collect();
    planted.extend(moved);
    m.weights[l] = normalize_vw(planted);
    assert!(m.weights[l].iter().any(|&(j, x)| legs.side[j as usize] == 2 && x > 0.1));
    let fixed = own_legs(&m, &m.weights);
    assert!(fixed[l].iter().all(|&(j, _)| legs.side[j as usize] != 2), "the right leg's bones are gone from a left shin vertex");
    for &(j, x) in &clean[l] {
        let got = fixed[l].iter().find(|e| e.0 == j).map_or(0.0, |e| e.1);
        assert!((got - x).abs() < 1e-6, "and its own bones have the weight back");
    }
    for v in (0..m.nverts()).filter(|&v| v != l) {
        assert!(fixed[v] == m.weights[v], "no other vertex changes");
    }
}

#[test]
fn an_uncut_body_is_left_alone() {
    let m = mannequin();
    assert!(crotch_band(&m, &m.weights) == m.weights);
    assert!(limbs_apart(&m, &m.weights) == m.weights);
}

#[test]
fn a_recorded_cut_gets_its_crotch_band() {
    let mut glb = fixtures::to_glb(&fixtures::build("clean"));
    let legs_y = {
        let m = Model::from_glb(fixtures::to_glb(&fixtures::build("clean"))).unwrap();
        Legs::of(&m).unwrap().center.y
    };
    // What unfuse.py writes after a cut up to the hip joints.
    glb.json["meshes"][0]["extras"] = serde_json::json!({"unfuse": {"legs": {"crotch": legs_y}}});
    let m = Model::from_glb(glb).unwrap();
    let legs = Legs::of(&m).unwrap();
    let w = crotch_band(&m, &m.weights);
    let leg_share = |vw: &VW| vw.iter().filter(|e| legs.side[e.0 as usize] != 0).map(|e| e.1).sum::<f64>();
    // Skin on the middle plane at and above the crotch follows no thigh.
    let mut seen = 0;
    for v in 0..m.nverts() {
        let q = m.rest[v] - legs.center;
        if q.x.abs() < 0.005 && q.y >= 0.0 && q.y < 0.1 {
            seen += 1;
            assert!(leg_share(&w[v]) < 0.02, "vertex {v} at {:?} keeps {:.2} of thigh", m.rest[v], leg_share(&w[v]));
        }
        // Leg weight is only ever capped.
        assert!(leg_share(&w[v]) <= leg_share(&m.weights[v]) + 1e-9);
    }
    assert!(seen > 0, "the mannequin has skin on its middle plane above the crotch");
    // The shins are far below the band: untouched.
    let (l, r) = shin_vertices(&m, &legs);
    assert!(w[l] == m.weights[l] && w[r] == m.weights[r]);
}

#[test]
fn the_start_weights_survive_a_write() {
    // A cut file whose left shin carries right-leg weight, as a cut leaves it.
    let mut f = fixtures::build("clean");
    let probe = Model::from_glb(fixtures::to_glb(&f)).unwrap();
    let legs = Legs::of(&probe).unwrap();
    let (l, _) = shin_vertices(&probe, &legs);
    let b = (0..f.body_pos.len())
        .min_by(|&x, &y| (f.body_pos[x] - probe.rest[l]).len().total_cmp(&(f.body_pos[y] - probe.rest[l]).len()))
        .unwrap();
    let moved: VW = f.body_w[b].iter().map(|&(j, x)| (legs.mirror[j as usize].map_or(j, |k| k as u32), 0.5 * x)).collect();
    let mut planted: VW = f.body_w[b].iter().map(|&(j, x)| (j, 0.5 * x)).collect();
    planted.extend(moved);
    f.body_w[b] = normalize_vw(planted);
    let mut glb = fixtures::to_glb(&f);
    glb.json["meshes"][0]["extras"] = serde_json::json!({"unfuse": {"legs": {"crotch": legs.center.y}}});
    let bytes = glb.to_bytes();
    let read = || Model::from_glb(weightforge_core::glb::Glb::parse(&bytes).unwrap()).unwrap();
    let m = read();
    assert!(m.weights[l].iter().any(|&(j, x)| legs.side[j as usize] == 2 && x > 0.1), "the fault is in the file");
    let start = weightforge_core::unfuse::cut_start(&m).expect("a cut file with weight on the wrong leg");
    // Written from the model as read (the writer keeps the file's bytes
    // where weights equal the loaded ones), the start weights are in the file.
    let out = m.write_weights(&start).unwrap().to_bytes();
    let back = Model::from_glb(weightforge_core::glb::Glb::parse(&out).unwrap()).unwrap();
    assert!(back.weights[l].iter().all(|&(j, _)| legs.side[j as usize] != 2));
    // Written from a model whose weights were replaced first, they are lost:
    // the reason `weights fix` keeps an untouched copy to write from.
    let mut swapped = read();
    swapped.weights = start.clone();
    let lost = swapped.write_weights(&start).unwrap().to_bytes();
    let back = Model::from_glb(weightforge_core::glb::Glb::parse(&lost).unwrap()).unwrap();
    assert!(back.weights[l].iter().any(|&(j, x)| legs.side[j as usize] == 2 && x > 0.1));
}
