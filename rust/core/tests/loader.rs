//! The loader's three fixes of 2026-11-13: every skin is read, meshes
//! parented to a bone without a skin are checked, and each bone's tail is
//! decided on its own.

use serde_json::json;
use weightforge_core::fixtures::{self, Bone};
use weightforge_core::glb::{Glb, f32_bytes};
use weightforge_core::math::{Quat, Vec3, v3};
use weightforge_core::metrics::{Ctx, CtxOpts, evaluate};
use weightforge_core::report::build;
use weightforge_core::scene::Model;

fn load(g: Glb) -> Model {
    Model::from_glb(Glb::parse(&g.to_bytes()).unwrap()).unwrap()
}

/// The cape moved to a second skin listing the joints in reverse order:
/// the same character, so the same weights and the same report.
#[test]
fn every_skin_is_read() {
    let f = fixtures::build("clean");
    let one = load(fixtures::to_glb(&f));
    let mut g = fixtures::to_glb(&f);
    let nb = f.bones.len();
    let skin0 = g.json["skins"][0].clone();
    let ibm0 = skin0["inverseBindMatrices"].as_u64().unwrap() as usize;
    let (_, m) = g.read_f64(ibm0).unwrap();
    let rev: Vec<f32> = (0..nb).rev().flat_map(|j| m[j * 16..j * 16 + 16].iter().map(|&x| x as f32).collect::<Vec<_>>()).collect();
    let bv = g.append_view(&f32_bytes(&rev), None);
    let acc = g.push_accessor(json!({"bufferView": bv, "componentType": 5126, "count": nb, "type": "MAT4"}));
    let joints: Vec<u64> = skin0["joints"].as_array().unwrap().iter().rev().map(|x| x.as_u64().unwrap()).collect();
    g.json["skins"].as_array_mut().unwrap().push(json!({"inverseBindMatrices": acc, "joints": joints}));
    let cape_node = g.json["nodes"].as_array().unwrap().iter().position(|n| n["name"] == "Cape").unwrap();
    g.json["nodes"][cape_node]["skin"] = json!(1);
    let cape_mesh = g.json["nodes"][cape_node]["mesh"].as_u64().unwrap() as usize;
    let ja = g.json["meshes"][cape_mesh]["primitives"][0]["attributes"]["JOINTS_0"].as_u64().unwrap() as usize;
    let wa = g.json["meshes"][cape_mesh]["primitives"][0]["attributes"]["WEIGHTS_0"].as_u64().unwrap() as usize;
    let (_, jv) = g.read_f64(ja).unwrap();
    let (_, wv) = g.read_f64(wa).unwrap();
    let rows: Vec<[f64; 4]> =
        jv.chunks(4).zip(wv.chunks(4)).map(|(j, w)| [0, 1, 2, 3].map(|k| if w[k] > 0.0 { (nb - 1) as f64 - j[k] } else { 0.0 })).collect();
    assert!(g.overwrite_vec4(ja, &rows).unwrap());
    let two = load(g);
    assert_eq!(two.skins.len(), 2);
    assert_eq!(two.nverts(), one.nverts());
    for v in 0..one.nverts() {
        for &(j, w) in &one.weights[v] {
            let w2 = weightforge_core::scene::weight_of(&two.weights[v], j);
            assert!((w - w2).abs() < 1e-6, "vertex {v} joint {j}: {w} vs {w2}");
        }
        assert!((one.rest[v] - two.rest[v]).len() < 1e-6);
    }
    let rep = |m: &Model| {
        let ctx = Ctx::new(m, &CtxOpts::default());
        let ev = evaluate(m, &ctx, &m.weights);
        build("t", m, &ctx, &ev)
    };
    let (r1, r2) = (rep(&one), rep(&two));
    assert_eq!(r1.score, r2.score);
    assert_eq!(r1.findings.len(), r2.findings.len());
    // Written back, the second skin keeps its own numbering.
    let back = load(two.write_weights(&two.weights).unwrap());
    for v in 0..one.nverts() {
        for &(j, w) in &two.weights[v] {
            assert!((weightforge_core::scene::weight_of(&back.weights[v], j) - w).abs() < 1e-6);
        }
    }
}

/// A box parented to the left hand, with no skin: it is loaded as part of
/// the body, rides the hand rigidly, and is never written.
#[test]
fn rigid_parts_are_checked_and_ride_their_bone() {
    let f = fixtures::build("clean");
    let mut g = fixtures::to_glb(&f);
    let hand = f.bones.iter().position(|b| b.name == "DEF-hand.L").unwrap();
    let c = [-0.02f32, 0.02];
    let mut pos: Vec<f32> = Vec::new();
    for &x in &c {
        for &y in &c {
            for &z in &c {
                pos.extend([x, y, z]);
            }
        }
    }
    let tris: [u32; 36] = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3];
    let bvp = g.append_view(&f32_bytes(&pos), Some(34962));
    let ap = g.push_accessor(
        json!({"bufferView": bvp, "componentType": 5126, "count": 8, "type": "VEC3", "min": [-0.02,-0.02,-0.02], "max": [0.02,0.02,0.02]}),
    );
    let idx: Vec<u8> = tris.iter().flat_map(|i| i.to_le_bytes()).collect();
    let bvi = g.append_view(&idx, Some(34963));
    let ai = g.push_accessor(json!({"bufferView": bvi, "componentType": 5125, "count": 36, "type": "SCALAR"}));
    g.json["meshes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name": "Bracer", "primitives": [{"attributes": {"POSITION": ap}, "indices": ai}]}));
    let mi = g.json["meshes"].as_array().unwrap().len() - 1;
    let node = g.json["nodes"].as_array().unwrap().len();
    g.json["nodes"].as_array_mut().unwrap().push(json!({"name": "Bracer", "mesh": mi, "translation": [0.0, 0.05, 0.0]}));
    let hand_node = g.json["skins"][0]["joints"][hand].as_u64().unwrap() as usize;
    let ch = g.json["nodes"][hand_node].get("children").cloned().unwrap_or(json!([]));
    let mut ch: Vec<serde_json::Value> = ch.as_array().unwrap().clone();
    ch.push(json!(node));
    g.json["nodes"][hand_node]["children"] = json!(ch);
    let m = load(g);
    let rigid: Vec<usize> = (0..m.nverts()).filter(|&v| m.rigid[v]).collect();
    assert_eq!(rigid.len(), 8);
    for &v in &rigid {
        assert_eq!(m.weights[v], vec![(hand as u32, 1.0)]);
    }
    let ctx = Ctx::new(&m, &CtxOpts::default());
    let p = ctx.poses.iter().position(|p| p.name == "elbow_140.L").unwrap();
    for &v in &rigid {
        let posed = weightforge_core::skin::lbs_vertex(&ctx.mats[p], &m.weights[v], m.rest[v]);
        let by_hand = ctx.mats[p][hand].transform_point(m.rest[v]);
        assert!((posed - by_hand).len() < 1e-9);
        assert!((posed - m.rest[v]).len() > 0.05, "the bracer moves with the bent arm");
    }
    // A repair never changes it, and writing leaves the mesh without skin.
    let mut w = m.weights.clone();
    w[rigid[0]] = vec![(0, 1.0)];
    m.lock_rigid(&mut w);
    assert_eq!(w[rigid[0]], vec![(hand as u32, 1.0)]);
    let out = m.write_weights(&m.weights).unwrap();
    assert!(out.json["meshes"][mi]["primitives"][0]["attributes"].get("JOINTS_0").is_none());
}

/// A body that ignores Blender's +Y convention (every joint world-aligned)
/// with 50 finger bones that follow it, as a grafted hand does. The tail
/// guess used to be one vote over the skeleton: the fingers outvoted the
/// body, every tail swung to +Y, and the thighs pointed up (a false shin
/// bleed). Decided per joint, the thigh ends at the shin.
#[test]
fn grafted_fingers_do_not_move_the_thighs_tails() {
    let mut f = fixtures::build("clean");
    let body = f.bones.len();
    for side in ["L", "R"] {
        let hand = f.bones.iter().position(|b| b.name == format!("DEF-hand.{side}")).unwrap();
        let (h0, h1) = (f.bones[hand].head, f.bones[hand].tail);
        let d = (h1 - h0).normalized();
        for k in 0..5 {
            let off = v3(0.0, 0.0, 0.012 * (k as f64 - 2.0));
            let mut parent = hand;
            let mut at = h1 + off;
            for seg in 0..5 {
                let name: &'static str = Box::leak(format!("DEF-f{k}.{seg:02}.{side}").into_boxed_str());
                let next = at + d * 0.012;
                f.bones.push(Bone { name, parent: Some(parent), head: at, tail: next });
                parent = f.bones.len() - 1;
                at = next;
            }
        }
    }
    let frame = |i: usize, b: &Bone| if i < body { Quat::IDENTITY } else { fixtures::bone_frame(b) };
    let m = load(fixtures::to_glb_framed(&f, &frame));
    let sk = &m.skel;
    for side in ["L", "R"] {
        let thigh = sk.joint_by_name(&format!("DEF-thigh.{side}")).unwrap();
        let shin = sk.joint_by_name(&format!("DEF-shin.{side}")).unwrap();
        assert!((sk.tail[thigh] - sk.head[shin]).len() < 1e-9, "thigh.{side} tail {:?}", sk.tail[thigh]);
    }
    let ctx = Ctx::new(&m, &CtxOpts::default());
    let ev = evaluate(&m, &ctx, &m.weights);
    let rep = build("t", &m, &ctx, &ev);
    assert!(!rep.findings.iter().any(|f| f.code == "D_BLEED"), "{:#?}", rep.findings);
    let _: Vec3 = sk.head[0];
}

/// A source skeleton named nothing like the target's maps joint for joint
/// by structural role (`fix::map_joints`), so a rival rig's weights (Tripo's
/// Mixamo names onto the standard skeleton) can be offered as a candidate.
#[test]
fn joints_map_by_role_when_names_differ() {
    let f = fixtures::build("clean");
    let dst = load(fixtures::to_glb(&f));
    let mut g = fixtures::build("clean");
    for (i, b) in g.bones.iter_mut().enumerate() {
        b.name = Box::leak(format!("mixamorig_thing_{i}").into_boxed_str());
    }
    let src = load(fixtures::to_glb(&g));
    let map = weightforge_core::fix::map_joints(&src, &dst);
    for (j, m) in map.iter().enumerate() {
        assert_eq!(*m, Some(j as u32), "{}", f.bones[j].name);
    }
}
