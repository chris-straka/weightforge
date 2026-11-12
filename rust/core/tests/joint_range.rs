//! A weight on a joint index past the end of the skin's joint list (the
//! file is invalid there: an engine indexes garbage or refuses it) used to
//! be dropped and the rest renormalised without a word, so the check
//! measured a different skin from the one in the file. It must be said.

use weightforge_core::fixtures;
use weightforge_core::metrics::{Ctx, CtxOpts, evaluate};
use weightforge_core::report::build;
use weightforge_core::scene::Model;

#[test]
fn weights_on_missing_joints_are_reported() {
    let mut glb = fixtures::to_glb(&fixtures::build("clean"));
    // The body's JOINTS_0 (u8 x 4 per vertex): point the first influence
    // of the first five vertices at joint 200 of a 20-joint skin.
    let acc = glb.json["meshes"][0]["primitives"][0]["attributes"]["JOINTS_0"].as_u64().unwrap() as usize;
    let view = glb.json["accessors"][acc]["bufferView"].as_u64().unwrap() as usize;
    let off = glb.json["bufferViews"][view]["byteOffset"].as_u64().unwrap_or(0) as usize;
    for v in 0..5 {
        glb.bin[off + v * 4] = 200;
    }
    let m = Model::from_glb(glb).unwrap();
    let ctx = Ctx::new(&m, &CtxOpts::default());
    let rep = build("t", &m, &ctx, &evaluate(&m, &ctx, &m.weights));
    let f = rep.findings.iter().find(|f| f.code == "D_JOINT_RANGE");
    assert!(f.is_some_and(|f| f.verts == 5 && f.detail.contains("200")), "{:#?}", rep.findings);
    // The clean file has none.
    let m = Model::from_glb(fixtures::to_glb(&fixtures::build("clean"))).unwrap();
    let ctx = Ctx::new(&m, &CtxOpts::default());
    let rep = build("t", &m, &ctx, &evaluate(&m, &ctx, &m.weights));
    assert!(!rep.findings.iter().any(|f| f.code == "D_JOINT_RANGE"));
}
