use weightforge_core::{metrics::*, scene::Model};
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let m = Model::load(std::path::Path::new(&p)).unwrap();
    let ctx = Ctx::new(&m, &CtxOpts { intersect: false, ..Default::default() });
    let n = weightforge_core::scene::vertex_normals(&m.rest, &m.tris);
    let idx = weightforge_core::transfer::TriIndex::new(
        &m.rest,
        m.tris.iter().filter(|t| t.iter().all(|&i| m.vpart[i as usize] == 0)).copied().collect(),
    );
    for (k, mm) in ctx.piece_matches.iter().enumerate() {
        if k % 13 != 0 && !mm.3 {
            continue;
        }
        let v = mm.0 as usize;
        let h = idx.closest(m.rest[v]).unwrap();
        let t = idx.tri(h.tri);
        let bn = (n[t[0] as usize] + n[t[1] as usize] + n[t[2] as usize]).normalized();
        println!("v{v} {:?} dist {:.3} cos {:.2} matched {} body {:?}", m.rest[v], h.dist, bn.dot(n[v]), mm.3, m.rest[t[0] as usize]);
    }
}
