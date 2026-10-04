use weightforge_core::{metrics::*, scene::Model};
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let m = Model::load(std::path::Path::new(&p)).unwrap();
    let mut shown = 0;
    for v in 0..m.nverts() {
        let n = noise_of(&m, &m.weights, v);
        if n > 0.5 && shown < 4 {
            shown += 1;
            println!("v{v} noise {n:.2} pos {:?} w {:?} part {}", m.rest[v], m.weights[v], m.vpart[v]);
            for &u in &m.adj[v] {
                println!("   n{u} {:?} w {:?}", m.rest[u as usize], m.weights[u as usize]);
            }
        }
    }
}
