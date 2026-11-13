//! Connected shells of a rig's welded mesh: vertex count, size (% of the
//! body's height), centre height (%), and the bones that own them.
//!     cargo run --release --example shells -- a.glb
use std::path::Path;
use weightforge_core as wf;
fn main() {
    for p in std::env::args().skip(1) {
        let m = wf::scene::Model::load(Path::new(&p)).unwrap();
        let n = m.nverts();
        let mut comp = vec![usize::MAX; n];
        let mut shells: Vec<Vec<usize>> = Vec::new();
        for s in 0..n {
            if comp[s] != usize::MAX {
                continue;
            }
            let id = shells.len();
            let mut st = vec![s];
            comp[s] = id;
            let mut vs = Vec::new();
            while let Some(v) = st.pop() {
                vs.push(v);
                for &u in &m.adj[v] {
                    if comp[u as usize] == usize::MAX {
                        comp[u as usize] = id;
                        st.push(u as usize);
                    }
                }
            }
            shells.push(vs);
        }
        let (mut lo, mut hi) = (m.rest[0], m.rest[0]);
        for q in &m.rest {
            lo = lo.min(*q);
            hi = hi.max(*q);
        }
        let h = hi.y - lo.y;
        shells.sort_by_key(|s| std::cmp::Reverse(s.len()));
        println!("== {} : {} verts, {} shells", Path::new(&p).file_name().unwrap().to_string_lossy(), n, shells.len());
        for s in shells.iter().take(25) {
            let (mut a, mut b) = (m.rest[s[0]], m.rest[s[0]]);
            for &v in s {
                a = a.min(m.rest[v]);
                b = b.max(m.rest[v]);
            }
            let mut own = std::collections::BTreeMap::<u32, usize>::new();
            for &v in s {
                if let Some(&(j, _)) = m.weights[v].first() {
                    *own.entry(j).or_default() += 1;
                }
            }
            let mut o: Vec<_> = own.into_iter().collect();
            o.sort_by_key(|x| std::cmp::Reverse(x.1));
            let names: Vec<String> = o.iter().take(3).map(|(j, c)| format!("{}:{}", m.skel.names[*j as usize], c)).collect();
            println!(
                "  {:>6} verts size {:>5.1}% at {:>5.1}%  {}",
                s.len(),
                100.0 * (b - a).len() / h,
                100.0 * (0.5 * (a.y + b.y) - lo.y) / h,
                names.join(" ")
            );
        }
    }
}
