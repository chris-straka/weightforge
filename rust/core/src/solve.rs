//! Sparse symmetric positive-definite solves (Jacobi-preconditioned
//! conjugate gradients) and harmonic weight inpainting on a vertex subset.

use crate::scene::{VW, Weights, normalize_vw};
use rayon::prelude::*;

/// Symmetric sparse matrix in CSR form.
pub struct Csr {
    pub n: usize,
    pub start: Vec<usize>,
    pub col: Vec<u32>,
    pub val: Vec<f64>,
}

impl Csr {
    pub fn mul(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.n {
            let mut s = 0.0;
            for k in self.start[i]..self.start[i + 1] {
                s += self.val[k] * x[self.col[k] as usize];
            }
            y[i] = s;
        }
    }
    pub fn diag(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| (self.start[i]..self.start[i + 1]).find(|&k| self.col[k] as usize == i).map(|k| self.val[k]).unwrap_or(1.0))
            .collect()
    }
}

pub fn cg(a: &Csr, b: &[f64], x: &mut [f64], tol: f64, max_iter: usize) -> usize {
    let n = a.n;
    let d = a.diag();
    let mut r = vec![0.0; n];
    a.mul(x, &mut r);
    for i in 0..n {
        r[i] = b[i] - r[i];
    }
    let bn = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
    let mut z: Vec<f64> = (0..n).map(|i| r[i] / d[i]).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let mut ap = vec![0.0; n];
    for it in 0..max_iter {
        if r.iter().map(|v| v * v).sum::<f64>().sqrt() / bn < tol {
            return it;
        }
        a.mul(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-300 {
            return it;
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        for i in 0..n {
            z[i] = r[i] / d[i];
        }
        let rz2: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz2 / rz;
        rz = rz2;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    max_iter
}

/// Harmonic inpainting: vertices with `known[v] = Some(w)` keep `w`; the
/// unknown vertices of `subset` get the solution of the graph Laplace
/// equation (each vertex the mean of its neighbors), solved per bone.
/// Unknown components that touch no known vertex get `fallback(v)`.
pub fn inpaint(adj: &[Vec<u32>], subset: &[u32], known: &[Option<VW>], fallback: &dyn Fn(usize) -> VW) -> Weights {
    let nv = adj.len();
    let mut in_set = vec![false; nv];
    for &v in subset {
        in_set[v as usize] = true;
    }
    let mut uidx = vec![u32::MAX; nv];
    let mut unknown: Vec<u32> = Vec::new();
    for &v in subset {
        if known[v as usize].is_none() {
            uidx[v as usize] = unknown.len() as u32;
            unknown.push(v);
        }
    }
    let mut out: Weights = vec![Vec::new(); nv];
    for &v in subset {
        if let Some(k) = &known[v as usize] {
            out[v as usize] = k.clone();
        }
    }
    if unknown.is_empty() {
        return out;
    }
    // Components of the unknown set that reach a known vertex.
    let mut comp_ok = vec![false; unknown.len()];
    let mut seen = vec![false; unknown.len()];
    for s in 0..unknown.len() {
        if seen[s] {
            continue;
        }
        let mut stack = vec![s];
        let mut members = Vec::new();
        let mut touches = false;
        seen[s] = true;
        while let Some(i) = stack.pop() {
            members.push(i);
            for &u in &adj[unknown[i] as usize] {
                let u = u as usize;
                if !in_set[u] {
                    continue;
                }
                if known[u].is_some() {
                    touches = true;
                } else if !seen[uidx[u] as usize] {
                    seen[uidx[u] as usize] = true;
                    stack.push(uidx[u] as usize);
                }
            }
        }
        for m in members {
            comp_ok[m] = touches;
        }
    }
    let solve_set: Vec<usize> = (0..unknown.len()).filter(|&i| comp_ok[i]).collect();
    let mut sidx = vec![u32::MAX; unknown.len()];
    for (k, &i) in solve_set.iter().enumerate() {
        sidx[i] = k as u32;
    }
    // L_UU (uniform graph Laplacian restricted to the solvable unknowns).
    let n = solve_set.len();
    let mut start = Vec::with_capacity(n + 1);
    let mut col = Vec::new();
    let mut val = Vec::new();
    start.push(0);
    for &i in &solve_set {
        let v = unknown[i] as usize;
        let mut row: Vec<(u32, f64)> = Vec::new();
        let mut deg = 0.0f64;
        for &u in &adj[v] {
            let u = u as usize;
            if !in_set[u] {
                continue;
            }
            deg += 1.0;
            if known[u].is_none() && comp_ok[uidx[u] as usize] {
                row.push((sidx[uidx[u] as usize], -1.0));
            }
        }
        row.push((sidx[i], deg.max(1.0)));
        row.sort_by_key(|e| e.0);
        for (c, x) in row {
            col.push(c);
            val.push(x);
        }
        start.push(col.len());
    }
    let a = Csr { n, start, col, val };
    let mut bones: Vec<u32> = subset.iter().filter_map(|&v| known[v as usize].as_ref()).flat_map(|k| k.iter().map(|e| e.0)).collect();
    bones.sort_unstable();
    bones.dedup();
    let cols: Vec<(u32, Vec<f64>)> = bones
        .par_iter()
        .map(|&b| {
            let rhs: Vec<f64> = solve_set
                .iter()
                .map(|&i| {
                    adj[unknown[i] as usize]
                        .iter()
                        .filter_map(|&u| known[u as usize].as_ref().filter(|_| in_set[u as usize]))
                        .map(|k| crate::scene::weight_of(k, b))
                        .sum()
                })
                .collect();
            let mut x = vec![0.0; n];
            cg(&a, &rhs, &mut x, 1e-8, 4000);
            (b, x)
        })
        .collect();
    for (k, &i) in solve_set.iter().enumerate() {
        let v = unknown[i] as usize;
        out[v] = normalize_vw(cols.iter().map(|(b, x)| (*b, x[k].max(0.0))).filter(|e| e.1 > 1e-6).collect());
    }
    for (i, &v) in unknown.iter().enumerate() {
        if !comp_ok[i] {
            out[v as usize] = fallback(v as usize);
        }
    }
    out
}
