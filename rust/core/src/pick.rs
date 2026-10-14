//! Region picking for `fix`: which candidate repairs which regions.
//!
//! A *move* puts one candidate on a set of regions (only their flagged
//! area changes, seam-blended into the input) and is measured once, alone,
//! against the input: the change of every region's mean energy it touches
//! and of every region's flagged-vertex counts. A *plan* is a set of moves
//! on disjoint regions; its region energies are the input's plus the sum
//! of its moves' changes. [`pick`] finds the plan with the fewest failing
//! findings (the gate), then the lowest score energy (`0.5 * vertex mean +
//! 0.5 * worst region`, the formula `check` reports), exactly, by branch
//! and bound, subject to two rules: no region above its cap (`1.1 x input
//! + slack`), and no region gaining a failing finding it did not have.
//!
//! Why this shape: every move is measured against the input, never against
//! an earlier pick, so a candidate's numbers do not depend on the order
//! regions are visited or on the other candidates; and the search is exact,
//! so nothing gets locked in. Lowering any move's energy changes (a better
//! candidate) (and not raising its flag counts) improves every plan that
//! uses it and leaves every other plan alone, so the optimum can only improve: a better candidate never picks a
//! worse plan (tests: `better_candidate_never_scores_worse`).

/// Flag bits counted per region (index into the counts arrays).
pub const NFLAG: usize = 8;

#[derive(Clone, Debug)]
pub struct Move {
    pub cand: usize,
    /// Regions this move assigns to `cand` (sorted, disjoint across a plan).
    pub members: Vec<usize>,
    /// Region -> change of mean energy vs the input.
    pub delta: Vec<(usize, f64)>,
    /// (region, flag index) -> change of flagged-vertex count vs the input.
    pub flags: Vec<(usize, usize, i64)>,
    /// Weight change vs the input (sum of per-vertex L1), for ties.
    pub change: f64,
}

pub struct Problem {
    /// Input mean energy per region.
    pub orig: Vec<f64>,
    /// Vertex count per region.
    pub size: Vec<f64>,
    /// Highest mean energy a region may end with.
    pub cap: Vec<f64>,
    /// Input flagged counts per region and flag; the count at which that
    /// flag is a failing finding (`i64::MAX`: never); and the highest count
    /// allowed (`fail_at - 1` where the region does not fail yet, so no
    /// region gains a failing finding; unlimited where it already fails).
    pub counts: Vec<[i64; NFLAG]>,
    pub fail_at: Vec<[i64; NFLAG]>,
    pub limit: Vec<[i64; NFLAG]>,
    /// Regions moves are anchored at (the failing ones), in search order.
    pub anchors: Vec<usize>,
    pub moves: Vec<Move>,
    /// Compare plans on the gate first (default). False: score energy only.
    pub gate_first: bool,
    /// Cost per unit of weight change (`Move::change`), added to the score
    /// energy: among plans that score about the same, the smaller edit wins
    /// (closer to what the rigger made). A fixed linear term, so the pick
    /// stays monotone in the moves' energies.
    pub edit_cost: f64,
    /// A move must lower a failing region by this fraction (0.2: no
    /// polishing of what already works) or clear a failing finding. The
    /// refinement rounds of `fix` use 0 (any real gain).
    pub min_gain: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// Indices into `Problem::moves`.
    pub moves: Vec<usize>,
    /// Failing findings and score energy predicted from the moves'
    /// measurements (the plan minimizes fails first, then energy).
    pub fails: i64,
    pub energy: f64,
    pub change: f64,
    /// False only if the search hit its node budget (best found returned).
    pub exact: bool,
}

/// Score energy of per-region mean energies `e`.
pub fn energy_of(size: &[f64], e: &[f64]) -> f64 {
    let n: f64 = size.iter().sum::<f64>().max(1.0);
    let mean = size.iter().zip(e).map(|(s, x)| s * x).sum::<f64>() / n;
    let worst = e.iter().cloned().fold(0.0, f64::max);
    0.5 * mean + 0.5 * worst
}

impl Problem {
    /// 1 if counts `c` leave any failing finding (the gate fails), else 0.
    /// Plans compare on this first, then on score energy: a plan that
    /// passes the gate beats one that does not, whatever their scores.
    pub fn fails(&self, c: &[[i64; NFLAG]]) -> i64 {
        let n: i64 = c.iter().zip(&self.fail_at).map(|(c, at)| (0..NFLAG).filter(|&f| c[f] >= at[f]).count() as i64).sum();
        if self.gate_first { n.min(1) } else { 0 }
    }

    fn get(m: &Move, r: usize) -> f64 {
        m.delta.iter().find(|d| d.0 == r).map(|d| d.1).unwrap_or(0.0)
    }

    fn getc(m: &Move, r: usize, f: usize) -> i64 {
        m.flags.iter().find(|d| d.0 == r && d.1 == f).map(|d| d.2).unwrap_or(0)
    }

    /// True when the move is a real repair: it lowers a failing region by
    /// at least 20% or clears one of its failing findings (no polishing of
    /// what already works). The rules (caps, no new failing finding) are
    /// checked on whole plans, since a move can need its neighbor (one
    /// thigh's band breaks the hips seam until the other thigh moves too).
    /// Depends only on the move's own numbers, so a better candidate never
    /// loses a move.
    pub fn admissible(&self, m: &Move) -> bool {
        let floor = if self.min_gain > 0.0 { 0.005 } else { 1e-6 };
        let lowers = m.members.iter().any(|&r| self.anchors.contains(&r) && Self::get(m, r) < -(self.min_gain * self.orig[r]).max(floor));
        let clears =
            m.flags.iter().any(|&(r, f, d)| d < 0 && self.counts[r][f] >= self.fail_at[r][f] && self.counts[r][f] + d < self.fail_at[r][f]);
        lowers || clears
    }
}

/// (fails, energy, change), compared fails first, energy beyond 1e-9
/// (adding and removing moves leaves rounding of that order), then the
/// smaller edit.
fn better(a: (i64, f64, f64), b: (i64, f64, f64)) -> bool {
    if a.0 != b.0 {
        return a.0 < b.0;
    }
    if (a.1 - b.1).abs() > 1e-9 {
        return a.1 < b.1;
    }
    a.2 < b.2 - 1e-6
}

/// A plan's region energies and flagged counts. A region a move covers
/// takes that move's own measurement (it was measured with the region
/// changed); a region no move covers takes the input plus every move's
/// spill onto it (seams on different sides add up). Summing spills onto a
/// covered region would count one improvement twice: a neighbor's repair
/// cleaning the seam the region's own repair already cleans.
pub fn plan_regions(p: &Problem, chosen: &[usize]) -> (Vec<f64>, Vec<[i64; NFLAG]>) {
    let nr = p.orig.len();
    let mut e = p.orig.clone();
    let mut c = p.counts.clone();
    let covered: Vec<bool> = (0..nr).map(|r| chosen.iter().any(|&mi| p.moves[mi].members.contains(&r))).collect();
    for &mi in chosen {
        let m = &p.moves[mi];
        for &(r, d) in &m.delta {
            if !covered[r] || m.members.contains(&r) {
                e[r] += d;
            }
        }
        for &(r, f, d) in &m.flags {
            if !covered[r] || m.members.contains(&r) {
                c[r][f] += d;
            }
        }
    }
    (e, c)
}

/// A move split for the search: its own numbers on the regions it covers,
/// and its spill onto the others.
struct Split {
    mem: Vec<(usize, f64, [i64; NFLAG])>,
    spill_e: Vec<(usize, f64)>,
    spill_c: Vec<(usize, usize, i64)>,
}

fn split(m: &Move) -> Split {
    Split {
        mem: m.members.iter().map(|&r| (r, Problem::get(m, r), std::array::from_fn(|f| Problem::getc(m, r, f)))).collect(),
        spill_e: m.delta.iter().copied().filter(|d| !m.members.contains(&d.0)).collect(),
        spill_c: m.flags.iter().copied().filter(|d| !m.members.contains(&d.0)).collect(),
    }
}

struct Search<'a> {
    p: &'a Problem,
    split: Vec<Split>,
    /// Admissible moves grouped by anchor index (a move belongs to its
    /// first member in anchor order), Pareto-pruned.
    by_anchor: Vec<Vec<usize>>,
    tmp_e: Vec<f64>,
    tmp_c: Vec<[i64; NFLAG]>,
    tmp_me: Vec<f64>,
    tmp_mc: Vec<[i64; NFLAG]>,
    scratch_e: Vec<f64>,
    scratch_c: Vec<[i64; NFLAG]>,
    /// Region covered by a chosen move, and that move's own change there.
    used: Vec<bool>,
    mem_e: Vec<f64>,
    mem_c: Vec<[i64; NFLAG]>,
    /// Sum of every chosen move's spill per region (counts where uncovered).
    spill_e: Vec<f64>,
    spill_c: Vec<[i64; NFLAG]>,
    chosen: Vec<usize>,
    change: f64,
    /// The `keep` best plans found so far, best first.
    best: Vec<Plan>,
    keep: usize,
    nodes: usize,
    budget: usize,
}

fn dominates(a: &Move, b: &Move) -> bool {
    // `a` is at least as good as `b` everywhere and blocks no more regions.
    if !a.members.iter().all(|r| b.members.contains(r)) || a.change > b.change {
        return false;
    }
    let regions = a.delta.iter().map(|d| d.0).chain(b.delta.iter().map(|d| d.0));
    for r in regions {
        if Problem::get(a, r) > Problem::get(b, r) {
            return false;
        }
    }
    for &(r, f, _) in a.flags.iter().chain(b.flags.iter()) {
        if Problem::getc(a, r, f) > Problem::getc(b, r, f) {
            return false;
        }
    }
    true
}

impl Search<'_> {
    fn region_e(&self, r: usize) -> f64 {
        self.p.orig[r] + if self.used[r] { self.mem_e[r] } else { self.spill_e[r] }
    }

    fn region_c(&self, r: usize) -> [i64; NFLAG] {
        let d = if self.used[r] { &self.mem_c[r] } else { &self.spill_c[r] };
        std::array::from_fn(|f| self.p.counts[r][f] + d[f])
    }

    fn state(&self) -> Option<(i64, f64)> {
        let p = self.p;
        let nr = p.orig.len();
        let e: Vec<f64> = (0..nr).map(|r| self.region_e(r)).collect();
        if (0..nr).any(|r| e[r] > p.cap[r] + 1e-12) {
            return None;
        }
        let c: Vec<[i64; NFLAG]> = (0..nr).map(|r| self.region_c(r)).collect();
        if (0..nr).any(|r| (0..NFLAG).any(|f| c[r][f] > p.limit[r][f])) {
            return None;
        }
        Some((p.fails(&c), energy_of(&p.size, &e) + p.edit_cost * self.change))
    }

    fn leaf(&mut self) {
        if let Some((fails, energy)) = self.state() {
            let key = (fails, energy, self.change);
            if self.best.len() >= self.keep && !better(key, self.bar()) {
                return;
            }
            let mut moves = self.chosen.clone();
            moves.sort_unstable();
            if self.best.iter().any(|b| b.moves == moves) {
                return;
            }
            let at = self.best.iter().position(|b| better(key, (b.fails, b.energy, b.change))).unwrap_or(self.best.len());
            self.best.insert(at, Plan { moves, fails, energy, change: self.change, exact: true });
            self.best.truncate(self.keep);
        }
    }

    /// The plan a new one must beat to enter the kept list.
    fn bar(&self) -> (i64, f64, f64) {
        if self.best.len() < self.keep {
            return (i64::MAX, f64::MAX, f64::MAX);
        }
        let b = &self.best[self.best.len() - 1];
        (b.fails, b.energy, b.change)
    }

    /// False when nothing below this node can beat the incumbent. Per
    /// region, the lowest energy (and counts) still reachable: fixed where
    /// a chosen move covers it; else the better of staying uncovered (its
    /// spills plus the most negative spill of each remaining anchor's free
    /// moves) and being covered by some free move.
    fn bound_ok(&mut self, i: usize) -> bool {
        let p = self.p;
        let nr = p.orig.len();
        let mut opt_e = std::mem::take(&mut self.scratch_e);
        let mut opt_c = std::mem::take(&mut self.scratch_c);
        opt_e.clear();
        opt_e.extend((0..nr).map(|r| self.spill_e[r]));
        opt_c.clear();
        opt_c.extend((0..nr).map(|r| self.spill_c[r]));
        for r in 0..nr {
            self.tmp_me[r] = f64::INFINITY;
            self.tmp_mc[r] = [i64::MAX; NFLAG];
        }
        for j in i..self.by_anchor.len() {
            if self.used[p.anchors[j]] {
                continue;
            }
            for &mi in &self.by_anchor[j] {
                if !self.free(mi) {
                    continue;
                }
                let sp = &self.split[mi];
                for &(r, d, c) in &sp.mem {
                    self.tmp_me[r] = self.tmp_me[r].min(d);
                    for f in 0..NFLAG {
                        self.tmp_mc[r][f] = self.tmp_mc[r][f].min(c[f]);
                    }
                }
                for &(r, d) in &sp.spill_e {
                    if d < 0.0 {
                        self.tmp_e[r] = self.tmp_e[r].min(d);
                    }
                }
                for &(r, f, d) in &sp.spill_c {
                    if d < 0 {
                        self.tmp_c[r][f] = self.tmp_c[r][f].min(d);
                    }
                }
            }
            for r in 0..nr {
                opt_e[r] += self.tmp_e[r];
                self.tmp_e[r] = 0.0;
                for f in 0..NFLAG {
                    opt_c[r][f] += self.tmp_c[r][f];
                    self.tmp_c[r][f] = 0;
                }
            }
        }
        for r in 0..nr {
            if self.used[r] {
                opt_e[r] = p.orig[r] + self.mem_e[r];
                for f in 0..NFLAG {
                    opt_c[r][f] = p.counts[r][f] + self.mem_c[r][f];
                }
            } else {
                opt_e[r] = p.orig[r] + opt_e[r].min(self.tmp_me[r]);
                for f in 0..NFLAG {
                    opt_c[r][f] = p.counts[r][f] + opt_c[r][f].min(self.tmp_mc[r][f]);
                }
            }
        }
        let ok = (|| {
            if (0..nr).any(|r| opt_e[r] > p.cap[r] + 1e-12) {
                return false;
            }
            if (0..nr).any(|r| (0..NFLAG).any(|f| opt_c[r][f] > p.limit[r][f])) {
                return false;
            }
            let bar = self.bar();
            let fails = p.fails(&opt_c);
            if fails != bar.0 {
                return fails < bar.0;
            }
            energy_of(&p.size, &opt_e) + p.edit_cost * self.change < bar.1 + 1e-9
        })();
        self.scratch_e = opt_e;
        self.scratch_c = opt_c;
        ok
    }

    fn apply(&mut self, mi: usize, sign: i64) {
        let sp = &self.split[mi];
        for &(r, d) in &sp.spill_e {
            self.spill_e[r] += sign as f64 * d;
        }
        for &(r, f, d) in &sp.spill_c {
            self.spill_c[r][f] += sign * d;
        }
        for &(r, d, c) in &sp.mem {
            self.used[r] = sign > 0;
            self.mem_e[r] = d;
            self.mem_c[r] = c;
        }
        self.change += sign as f64 * self.p.moves[mi].change;
    }

    fn free(&self, mi: usize) -> bool {
        !self.p.moves[mi].members.iter().any(|&r| self.used[r])
    }

    /// A first incumbent: per anchor, the move that improves the current
    /// state most, then single-move swaps (add, drop, replace) until none
    /// helps. Only a starting point: the search below is exact.
    fn greedy(&mut self) {
        for i in 0..self.by_anchor.len() {
            self.try_best_at(i);
        }
        for _round in 0..20 {
            let before = self.state().map(|(f, e)| (f, e, self.change));
            // Drop a move.
            for k in (0..self.chosen.len()).rev() {
                let mi = self.chosen[k];
                let now = self.state().map(|(f, e)| (f, e, self.change)).unwrap_or((i64::MAX, f64::MAX, f64::MAX));
                self.apply(mi, -1);
                match self.state() {
                    Some((f, e)) if better((f, e, self.change), now) => {
                        self.chosen.remove(k);
                    }
                    _ => self.apply(mi, 1),
                }
            }
            // Replace or add at each anchor.
            for i in 0..self.by_anchor.len() {
                let a = self.p.anchors[i];
                let held: Vec<usize> = self.chosen.iter().copied().filter(|&mi| self.p.moves[mi].members.contains(&a)).collect();
                let now = self.state().map(|(f, e)| (f, e, self.change)).unwrap_or((i64::MAX, f64::MAX, f64::MAX));
                for &mi in &held {
                    self.apply(mi, -1);
                }
                self.chosen.retain(|m| !held.contains(m));
                if !self.try_best_at(i) || self.state().map(|(f, e)| !better((f, e, self.change), now)).unwrap_or(true) {
                    // Nothing better here: restore.
                    let added: Vec<usize> = self.chosen.iter().copied().filter(|&mi| self.p.moves[mi].members.contains(&a)).collect();
                    for &mi in &added {
                        self.apply(mi, -1);
                    }
                    self.chosen.retain(|m| !added.contains(m));
                    for &mi in &held {
                        self.apply(mi, 1);
                        self.chosen.push(mi);
                    }
                }
            }
            let after = self.state().map(|(f, e)| (f, e, self.change));
            match (before, after) {
                (Some(b), Some(a)) if better(a, b) => continue,
                _ => break,
            }
        }
        self.leaf();
        for mi in std::mem::take(&mut self.chosen) {
            self.apply(mi, -1);
        }
    }

    /// Applies the free move of anchor `i` that most improves the state;
    /// false if none does.
    fn try_best_at(&mut self, i: usize) -> bool {
        let mut pick: Option<(usize, (i64, f64, f64))> = None;
        for k in 0..self.by_anchor[i].len() {
            let mi = self.by_anchor[i][k];
            if !self.free(mi) {
                continue;
            }
            self.apply(mi, 1);
            if let Some((f, e)) = self.state() {
                let cur = pick.map(|x| x.1).unwrap_or((i64::MAX, f64::MAX, f64::MAX));
                if better((f, e, self.change), cur) {
                    pick = Some((mi, (f, e, self.change)));
                }
            }
            self.apply(mi, -1);
        }
        let now = self.state().map(|(f, e)| (f, e, self.change));
        if let (Some((mi, s)), Some(now)) = (pick, now) {
            if better(s, now) {
                self.apply(mi, 1);
                self.chosen.push(mi);
                return true;
            }
        }
        false
    }

    fn dfs(&mut self, i: usize) {
        self.nodes += 1;
        if self.nodes > self.budget {
            return;
        }
        if i == self.by_anchor.len() {
            self.leaf();
            return;
        }
        if !self.bound_ok(i) {
            return;
        }
        if !self.used[self.p.anchors[i]] {
            for k in 0..self.by_anchor[i].len() {
                let mi = self.by_anchor[i][k];
                if !self.free(mi) {
                    continue;
                }
                self.apply(mi, 1);
                self.chosen.push(mi);
                self.dfs(i + 1);
                self.chosen.pop();
                self.apply(mi, -1);
            }
        }
        self.dfs(i + 1);
    }
}

/// The best plan: passing the gate first, then lowest score energy, then
/// smallest change. The empty plan (the input) is always feasible, so this
/// never returns worse than the input as measured.
pub fn pick(p: &Problem) -> Plan {
    pick_top(p, 1).remove(0)
}

/// The `keep` best plans, best first (`fix` re-scores them on the mesh,
/// where moves measured apart can interact). Exact unless the search hits
/// its node budget (`Plan::exact`); the first is always `pick`'s plan.
pub fn pick_top(p: &Problem, keep: usize) -> Vec<Plan> {
    let nr = p.orig.len();
    let na = p.anchors.len();
    let mut by_anchor: Vec<Vec<usize>> = vec![Vec::new(); na];
    for (mi, m) in p.moves.iter().enumerate() {
        if !p.admissible(m) {
            continue;
        }
        if let Some(i) = p.anchors.iter().position(|a| m.members.contains(a)) {
            by_anchor[i].push(mi);
        }
    }
    let weighted = |m: &Move| m.delta.iter().map(|&(r, d)| p.size[r] * d).sum::<f64>();
    // Pareto pruning: drop a move another move of the same anchor beats
    // everywhere (ties: keep the lower index). Exactness is unaffected.
    for list in by_anchor.iter_mut() {
        let keep: Vec<usize> = list
            .iter()
            .copied()
            .filter(|&b| {
                !list.iter().any(|&a| a != b && dominates(&p.moves[a], &p.moves[b]) && (!dominates(&p.moves[b], &p.moves[a]) || a < b))
            })
            .collect();
        *list = keep;
        // Most promising first: better incumbents prune more.
        list.sort_by(|&a, &b| weighted(&p.moves[a]).partial_cmp(&weighted(&p.moves[b])).unwrap().then(a.cmp(&b)));
    }
    let input = Plan { moves: Vec::new(), fails: p.fails(&p.counts), energy: energy_of(&p.size, &p.orig), change: 0.0, exact: true };
    let mut s = Search {
        p,
        split: p.moves.iter().map(split).collect(),
        by_anchor,
        tmp_e: vec![0.0; nr],
        tmp_c: vec![[0; NFLAG]; nr],
        tmp_me: vec![0.0; nr],
        tmp_mc: vec![[0; NFLAG]; nr],
        scratch_e: Vec::new(),
        scratch_c: Vec::new(),
        used: vec![false; nr],
        mem_e: vec![0.0; nr],
        mem_c: vec![[0; NFLAG]; nr],
        spill_e: vec![0.0; nr],
        spill_c: vec![[0; NFLAG]; nr],
        chosen: Vec::new(),
        change: 0.0,
        best: Vec::new(),
        keep: keep.max(1),
        nodes: 0,
        budget: 2_000_000,
    };
    // The empty plan is the first incumbent when it keeps the rules (it
    // always does from the input; a refinement may start from a state that
    // breaks one, and then only plans that repair it count).
    if s.state().is_some() {
        s.best.push(input.clone());
    }
    s.greedy();
    let t0 = std::time::Instant::now();
    s.dfs(0);
    if std::env::var_os("WF_DEBUG").is_some() {
        let sizes: Vec<usize> = s.by_anchor.iter().map(Vec::len).collect();
        eprintln!("pick: {} moves, admissible per anchor {sizes:?}, {} nodes, {:.2}s", p.moves.len(), s.nodes, t0.elapsed().as_secs_f64());
    }
    let exact = s.nodes <= s.budget;
    let mut best = if s.best.is_empty() { vec![input] } else { s.best };
    for b in best.iter_mut() {
        b.exact = exact;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Rng;

    /// A random picking problem shaped like a real one: a chain of regions,
    /// some failing, moves of each candidate on a region alone and with its
    /// neighbors, touching the neighbors' energies and flag counts.
    fn random_problem(rng: &mut Rng, ncand: usize) -> Problem {
        let nr = 4 + (rng.next_u64() % 6) as usize;
        let orig: Vec<f64> = (0..nr).map(|_| 0.02 + 0.4 * rng.f64()).collect();
        let size: Vec<f64> = (0..nr).map(|_| 50.0 + 400.0 * rng.f64()).collect();
        let anchors: Vec<usize> = (0..nr).filter(|_| rng.f64() < 0.5).collect();
        let anchors = if anchors.is_empty() { vec![0] } else { anchors };
        let cap: Vec<f64> = orig.iter().map(|e| e * 1.1 + 0.01).collect();
        let counts: Vec<[i64; NFLAG]> =
            (0..nr).map(|r| if anchors.contains(&r) { [5, 0, 0, 0, 0, 0, 0, 0] } else { [1, 0, 0, 0, 0, 0, 0, 0] }).collect();
        let fail_at: Vec<[i64; NFLAG]> =
            (0..nr).map(|_| [3, i64::MAX, i64::MAX, i64::MAX, i64::MAX, i64::MAX, i64::MAX, i64::MAX]).collect();
        let limit: Vec<[i64; NFLAG]> = (0..nr).map(|r| if anchors.contains(&r) { [i64::MAX; NFLAG] } else { [2; NFLAG] }).collect();
        let mut moves = Vec::new();
        for c in 1..=ncand {
            for &a in &anchors {
                for set in [vec![a], (a.saturating_sub(1)..=(a + 1).min(nr - 1)).collect::<Vec<_>>()] {
                    let touched: Vec<usize> = (set[0].saturating_sub(1)..=(set[set.len() - 1] + 1).min(nr - 1)).collect();
                    let delta = touched
                        .iter()
                        .map(|&r| {
                            let d = if set.contains(&r) { -orig[r] * rng.f64() * 0.9 } else { orig[r] * (0.3 * rng.f64() - 0.2) };
                            (r, d)
                        })
                        .collect();
                    let flags = touched.iter().map(|&r| (r, 0usize, (rng.next_u64() % 3) as i64 - 1)).collect();
                    moves.push(Move { cand: c, members: set.clone(), delta, flags, change: 10.0 * rng.f64() });
                }
            }
        }
        Problem { orig, size, cap, counts, fail_at, limit, anchors, moves, gate_first: true, edit_cost: 1e-5, min_gain: 0.2 }
    }

    /// Brute force over every plan, for cross-checking the search.
    fn brute(p: &Problem) -> (i64, f64) {
        let nr = p.orig.len();
        let ok: Vec<usize> = (0..p.moves.len()).filter(|&i| p.admissible(&p.moves[i])).collect();
        let mut best = (p.fails(&p.counts), energy_of(&p.size, &p.orig));
        let mut stack: Vec<(usize, Vec<usize>)> = vec![(0, Vec::new())];
        while let Some((i, chosen)) = stack.pop() {
            if i == ok.len() {
                let (e, c) = plan_regions(p, &chosen);
                if (0..nr).all(|r| e[r] <= p.cap[r] + 1e-12 && (0..NFLAG).all(|f| c[r][f] <= p.limit[r][f])) {
                    let ch: f64 = chosen.iter().map(|&mi| p.moves[mi].change).sum();
                    let x = (p.fails(&c), energy_of(&p.size, &e) + p.edit_cost * ch);
                    if x.0 < best.0 || (x.0 == best.0 && x.1 < best.1) {
                        best = x;
                    }
                }
                continue;
            }
            stack.push((i + 1, chosen.clone()));
            let m = &p.moves[ok[i]];
            if !chosen.iter().any(|&o| p.moves[o].members.iter().any(|r| m.members.contains(r))) {
                let mut c2 = chosen;
                c2.push(ok[i]);
                stack.push((i + 1, c2));
            }
        }
        best
    }

    #[test]
    fn search_is_exact() {
        let mut rng = Rng::new(11);
        for _ in 0..300 {
            let p = random_problem(&mut rng, 2);
            if p.moves.len() > 22 {
                continue;
            }
            let got = pick(&p);
            assert!(got.exact);
            let b = brute(&p);
            assert_eq!(got.fails, b.0);
            assert!((got.energy - b.1).abs() < 1e-9, "search {} vs brute {}", got.energy, b.1);
        }
    }

    /// The property behind "a better candidate never yields a worse fix":
    /// make one candidate better (lower its moves' energy changes in some
    /// regions, never raise any; optionally fewer flags) and the picked
    /// plan never scores worse. Also: never worse than the input.
    #[test]
    fn better_candidate_never_scores_worse() {
        let mut rng = Rng::new(5);
        let mut checked = 0;
        for _ in 0..2000 {
            let p = random_problem(&mut rng, 3);
            let before = pick(&p);
            let input = (p.fails(&p.counts), energy_of(&p.size, &p.orig));
            assert!(before.fails < input.0 || (before.fails == input.0 && before.energy <= input.1 + 1e-9));
            let c = 1 + (rng.next_u64() % 3) as usize;
            let mut q = Problem {
                orig: p.orig.clone(),
                size: p.size.clone(),
                cap: p.cap.clone(),
                counts: p.counts.clone(),
                fail_at: p.fail_at.clone(),
                limit: p.limit.clone(),
                anchors: p.anchors.clone(),
                moves: p.moves.clone(),
                gate_first: p.gate_first,
                edit_cost: p.edit_cost,
                min_gain: p.min_gain,
            };
            for m in q.moves.iter_mut().filter(|m| m.cand == c) {
                for d in m.delta.iter_mut() {
                    if rng.f64() < 0.6 {
                        d.1 -= 0.2 * rng.f64() * p.orig[d.0];
                    }
                }
                for f in m.flags.iter_mut() {
                    if rng.f64() < 0.3 {
                        f.2 -= 1;
                    }
                }
            }
            let after = pick(&q);
            assert!(before.exact && after.exact);
            assert!(
                after.fails < before.fails || (after.fails == before.fails && after.energy <= before.energy + 1e-9),
                "better candidate {c}: ({}, {}) -> ({}, {})",
                before.fails,
                before.energy,
                after.fails,
                after.energy
            );
            checked += 1;
        }
        assert_eq!(checked, 2000);
    }
}
