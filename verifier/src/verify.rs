//! THE verifier — the whole verification side in one file (mirrors verify.py):
//! decode the prover's per-round replies, run the staged/fused protocol drawing
//! fresh coins, compile our OWN constraints, and run the 6 checks. The prover is
//! external (Python, behind the Prover trait) and untrusted; decoding is explicit,
//! dependency-free (serde_json::Value accessors only — no derive/macros), and
//! every decoder returns Option so a malformed reply is a clean REJECT, not a
//! panic. The 6 checks (bottom of the file) are bit-exact with verify.py and
//! rayon-parallel; they only trust constraints WE compiled (compile_claims), never
//! prover- or Python-supplied ones.

use std::collections::HashMap;
use serde_json::Value;
use rayon::prelude::*;

use crate::field::{add, mul, sub};
use crate::claim::ClaimSet;
use crate::compile::{CoefSrc, Constraints, Expander, Run};
use crate::handlers::compile_claims;
use crate::prover::Prover;
use crate::protocol::{challenge, lagrange, merkle_leaf, merkle_verify, poly_eval,
                      random_columns, Chal, Config, BLIND_IRS, BLIND_LIN, BLIND_QUAD,
                      EMPTY_COMMIT_ROOT, NUM_BLINDING_ROWS};

// ===========================================================================
// Wire decode (untrusted prover output → typed values). Explicit, Option-based.
// ===========================================================================
fn hex32(s: &str) -> Option<[u8; 32]> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    if s.len() != 64 { return None; }
    let mut b = [0u8; 32];
    for i in 0..32 {
        b[i] = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(b)
}
fn u64s(v: &Value) -> Option<Vec<u64>> {
    v.as_array()?.iter().map(|x| x.as_u64()).collect()
}
fn root(v: &Value, k: &str) -> Option<[u8; 32]> {
    hex32(v.get(k)?.as_str()?)
}
/// {"<j>": [col…], …} — a per-query-index column map (JSON keys are strings).
fn opened_map(v: &Value, k: &str) -> Option<HashMap<u64, Vec<u64>>> {
    v.get(k)?.as_object()?.iter()
        .map(|(j, col)| Some((j.parse().ok()?, u64s(col)?)))
        .collect()
}
/// {"<j>": [[hexsibling, side], …], …} — a per-query-index merkle-path map.
fn paths_map(v: &Value, k: &str) -> Option<HashMap<u64, Vec<([u8; 32], u8)>>> {
    v.get(k)?.as_object()?.iter().map(|(j, steps)| {
        let path = steps.as_array()?.iter().map(|step| {
            let a = step.as_array()?;
            Some((hex32(a.first()?.as_str()?)?, a.get(1)?.as_u64()? as u8))
        }).collect::<Option<Vec<_>>>()?;
        Some((j.parse().ok()?, path))
    }).collect()
}

pub struct Round3 {
    pub q_irs: Vec<u64>,
    pub q_lin: Vec<u64>,
    pub p_0:   Vec<u64>,
}
/// Round-4 openings, one entry per commitment block, in ROW-BLOCK ORDER (the
/// order the joint column concatenates and the compiled row_starts index into).
/// Two blocks today ([p1, p2]); the persistent W block prepends a third
/// ([w, p1, p2], analysis/persistent-weights.md). Vec-shaped so the checks are
/// N-generic; the two-block case is byte-identical.
pub struct Round4 {
    pub opened: Vec<HashMap<u64, Vec<u64>>>,
    pub paths:  Vec<HashMap<u64, Vec<([u8; 32], u8)>>>,
}
impl Round3 {
    fn parse(v: &Value) -> Option<Round3> {
        Some(Round3 { q_irs: u64s(v.get("q_irs")?)?,
                      q_lin: u64s(v.get("q_lin")?)?,
                      p_0:   u64s(v.get("p_0")?)? })
    }
}

/// The commitment-block suffixes in row-block order. Read from the transcript's
/// optional "blocks" field; absent → the legacy two blocks ["p1","p2"], so old
/// proofs parse identically.
pub fn block_suffixes(v: &Value) -> Vec<String> {
    match v.get("blocks").and_then(|b| b.as_array()) {
        Some(arr) => arr.iter().filter_map(|s| s.as_str().map(str::to_string)).collect(),
        None => vec!["p1".into(), "p2".into()],
    }
}

impl Round4 {
    fn parse(v: &Value) -> Option<Round4> {
        let sfx = block_suffixes(v);
        let opened = sfx.iter().map(|s| opened_map(v, &format!("opened_{s}")))
            .collect::<Option<Vec<_>>>()?;
        let paths = sfx.iter().map(|s| paths_map(v, &format!("paths_{s}")))
            .collect::<Option<Vec<_>>>()?;
        Some(Round4 { opened, paths })
    }
}

/// Decode a flat transcript object into the round values — shared by the fused
/// live path AND the offline verify_proof binary.
pub fn parse_transcript(v: &Value) -> Option<(Vec<[u8; 32]>, Round3, Round4)> {
    let roots = block_suffixes(v).iter()
        .map(|s| root(v, &format!("root_{s}"))).collect::<Option<Vec<_>>>()?;
    Some((roots, Round3::parse(v)?, Round4::parse(v)?))
}

// ===========================================================================
// Round adapters: cross the prover boundary, decode one round's reply.
// ===========================================================================
fn round1(p: &mut impl Prover) -> Option<[u8; 32]> {
    root(&p.request(1, None, None, None), "root_p1")
}
fn round2(p: &mut impl Prover, s_op: &[u8]) -> Option<[u8; 32]> {
    root(&p.request(2, Some(s_op), None, None), "root_p2")
}
fn round3(p: &mut impl Prover, s_op: &[u8], s_comb: &[u8]) -> Option<Round3> {
    Round3::parse(&p.request(3, Some(s_op), Some(s_comb), None))
}
fn round4(p: &mut impl Prover, s_op: &[u8], s_comb: &[u8], s_col: &[u8]) -> Option<Round4> {
    Round4::parse(&p.request(4, Some(s_op), Some(s_comb), Some(s_col)))
}
fn round0(p: &mut impl Prover, s_op: &[u8], s_comb: &[u8], s_col: &[u8])
          -> Option<(Vec<[u8; 32]>, Round3, Round4)> {
    parse_transcript(&p.request(0, Some(s_op), Some(s_comb), Some(s_col)))
}

// ===========================================================================
// Protocol entry points.
// ===========================================================================

/// INTERACTIVE protocol (SOUND) — port of verify.py::run_verification. Draw each
/// round's seed only AFTER the prover has committed that round. `rand` is the
/// verifier's coin source (MUST be a CSPRNG). False on any prover protocol error.
pub fn run_verification<Pr: Prover, R: FnMut() -> Vec<u8>>(
    p: &mut Pr, cs: &mut ClaimSet, rand: R) -> bool {
    staged(p, cs, rand).unwrap_or_else(|| {
        eprintln!("verify: malformed prover response → REJECT"); false
    })
}
fn staged<Pr: Prover, R: FnMut() -> Vec<u8>>(
    p: &mut Pr, cs: &mut ClaimSet, mut rand: R) -> Option<bool> {
    let root_p1 = round1(p)?;
    let s_op    = rand();                 // only now — after R_p1
    let root_p2 = round2(p, &s_op)?;
    let s_comb  = rand();                 // only now — after R_p2
    let r3      = round3(p, &s_op, &s_comb)?;
    let s_col   = rand();                 // only now — after the polys are fixed
    let r4      = round4(p, &s_op, &s_comb, &s_col)?;
    Some(verify(cs, &[root_p1, root_p2], &r3, r4, &s_op, &s_comb, &s_col).0)
    // NOTE: the staged interactive path commits R_p1, R_p2 one root per round.
    // The persistent W block (a third root, committed in the phase-1 round)
    // extends this path when the interactive transport lands; the offline
    // verify_proof path (parse_transcript) is already N-generic.
}

/// FUSED variant (~4×, ONE pass) — port of verify.py::run_verification_fast. All
/// three seeds up front. NOT sound on its own (prover sees every challenge first).
pub fn run_verification_fast<Pr: Prover, R: FnMut() -> Vec<u8>>(
    p: &mut Pr, cs: &mut ClaimSet, mut rand: R) -> bool {
    let (s_op, s_comb, s_col) = (rand(), rand(), rand());
    match round0(p, &s_op, &s_comb, &s_col) {
        Some((roots, r3, r4)) => verify(cs, &roots, &r3, r4, &s_op, &s_comb, &s_col).0,
        None => { eprintln!("verify: malformed prover response → REJECT"); false }
    }
}

/// Compile our own constraints from the claims + s_op, then run the 6 checks on
/// the round values. Returns (overall, per-check) so the diff-test can localize.
pub fn verify(cs: &mut ClaimSet,
              roots: &[[u8; 32]], r3: &Round3, r4: Round4,
              s_op: &[u8], s_comb: &[u8], s_col: &[u8]) -> (bool, Vec<(&'static str, bool)>) {
    let cfg: Config = cs.cfg;
    let cons = compile_claims(cs, s_op);
    let q = random_columns(s_col, &cfg);
    let cols = match opened_columns(r4, &q) {
        Some(c) => c,
        None => return (false, vec![("opened_columns", false)]),
    };

    // Progress markers (stderr) — observability only, not part of any check.
    let t0 = std::time::Instant::now();
    let mark = |name: &str| eprintln!("[verify] {name} @ {:.1} min", t0.elapsed().as_secs_f64() / 60.0);
    let mut r = Vec::new();
    mark("merkle");    r.push(("merkle",    merkle_test(&cols, roots)));
    // Only merkle needs the raw per-commit subcolumns. Join them into one set once
    // (freeing the raw form as it goes), then irs/lin/quad share this single cj —
    // built once instead of three times, and never held alongside the raw columns.
    let cj = cols.into_joint();
    mark("irs_col");   r.push(("irs_col",   irs_column_test(&cj, &r3.q_irs, &q, s_comb, &cfg)));
    mark("lin_sum");   r.push(("lin_sum",   linear_constraint_test(&r3.q_lin, &cons, s_comb, &cfg)));
    mark("lin_col");   r.push(("lin_col",   linear_column_test(&cj, &r3.q_lin, &q, &cons, s_comb, &cfg)));
    mark("quad_zero"); r.push(("quad_zero", quadratic_constraint_test(&r3.p_0, &cfg)));
    mark("quad_col");  r.push(("quad_col",  quadratic_column_test(&cj, &r3.p_0, &q, &cons, s_comb, &cfg)));
    mark("done");
    let ok = r.iter().all(|(_, b)| *b);
    (ok, r)
}

// ===========================================================================
// The 6 checks (bit-exact with verify.py, rayon-parallel). Constraints come from
// OUR compile (compile_claims), so the verifier never trusts prover-supplied ones.
// ===========================================================================

/// Gate 2 of analysis/docs/linear-fold-unification.md (bit-exact corpus
/// comparisons): with VERIFY_DUMP_CHECK_VALUES set, the linear checks print
/// their computed values to stderr so two verifier builds can be diffed on a
/// stored proof. Observability only — never part of the verdict.
fn dump_check_values() -> bool {
    std::env::var_os("VERIFY_DUMP_CHECK_VALUES").is_some()
}

/// One opened column at query index j: the joint column (commit 0 rows ‖ commit 1
/// rows) plus the per-commit merkle path, in query order Q.
pub struct OpenedColumns {
    pub subcols: Vec<Vec<Vec<u64>>>,          // subcols[commit][query_position]
    pub paths: Vec<Vec<Vec<([u8; 32], u8)>>>,
}
impl OpenedColumns {
    /// Consume the per-commit subcolumns into the T joint columns (commit 0 ‖
    /// commit 1), moving each source column in so it frees as we go — the raw and
    /// joined forms never both fully exist, halving peak column memory. Paths drop
    /// here too: merkle is the only check that needs them, and it runs first.
    fn into_joint(self) -> Vec<Vec<u64>> {
        let OpenedColumns { subcols, paths } = self;
        drop(paths);
        let t = subcols[0].len();
        let mut cj: Vec<Vec<u64>> = (0..t).map(|_| Vec::new()).collect();
        for commit in subcols {                       // consume each commitment's columns…
            for (qi, col) in commit.into_iter().enumerate() {
                cj[qi].extend(col);                   // …moving them into the joint column
            }
        }
        cj
    }
}

/// Build OpenedColumns from the decoded round 4, in query order. None if the
/// prover omitted any queried column → REJECT.
fn opened_columns(r4: Round4, q: &[u64]) -> Option<OpenedColumns> {
    // MOVE the queried columns out of r4 (remove, not clone) — the parsed proof's
    // copy frees as we extract, so each column lives once (here → cj), not twice.
    let Round4 { mut opened, mut paths } = r4;
    let subcols = opened.iter_mut()
        .map(|m| q.iter().map(|j| m.remove(j)).collect::<Option<Vec<_>>>())
        .collect::<Option<Vec<_>>>()?;
    let paths = paths.iter_mut()
        .map(|m| q.iter().map(|j| m.remove(j)).collect::<Option<Vec<_>>>())
        .collect::<Option<Vec<_>>>()?;
    Some(OpenedColumns { subcols, paths })
}

/// L_c(η_qi) for every message column c∈[0,ncols) and every query qi — computed
/// ONCE (parallel over queries) and reused. Replaces the naive per-(row,col,query)
/// recomputation (a pow + a Fermat inverse each) with ncols·T. Bit-identical
/// values; the only change is memoization.
fn lagrange_table(cfg: &Config, etas: &[u64], ncols: usize) -> Vec<u64> {
    let mut lag = vec![0u64; etas.len() * ncols];
    lag.par_chunks_mut(ncols).zip(etas.par_iter()).for_each(|(row, &eta)| {
        for c in 0..ncols { row[c] = lagrange(cfg, c as u64, eta); }
    });
    lag
}

// 1. Merkle: every opened sub-column hashes to its commit's root.
fn merkle_test(cols: &OpenedColumns, roots: &[[u8; 32]]) -> bool {
    for (ci, rt) in roots.iter().enumerate() {
        if *rt == EMPTY_COMMIT_ROOT {
            // The all-zeros root is the prover's sentinel for a ZERO-ROW block
            // (no tree exists to verify against — e.g. an empty p2 on a tape
            // with no phase-2 aux). It is only acceptable when the block's
            // opened sub-columns are actually empty: a NON-empty block
            // presenting the sentinel would otherwise skip merkle binding
            // entirely, leaving its opened values unbound by any commitment
            // (P6 audit finding S2).
            if cols.subcols[ci].iter().all(|c| c.is_empty()) { continue; }
            return false;
        }
        for qi in 0..cols.subcols[ci].len() {
            if !merkle_verify(merkle_leaf(&cols.subcols[ci][qi]), &cols.paths[ci][qi], *rt) {
                return false;
            }
        }
    }
    true
}

// 2. IRS column identity: q_irs(η_j) = Σ_i r_irs[i]·col[NB+i] + col[BLIND_IRS].
fn irs_column_test(cj: &[Vec<u64>], irs_poly: &[u64],
                   q: &[u64], s_comb: &[u8], cfg: &Config) -> bool {
    let t = q.len();
    let m_witness = cj[0].len() - NUM_BLINDING_ROWS;
    // Parallel over rows i: each row combiner r_i computed ONCE (not per column).
    let partial = (0..m_witness).into_par_iter().fold(
        || vec![0u64; t],
        |mut acc, i| {
            let ri = challenge(s_comb, i as u64, "irs");
            for qi in 0..t { acc[qi] = add(acc[qi], mul(ri, cj[qi][NUM_BLINDING_ROWS + i])); }
            acc
        },
    ).reduce(|| vec![0u64; t], |mut a, b| { for qi in 0..t { a[qi] = add(a[qi], b[qi]); } a });
    (0..t).all(|qi| add(partial[qi], cj[qi][BLIND_IRS]) == poly_eval(irs_poly, cfg.eta(q[qi])))
}

// 3. Linear sum identity: Σ_c q_lin(ζ_c) = Σ_g r_lin[g]·rhs_g.
fn linear_constraint_test(lin_poly: &[u64], cons: &Constraints,
                          s_comb: &[u8], cfg: &Config) -> bool {
    let mut sum_q = 0u64;
    for c in 0..cfg.ell { sum_q = add(sum_q, poly_eval(lin_poly, cfg.zeta(c))); }
    // Parallel over runs: each is an independent Chal::sum (ascending, same
    // order as before); field add is associative/commutative, so the reduce
    // is bit-identical.
    let chal = Chal::new(s_comb);
    let rhs = cons.rhs.par_iter()
        .map(|&(g0, length, b_g)| mul(chal.sum(g0, g0 + length), b_g))
        .reduce(|| 0u64, add);
    if dump_check_values() {
        eprintln!("[check-values] lin_sum lhs={sum_q} rhs={rhs}");
    }
    sum_q == rhs
}

// 4. Linear column identity: q_lin(η_j) = Σ_i r_i(η_j)·col[i] + col[BLIND_LIN].
// The O(W·T) check. Parallel over rows; r^T A is never materialized. Per row,
// accumulate (col → Σ challenge·coef) into a DENSE scratch (O(terms), vs an
// O(terms²) linear-search dedup on dense LogUp-table rows), then evaluate via the
// memoized Lagrange table. Message columns are always < ELL.
/// prefix[qi*(ncols+1) + k] = Σ_{c<k} lag[qi*ncols + c]. Lets a constant-coef
/// Lagrange sum over a contiguous slot run [lo, hi) be one subtraction,
/// prefix[hi] − prefix[lo]. Built once per verify (mirrors quad_col's `prefix`).
fn prefix_lagrange(lag: &[u64], ncols: usize, t: usize) -> Vec<u64> {
    let w = ncols + 1;
    let mut pre = vec![0u64; t * w];
    for qi in 0..t {
        for c in 0..ncols { pre[qi * w + c + 1] = add(pre[qi * w + c], lag[qi * ncols + c]); }
    }
    pre
}

/// One row's contribution to lin_col, as ONE generic fold over `for_runs`
/// (Phase 1.3 of linear-fold-unification.md — replaces the four hand-written
/// fast paths, which survive as the test oracle `row_contrib_fastpaths`).
/// Per run: `Repeat` hashes its one cid once (const coef → prefix-Lagrange
/// difference; vector coef → coef·Lagrange dot); `OneToOne` pays one challenge
/// per slot, served from the family PRELOAD when present (the Freivalds
/// strided-repeat kinds — k distinct cids recurring every row); `Fan` takes one
/// challenge range-sum shared across queries. Bit-identical to the per-term
/// emit fold: field ops are exact, only the grouping differs (difftested
/// against both oracles). `pre` is the prefix-sum-of-Lagrange table.
fn row_contrib(exp: &Expander, flat_lo: usize, n_slots: usize, chal: &Chal,
               pre_chal: Option<(usize, &[u64])>, lag: &[u64], pre: &[u64],
               ncols: usize, t: usize, out: &mut [u64]) {
    let ch = |cid: usize| match pre_chal {
        Some((b, tab)) => tab[cid - b],
        None => chal.at(cid),
    };
    exp.for_runs(flat_lo, n_slots, &mut |run| match run {
        Run::Repeat { slot_lo, len, cid, coef: CoefSrc::Const(v) } => {
            let sc = mul(ch(cid), v);
            let w = ncols + 1;
            for qi in 0..t {
                let psum = sub(pre[qi * w + slot_lo + len], pre[qi * w + slot_lo]);
                out[qi] = add(out[qi], mul(sc, psum));
            }
        }
        Run::Repeat { slot_lo, len, cid, coef } => {
            let c = ch(cid);
            for qi in 0..t {
                let mut acc = 0u64;
                for s in 0..len {
                    acc = add(acc, mul(coef.at(s), lag[qi * ncols + slot_lo + s]));
                }
                out[qi] = add(out[qi], mul(c, acc));
            }
        }
        Run::OneToOne { slot_lo, len, cid_lo, cid_step, coef } => {
            for s in 0..len {
                let sc = mul(ch(cid_lo + s * cid_step), coef.at(s));
                for qi in 0..t {
                    out[qi] = add(out[qi], mul(sc, lag[qi * ncols + slot_lo + s]));
                }
            }
        }
        Run::Fan { slot, cid_lo, len, coef } => {
            let sc = mul(chal.sum(cid_lo, cid_lo + len), coef);
            for qi in 0..t {
                out[qi] = add(out[qi], mul(sc, lag[qi * ncols + slot]));
            }
        }
    });
}

/// Family-scoped challenge preload for the strided-repeat kinds: their cids
/// (base + f mod k) recur on every row with no contiguous runs, so one small
/// dense buffer (span = k ≤ H·K, or h for FreivaldsC) replaces per-slot
/// hashing across the whole family. Capped so a pathological span (attention
/// AV at extreme context: k = h·S) falls back to on-the-fly hashing.
const MAX_PRELOAD: usize = 1 << 20;
fn preload_for(exp: &Expander, chal: &Chal) -> Option<(usize, Vec<u64>)> {
    let (base, span) = match exp {
        Expander::FreivaldsB { base, k, .. } => (*base, *k),
        Expander::FreivaldsC { base, h, .. } => (*base, *h),
        _ => return None,
    };
    (span <= MAX_PRELOAD).then(|| (base, chal.preload(base, base + span)))
}

/// lin_col: `Σ over constraint families of Σ over their rows of (row's
/// coef·Lagrange) · cj[row]`, checked against `lin_poly(η_qi)` at every opened
/// column `qi`. Serial over families, parallel over the rows within each (rayon
/// splits the range — the big families fill the cores, the small ones are cheap).
/// Bounded memory: one spanning Expander per family, each row reconstructed on the
/// fly via `row_contrib` (window `[ro·ell, ·)`), nothing per-row stored. Field add
/// is associative/commutative, so the row partition doesn't change the sum.
fn linear_column_test(cj: &[Vec<u64>], lin_poly: &[u64], q: &[u64],
                      cons: &Constraints, s_comb: &[u8], cfg: &Config) -> bool {
    let t = q.len();
    let etas: Vec<u64> = q.iter().map(|&j| cfg.eta(j)).collect();
    let ncols = cfg.ell as usize;
    let lag = lagrange_table(cfg, &etas, ncols);
    let pre = prefix_lagrange(&lag, ncols, t);

    let chal = Chal::new(s_comb);
    let mut acc = vec![0u64; t];
    let total: usize = cons.families.iter().map(|f| f.nrows()).sum::<usize>().max(1);
    let (mut done, mut next) = (0usize, 5usize);   // lin_col progress (stderr, observability only)
    for fam in &cons.families {
        let (ell, row_start, length) = (fam.ell, fam.row_start, fam.length);
        let pre_tab = preload_for(&fam.exp, &chal);   // band-scoped, dropped per family
        let part = (0..fam.nrows()).into_par_iter()
            .fold(|| (vec![0u64; t], vec![0u64; t]),
                |(mut a, mut reta), ro| {
                    let (flat_lo, n_slots) = (ro * ell, ell.min(length - ro * ell));
                    for x in reta.iter_mut() { *x = 0; }
                    row_contrib(&fam.exp, flat_lo, n_slots, &chal,
                                pre_tab.as_ref().map(|(b, v)| (*b, v.as_slice())),
                                &lag, &pre, ncols, t, &mut reta);
                    let gi = row_start + ro;
                    for qi in 0..t { if reta[qi] != 0 { a[qi] = add(a[qi], mul(reta[qi], cj[qi][gi])); } }
                    (a, reta)
                })
            .map(|(a, _)| a)
            .reduce(|| vec![0u64; t], |mut a, b| { for qi in 0..t { a[qi] = add(a[qi], b[qi]); } a });
        for qi in 0..t { acc[qi] = add(acc[qi], part[qi]); }
        done += fam.nrows();
        let pct = done * 100 / total;
        if pct >= next { eprintln!("[verify]   lin_col ~{pct}% ({done}/{total} rows)"); next = pct + 5; }
    }
    let dump = dump_check_values();
    (0..t).all(|qi| {
        let lhs = add(acc[qi], cj[qi][BLIND_LIN]);
        let rhs = poly_eval(lin_poly, etas[qi]);
        if dump { eprintln!("[check-values] lin_col q{qi} lhs={lhs} rhs={rhs}"); }
        lhs == rhs
    })
}

// 5. Quadratic zero identity: p_0(ζ_c) = 0 ∀ c.
fn quadratic_constraint_test(quad_poly: &[u64], cfg: &Config) -> bool {
    (0..cfg.ell).all(|c| poly_eval(quad_poly, cfg.zeta(c)) == 0)
}

// 6. Quadratic column identity:
//    p_0(η_j) = Σ_t r_quad[t]·(Ux·Uy + a·mask·Uz − b·mask)|η_j + col[BLIND_QUAD],
// mask = Σ_{c<n} L_c(η). Prefix-sum the Lagrange table so each mask is O(1).
fn quadratic_column_test(cj: &[Vec<u64>], quad_poly: &[u64],
                         q: &[u64], cons: &Constraints, s_comb: &[u8],
                         cfg: &Config) -> bool {
    let t = q.len();
    let etas: Vec<u64> = q.iter().map(|&j| cfg.eta(j)).collect();
    let max_n = cons.quadratic.iter().map(|qf| qf.ell.min(qf.length)).max().unwrap_or(0);
    let ncols = (cfg.ell as usize).max(max_n).max(1);
    let lag = lagrange_table(cfg, &etas, ncols);
    let mut prefix = vec![0u64; t * (ncols + 1)];   // prefix[qi][n] = Σ_{c<n} L_c(η_qi)
    for qi in 0..t {
        for n in 0..ncols {
            prefix[qi * (ncols + 1) + n + 1] = add(prefix[qi * (ncols + 1) + n], lag[qi * ncols + n]);
        }
    }
    // Quad lift: iterate FAMILIES (serial) × their rows (rayon) — the lin_col
    // pattern. Row tt of a family is quad index_base+tt (positional r_quad
    // pairing, unchanged from the per-row Vec) on rows x_row+tt / y_row+tt /
    // z_row+tt with n_at(tt) constrained slots. Same terms, same sum.
    let nq_total: usize = cons.quadratic.iter().map(|qf| qf.nrows()).sum();
    let done = std::sync::atomic::AtomicUsize::new(0);   // observability tick only
    let step = (nq_total / 100).max(1);
    let mut partial = vec![0u64; t];
    for qf in &cons.quadratic {
        let part = (0..qf.nrows()).into_par_iter().fold(
            || vec![0u64; t],
            |mut acc, tt| {
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n % step == 0 { eprintln!("[verify]   quad_col {}/{} quads", n, nq_total); }
                let s = challenge(s_comb, (qf.index_base + tt) as u64, "quad");
                let n_slots = qf.n_at(tt);
                for qi in 0..t {
                    let col = &cj[qi];
                    let mask = prefix[qi * (ncols + 1) + n_slots];
                    let (ux, uy, uz) =
                        (col[qf.x_row + tt], col[qf.y_row + tt], col[qf.z_row + tt]);
                    let term = sub(add(mul(ux, uy), mul(mul(qf.a, mask), uz)), mul(qf.b, mask));
                    acc[qi] = add(acc[qi], mul(s, term));
                }
                acc
            },
        ).reduce(|| vec![0u64; t], |mut a, b| { for qi in 0..t { a[qi] = add(a[qi], b[qi]); } a });
        for qi in 0..t { partial[qi] = add(partial[qi], part[qi]); }
    }
    let dump = dump_check_values();
    (0..t).all(|qi| {
        let lhs = add(partial[qi], cj[qi][BLIND_QUAD]);
        let rhs = poly_eval(quad_poly, etas[qi]);
        if dump { eprintln!("[check-values] quad_col q{qi} lhs={lhs} rhs={rhs}"); }
        lhs == rhs
    })
}

// Per-family eval difftests live in verify_difftest.rs (test-only; NOT part of the
// verdict-path TCB). They gate the lin_contrib shortcuts (== the emit oracle) and
// the whole-variable span equivalence. Run with `cargo test`.
#[cfg(test)]
#[path = "verify_difftest.rs"]
mod lin_contrib_difftest;
