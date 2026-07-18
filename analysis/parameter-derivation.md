# VerInf / Ligero: protocol and parameter derivation

This document has two parts. Part I describes the protocol: the problem statement,
the arithmetization, the cryptographic primitives, and the proof scheme. Part II
is the analysis: the derivation of the cheating probability, the cost model, and
each parameter derived from its governing constraints — as general formulas.

Numeric substitution for the Llama-4-Maverick run, all worked figures, the
optimization plots, and the toy-model validation live in the companion notebook
`verification-parameter-analysis.ipynb`. This document deliberately contains no
arithmetic: only the formulas and the reasoning that selects each parameter.

Symbols are introduced at first use and collected in a table at the end.

---

# Part I. Protocol

## 1. Problem statement

The prover has run a model's forward pass (Llama-4-Maverick) on a token input and
obtained an output. A proof is required that the computation was performed
honestly, under the following conditions:

- the model weights are not revealed;
- the input tokens and intermediate activations are not revealed;
- the result is a single artifact that any party can check against the public
  commitments, without access to the prover's secrets.

Note that verification here is **not** cheaper than the forward pass — the verifier
opens and checks a large proof and its cost is measured in minutes to hours, far
above the seconds of a single forward pass. The value of the protocol is the two
privacy properties above plus public checkability of a fixed artifact, not a
saving in compute. (Succinct verification, where checking is asymptotically cheaper
than the computation, is a property of other proof systems, not of this Ligero
configuration.)

The protocol is Ligero: a zero-knowledge argument built on Reed–Solomon codes and
Merkle commitments.

## 2. Arithmetization

### 2.1. Finite field

All protocol arithmetic is carried out in the finite field $F = \mathbb{Z}_p$ —
the integers modulo a prime $p$. The Goldilocks field is used:

$$p = 2^{64} - 2^{32} + 1, \qquad |F| = p \approx 2^{64}.$$

Two properties motivate the choice:

1. $p < 2^{64}$: every field element fits in a 64-bit machine word, and the special
   form $2^{64} - 2^{32} + 1$ admits a reduction using only a few shifts and adds
   (no general division). This is the reason Goldilocks is chosen; it is one of
   several 64-bit primes with the next property, not the only one.
2. $p - 1 = 2^{32}\,(2^{32} - 1)$: the field contains roots of unity of order $n$
   for every $n \mid 2^{32}$ — the existence condition for an NTT of length $n$
   (see §4.2). Its high 2-adic valuation ($2^{32}$) is what makes long
   power-of-two NTTs possible.

§12.1 shows why the field must be 64-bit at all (the overflow bound rules out
31-bit fields); among 64-bit primes with a large 2-adic valuation, Goldilocks is
selected for its cheap reduction.

Field arithmetic is exact (no rounding error) and bounded (every result is again a
residue modulo $p$). This is what makes it possible to state the computation as a
system of exact equations.

### 2.2. Fixed-point representation

The field has no real numbers. A real value $x$ is represented as a fixed-point
integer:

$$\hat x = \mathrm{round}(s \cdot x),$$

with scale $s$. The fixed-point parameters are analyzed in §12.

### 2.3. Witness

The witness is a flat list of every field element arising in the computation:
weights, the activations of each layer, and the protocol's auxiliary quantities.
Its length is denoted $W$. The witness is secret and is not sent to the verifier.

### 2.4. Constraint system

Each elementary operation of the computation is encoded as an equation over
witness positions:

- a multiplication $z = x \cdot y$ is a **quadratic constraint** (it contains a
  product of two witness variables); their number is denoted $Q$;
- an addition $c = a + b$ is a **linear constraint** of the form
  $\sum_i a_i x_i = 0$ (a weighted sum of variables, no products); their number is
  denoted $L$.

The prover's claim after arithmetization is: "there exists a witness of length $W$
satisfying all $Q + L$ constraints, whose output positions contain the claimed
result." The constraint system encodes the computation completely: any witness
satisfying all constraints corresponds to an honest run.

### 2.5. Matrix multiplications: Freivalds reduction

Directly encoding a matrix product of shapes $(m \times k)\cdot(k \times n)$ would
require $m\,n\,k$ quadratic constraints. Instead the Freivalds check is used: for
a random vector $r$ chosen after the matrices $A, B, C$ are fixed, the equality

$$C r = A\,(B r)$$

is checked. If $C \ne AB$, the equality fails for a random $r$ with probability at
least $1 - 2/|F|$. The check is expressed as linear constraints on the committed
witness. Its contribution to the cheating probability is $2/|F|$ per matrix
multiplication.

### 2.6. Nonlinear functions: LogUp

The functions $\exp$ (softmax), SiLU, and rsqrt (RMSNorm) are not expressible
through addition and multiplication in a finite field. For each, a table of
(input, output) pairs with a fixed step is published. The prover proves the
statement "every pair used is present in the table" with the LogUp mechanism.

The LogUp contribution to the cheating probability is

$$\varepsilon_{\mathrm{LogUp}} \approx \frac{M}{|F|},$$

where $M$ is the total number of table accesses over the run. This quantity is
independent of the other protocol parameters and sets the lower bound on the
cheating probability (see §9.4).

## 3. Cryptographic primitives

### 3.1. Hash function

BLAKE3 is used. Notation for the cost model: $h$ is the time of one compression
operation; one compression processes 8 field elements.

### 3.2. Merkle tree and commitment

A Merkle tree over a set of $n$ values: the values are hashed, the resulting
hashes are hashed pairwise at the next level, and so on up to a single root hash.
Publishing the root is called a commitment and has two properties:

- **binding**: any change to a value after the root is published is detectable,
  since it changes the root;
- **selective opening**: value number $i$ is proved by presenting the value itself
  together with the $\log_2 n$ hashes on its path to the root.

### 3.3. The "commit → randomness" ordering, and Fiat–Shamir

Every verifier random challenge (the folding coefficients of §4.1, the Freivalds
vectors of §2.5, the opened-column indices of §5) is issued strictly after the
data it applies to has been committed. The probability analysis of Part II is
valid only under this ordering: randomness known to the prover before the data is
fixed would let the prover fit the data to the check.

**This protocol is non-interactive (Fiat–Shamir).** The challenges are not sent by
a live verifier; each is derived by a public hash (BLAKE3) from a seed that binds
the prior commitments — the implementation derives every challenge as
$\texttt{challenge}(\text{seed}, i, \text{label})$ and no challenge value is ever
transmitted. The consequence for soundness (§9): a cheating prover can *grind* —
re-derive the challenges under many nonces and keep an attempt that happens to
pass. A per-instance cheating probability $\varepsilon$ therefore protects only to
$\approx -\log_2\varepsilon$ bits *against the prover's hashing budget*. A target
that is safe interactively (e.g. $\varepsilon = 2^{-16.6}$) is grindable in seconds
non-interactively; the deployed soundness must exceed the grinding budget, which
makes the floor of §9.4 — not the choice of $T$ — the binding limit.

## 4. Encoding the witness

### 4.1. Random folding of constraints

Checking $Q + L$ constraints individually is unaffordable. Instead they are batched
into one check by a random linear combination. For each constraint a residual is
defined (left side minus right side; for an honest witness all residuals are zero).

**The construction used here is a random *vector*, not powers of a single field
element.** The verifier draws an independent random coefficient $\gamma_i$ per
constraint — in the implementation $\gamma_i = \texttt{challenge}(\text{seed}, i,
\text{label})$, a fresh pseudorandom field element for each $i$ — and considers

$$S = \sum_i \gamma_i \cdot (\text{residual}_i).$$

If all constraints hold, $S = 0$. If at least one residual is nonzero, then over
the uniform choice of the $\gamma_i$,

$$\Pr\big[S = 0 \;\big|\; \text{some residual} \ne 0\big] \;=\; \frac{1}{|F|},$$

because fixing all but one coefficient leaves a single uniform $\gamma_i$
multiplying a nonzero value. The batching costs $1/|F|$ per fold, **independent of
the number of constraints** $Q+L$.

**Contrast with the powers-of-one-element variant.** A weaker but common pedagogical
form uses $S(r) = \sum_i r^i\,(\text{residual}_i)$ with a single random $r$; this is
a nonzero polynomial of degree $\le Q+L$, so Schwartz–Zippel gives error
$(Q+L)/|F|$. With $Q+L \approx 3.7\times10^{11}$ that is $\approx 2^{-25.6}$ — which
would *dominate* every other floor term. The independent-vector construction avoids
this: its error is $1/|F|$ per fold, and the fold contribution to §9.4 is the small
term shown there, not $(Q+L)/|F|$. The distinction is load-bearing for the floor.

### 4.2. Reed–Solomon code

Two properties of polynomials over a field are used:

- **(A)** through any $K$ points passes exactly one polynomial of degree $\le K-1$;
- **(B)** two distinct polynomials of degree $< K$ agree in at most $K-1$ points
  (their difference is a nonzero polynomial of degree $< K$, with $< K$ roots).

Encoding a row of $K$ elements: by (A) the row is interpreted as the values of a
polynomial $f$ of degree $\le K-1$ at $K$ fixed points; the values of $f$ at
$N > K$ points are computed. The result is a codeword of length $N$. Code
parameters:

$$\rho = \frac{N}{K} \;\;\text{(inverse rate / blowup)}, \qquad
d = N - K + 1 = (\rho - 1)K + 1 \;\;\text{(code distance)}.$$

By (B) two distinct codewords differ in at least $d$ positions. Consequence:
changing even one of the $K$ source elements changes at least $d$ of the $N$
codeword positions. An error cannot be local — this is what makes selective
checking (§5) effective.

Computationally, encoding is a pair of NTTs (Number-Theoretic Transform — a fast
Fourier transform over a finite field, cost $O(n \log n)$): an inverse NTT of
length $K$ maps values to coefficients, a forward NTT of length $N$ maps
coefficients to $N$ values. An NTT of length $n$ exists when the field has a root
of unity of order $n$; for Goldilocks these are all powers of two up to $2^{32}$.

**Constraint 1.**

$$K,\ N \ \text{are powers of two}, \qquad N \le 2^{32}.$$

In particular, $\rho$ is a power of two.

### 4.3. Splitting the witness into a matrix

The witness is split into rows. Row structure:

$$K = \mathrm{ELL} + \mathrm{pad},$$

where $\mathrm{ELL}$ is the number of witness elements in the row and
$\mathrm{pad}$ is the number of random masking elements (purpose — §6). The number
of rows:

$$m = \frac{W}{\mathrm{ELL}}.$$

Each row is encoded per §4.2 into a row of length $N = \rho K$. The result is an
$m \times N$ matrix.

The key quantity for the cost model:

$$\lambda = \frac{K}{\mathrm{ELL}}$$

— the ratio of the full row length to its payload. Every protocol operation (NTT,
hashing, folding) runs over the whole row, so $\lambda$ is a cost multiplier per
witness element (§10).

## 5. Proof scheme

**Step 1. Commit.** The $m \times N$ matrix is committed by a Merkle tree over
columns. Column $j$ holds the value of every row's polynomial at point $j$; one
column query yields one probe from each row.

**Step 2. Test polynomials.** After the commit the verifier issues random
coefficients and the prover returns three folds (per §4.1):

| test | claim | construction |
|---|---|---|
| IRS, $q_{\mathrm{irs}} = \sum_i r_i f_i$ | every matrix row is a codeword (a polynomial of degree $< K$) | random sum of the row polynomials $f_i$ |
| linear, $q_{\mathrm{lin}} = \sum_i r_i(x) f_i(x)$ | all $L$ linear constraints hold | sum of (row polynomial) × (coefficient polynomial) products |
| quadratic, $p_0 = \sum_t r_t (p_x p_y + p_a p_z - p_b)$ | all $Q$ quadratic constraints hold | contains products of pairs of row polynomials |

**Step 3. Opening.** The verifier names $T$ random column indices. The prover
opens them with Merkle paths. For each column the verifier checks: consistency
with the root, and consistency of the test polynomials' values at that point with
the rows' values at that point.

The catching logic: an honest test polynomial is uniquely determined by the
committed rows. A forged test polynomial differs from the honest one and, by
property (B) of §4.2, agrees with the committed rows only in a small fraction of
domain points; opening a point outside that fraction exposes the forgery. The
quantitative analysis is §8.

## 6. Zero-knowledge: masking elements

An opened column contains one codeword value from each row. Without masking these
values are linear combinations of witness elements and leak information about it.

Masking is provided by the $\mathrm{pad}$ random elements in each row (§4.3).
Encoding is linear, so every codeword value is a weighted sum of all $K$ slots,
including the random ones. Masking is perfect in the following sense: for any set
of at most $\mathrm{pad}$ opened columns and any payload values, there exist
values of the masking elements that produce exactly the observed opened values.
Hence the opened values carry no information about the payload. (Solvability of
the corresponding linear system is guaranteed by the full rank of the Vandermonde
submatrix relating the masking slots to the opened points.)

The perfection condition is that the total number of openings per commitment does
not exceed $\mathrm{pad}$. Each proof opens $T$ columns; if a commitment is reused
across $Q_c$ proofs, the openings accumulate.

**Constraint 2 (zero-knowledge).**

$$\mathrm{pad} \;\ge\; T \cdot Q_c.$$

The right side is an absolute number of columns, not a fraction of the row. $Q_c$
differs across witness blocks: the weight block is committed once and serves many
proofs ($Q_c$ large); activation blocks are recomputed and recommitted in every
proof ($Q_c = 1$).

---

# Part II. Analysis

## 7. Test-polynomial degrees

The probabilistic analysis of §8 is governed by the degrees of the test
polynomials.

- **IRS.** A sum of row polynomials. A row polynomial passes through $K$ points —
  degree $\le K-1$ (property (A)). Degree of the sum:

$$\deg q_{\mathrm{irs}} \le K - 1.$$

- **Linear.** A product of a row polynomial (degree $\le K-1$) and a
  constraint-coefficient polynomial. Constraints touch only the $\mathrm{ELL}$
  payload slots (masking elements do not appear in the equations), so the
  coefficient polynomial has degree $\le \mathrm{ELL}-1$. The product degree is
  the sum:

$$\deg q_{\mathrm{lin}} \le K + \mathrm{ELL} - 2.$$

- **Quadratic.** Contains a product of two row polynomials, each of degree
  $\le K-1$:

$$\deg p_0 \le 2K - 2 \quad \text{— the largest of the three.}$$

Selective checking (§5, step 3) works only if there is a gap between the
polynomial degree and the domain size $N$: a polynomial of degree $D-1$ that passes
through all $N$ points leaves a forgery nowhere to diverge. For the quadratic
polynomial ($D-1 = 2K-2$) the bare gap requirement is

**Constraint 3.**

$$2K - 2 < N \qquad\Longleftrightarrow\qquad \rho > 2 - \tfrac{2}{K}.$$

In code this is `assert 2 * K_DEG <= N_LIG` (`prover/core.py`), i.e. $2K \le N$,
which for $N = \rho K$ is $\rho \ge 2$.

Note the subtlety: $\rho = 2$ (i.e. $N = 2K$) **passes** this bare inequality — the
gap is exactly two points, $N - (2K-2) = 2$. So $\rho = 2$ is *not* excluded by the
degree gap; it merely makes the gap vanishingly small. The per-column soundness it
yields is analyzed in §9.1, and it is what rules $\rho = 2$ out in practice, not
Constraint 3.

## 8. Cheating probability per opening

Let the honest test polynomial $g$ and a forgery $g' \ne g$ have degree $< D$. By
property (B) they agree in at most $D-1$ points of $N$. The probability that a
randomly opened column lands on an agreement point (forgery undetected):

$$\Pr[\text{miss}] \le \frac{D-1}{N}.$$

Substituting the degrees from §7:

$$\Pr[\text{miss}]_{\mathrm{lin}} \le \frac{K + \mathrm{ELL} - 2}{N}, \qquad
\Pr[\text{miss}]_{\mathrm{quad}} \le \frac{2K - 2}{N} \;\xrightarrow[\text{large }K]{}\; \frac{2}{\rho}.$$

The convenient form $2/\rho$ drops the $-2$; keep the exact $(2K-2)/N$ when the gap
is small (it is what distinguishes $\rho = 2$ from "never", see §9.1).

For the IRS test the claim is weaker ("a row is close to some codeword"), and the
Ligero analysis gives a per-opening catch probability $e/N$ under

$$e < \frac{d}{3},$$

where $d$ is the code distance. The condition $e < d/3$ ensures a unique nearest
codeword for a corrupted row; without uniqueness the linear and quadratic tests
would be checking constraints against an undefined witness. At the maximal
$e \approx (\rho-1)K/3$:

$$\frac{e}{N} = \frac{(\rho-1)K/3}{\rho K} = \frac{\rho-1}{3\rho}, \qquad
\Pr[\text{miss}]_{\mathrm{IRS}} \le 1 - \frac{\rho-1}{3\rho}.$$

The $T$ openings are independent; the miss probabilities are raised to the power
$T$:

$$\varepsilon_{\mathrm{IRS}} \le \left(1 - \frac{\rho-1}{3\rho}\right)^{T}, \qquad
\varepsilon_{\mathrm{lin}} \le \left(\frac{K + \mathrm{ELL}}{N}\right)^{T}, \qquad
\varepsilon_{\mathrm{quad}} \le \left(\frac{2}{\rho}\right)^{T}.$$

**Note on the formula used elsewhere.** A commonly quoted form is
$\varepsilon_{\mathrm{IRS}} = (1 - 1/\rho)^T$. It coincides with the correct
expression above only at $\rho = 4$, where
$\tfrac{\rho-1}{3\rho} = \tfrac14 = \tfrac1\rho$ — a numerical coincidence, not an
identity. For $\rho > 4$ the quoted form understates the soundness benefit of a
larger $\rho$; the corrected formula should be used whenever $\rho$ is revised.

## 9. Deriving $\rho$ and $T$

### 9.1. $\rho$: lower bound

At $\rho = 2$ ($N = 2K$) the quadratic test's *exact* miss probability is

$$\Pr[\text{miss}]_{\mathrm{quad}} = \frac{2K - 2}{2K} = 1 - \frac{1}{K},$$

not $1$: the test still catches, but only with probability $1/K$ per column. The
per-column soundness is $-\log_2(1 - 1/K) \approx 1/(K\ln 2)$ bits — for
$K = 2^{14}$ that is about $8.8\times10^{-5}$ bits, so reaching even 16.6 bits would
need on the order of $2\times10^{5}$ opened columns. That is impractical, not
impossible: $\rho = 2$ is ruled out by cost, not by an impossibility.

The dominant term also flips at $\rho = 2$: $\varepsilon_{\mathrm{quad}}$ (miss
$\approx 1$) exceeds $\varepsilon_{\mathrm{IRS}}$ (miss $\tfrac{5}{6}$), so the
quadratic test, not IRS, sets $T$ there. For $\rho \ge 4$ the IRS term dominates
and the quadratic miss ($\le \tfrac12$) is comfortably below it. By Constraint 1
$\rho$ is a power of two, and $\rho = 4$ is the smallest value with a usable
per-column soundness:

$$\boxed{\rho = 4.}$$

The value is forced by cost, not by the bare degree gap (§7).

### 9.2. $\rho = 8$: analysis of the alternative

$\rho = 8$ raises the per-column soundness (its correct value from §8, not the
understated form) and reduces $T$ and the proof proportionally. The price:
$N = 8K$ doubles the forward NTT and the hashing of each row. Since the bottleneck
is the prover, not the proof size, $\rho = 4$ is chosen. This choice is a
trade-off and is subject to revision if the cost balance changes.

### 9.3. Soundness at $\rho = 4$

$$\varepsilon_{\mathrm{IRS}} = \left(\tfrac{3}{4}\right)^T \ \text{(dominant)}, \qquad
\varepsilon_{\mathrm{lin}} \le \left(\tfrac{3}{8}\right)^T, \qquad
\varepsilon_{\mathrm{quad}} = \left(\tfrac{1}{2}\right)^T.$$

The cost of one opened column is $-\log_2(3/4)$ bits. For a target soundness of
$s$ bits ($\varepsilon \le 2^{-s}$):

$$\boxed{T \ge \frac{s}{\log_2(4/3)}.}$$

### 9.4. Lower bound on the cheating probability (floor)

The total cheating probability is the sum of all mechanisms' contributions:

$$\varepsilon \;\le\;
\underbrace{\varepsilon_{\mathrm{IRS}} + \varepsilon_{\mathrm{lin}} + \varepsilon_{\mathrm{quad}}}_{\text{decrease with } T}
\;+\; \underbrace{\frac{M}{|F|}}_{\text{LogUp}}
\;+\; \underbrace{\frac{N+3}{|F|}}_{\text{folds}}
\;+\; \underbrace{n_{\mathrm{mm}} \cdot \frac{2}{|F|}}_{\text{Freivalds}},$$

where the last three terms are independent of $T$ and form a floor on
$\varepsilon$. The fold term is $(N+3)/|F|$ — small — *because* the fold uses the
independent-vector construction of §4.1 ($1/|F|$ per fold, three folds, plus the
$N$ opened evaluations). Had the fold used powers of a single $r$, this term would
be $(Q+L)/|F| \approx 2^{-25.6}$ and would dominate everything; the construction is
what keeps the LogUp term $M/|F|$ the largest, hence the binding floor.

Because the floor is a lower bound on $\varepsilon$, the achievable soundness is
strictly below $-\log_2\varepsilon_{\text{floor}}$ for every finite $T$: total
$\varepsilon = \varepsilon_{\mathrm{IRS}}(T) + \varepsilon_{\text{floor}}$. Define
$T_{\max}$ as the point where the falling term meets the floor,
$\varepsilon_{\mathrm{IRS}}(T_{\max}) = \varepsilon_{\text{floor}}$:

$$T_{\max} = \frac{-\log_2 \varepsilon_{\text{floor}}}{\log_2(4/3)}.$$

At $T_{\max}$ the two equal terms sum, so the *delivered* soundness is
$-\log_2\varepsilon_{\text{floor}} - 1$ bits — one bit below the floor level, and
the asymptotic ceiling $-\log_2\varepsilon_{\text{floor}}$ is only approached as
$T \to \infty$. So $T_{\max}$ is where marginal columns stop helping, **not** a
soundness target; naming a soundness level is a required separate input, and any
level within about a bit of the floor needs impractically large $T$.

Ways to lift the floor itself: the LogUp term — by repeating its random challenge
(the term is squared, $(M/|F|)^2$); the field terms — only by enlarging the field
(a degree-2 Goldilocks extension gives $|F^2| \approx 2^{128}$). Under Fiat–Shamir
(§3.3) these — not $T$ — are the levers that matter, because the floor caps the
grinding-resistant soundness.

## 10. Cost model

### 10.1. Prover cost

Notation: $c(n)$ is the NTT time per element at length $n$ (a property of the GPU
kernel); $h$ is the BLAKE3 compression time (8 elements per operation).

Cost of the operations over one row:

| operation | composition | row cost |
|---|---|---|
| encoding | inverse NTT of length $K$ + forward NTT of length $N$ | $K\,c(K) + N\,c(N)$ |
| re-encoding (opening round) | the same | $K\,c(K) + N\,c(N)$ |
| linear fold | 2 transforms of length $2K$ | $4K\,c(2K)$ |
| hashing | $N$ values, 8 per compression | $(N/8)\,h$ |

A row's payload is $\mathrm{ELL}$ elements; the cost per witness element is the row
cost $/\ \mathrm{ELL} = \lambda \times$ cost per slot:

$$A_c = \lambda\big(c(K) + \rho\,c(\rho K)\big), \qquad
A_f = 4\lambda\,c(2K), \qquad
D_h = \frac{\lambda\,\rho\,h}{8}.$$

$$\boxed{\text{every cost constant} \;\propto\; \lambda.}$$

Total prover time:

$$T_{\text{prove}} \;\approx\;
\underbrace{4\,T_{\text{wit}}}_{\text{independent of }\lambda, K}
+ \lambda\Big[2\big(c(K) + \rho\,c(\rho K)\big) + 4c(2K) + \tfrac{\rho h}{8}\Big] W
+ \lambda \cdot 2\kappa\,c(2K)\,Q + B\,L + E\,W,$$

where $4\,T_{\text{wit}}$ is the four witness recomputations and $\kappa$ is the
number of transforms per quadratic row. The parameter $T$ does not enter the
prover cost: soundness is cheap for the prover and expensive for the verifier and
the proof size.

The per-slot cost constants are the product (geometry $\lambda$) × (machine); the
geometric factor is therefore optimizable independently of the hardware. Two
caveats on this identity as a predictor: (i) $\kappa$ is a *calibrated* constant
(fitted to the measured quadratic-fold cost), not counted from the code, so the
$Q$ term is a fit, not a first-principles bound; (ii) the identity is a *floor* —
the measured prover time exceeds it by roughly a factor of two (fold memory
traffic and orchestration outside the leading terms), so any wall-clock or
dollar figure derived from the identity alone is optimistic by about that factor.

### 10.2. Proof size and verifier cost

The proof consists of $T$ opened columns of height $m = W/\mathrm{ELL}$ (8 bytes
per element) and the test polynomials (length proportional to their degree, i.e.
to $K$):

$$P = \underbrace{8\,T\,\frac{W}{\mathrm{ELL}}}_{\text{opened columns}}
+ \underbrace{8\,(4K + \mathrm{ELL})}_{\text{test polynomials}} \ \text{bytes}.$$

The verifier's work is proportional to $T \cdot m$.

## 11. Deriving $\mathrm{pad}$ and $K$

### 11.1. $\mathrm{pad}$

From Constraint 2 and the definition of $\lambda$:

$$\lambda = \frac{K}{K - T\,Q_c} \qquad\Longrightarrow\qquad
\text{prover, proof, and verifier pay the multiplier } \frac{1}{1 - T Q_c / K}.$$

$Q_c$ is set by the recommit policy:

- activation blocks: $Q_c = 1$ by construction (recommit every proof), sufficient
  $\mathrm{pad} = T$;
- the weight block: $Q_c$ is set by the trade-off between $\lambda$ and the
  amortized recommit cost. One recommit is a single encoding pass over the
  weight-block slots:

$$C_{\text{recommit}} = \frac{W_{\text{weights}}}{\mathrm{ELL}}
\cdot \big(K\,c(K) + N\,c(N)\big).$$

$\mathrm{pad}$ must be set per block:

$$\boxed{\mathrm{ELL} = K - T\,Q_c, \qquad Q_c = 1 \ \text{for per-block-committed blocks}.}$$

A single global $\mathrm{pad}$ imposes on the activation blocks the $\lambda$
computed for the long-lived weight commitment.

### 11.2. $K$

The only hard constraint on $K$ is the ceiling from Constraint 1
($N \le 2^{32}$, i.e. $K \le 2^{32}/\rho$). $K$ enters the proof size (§10.2)
through two terms with opposite dependence (at $\mathrm{ELL} \approx K$):

$$P(K) \approx \frac{8\,T\,W}{K} + 32\,K.$$

The first term (opened columns) decreases with $K$: fewer rows, shorter columns.
The second (test polynomials) grows linearly. The minimum:

$$\frac{dP}{dK} = -\frac{8TW}{K^2} + 32 = 0
\qquad\Longrightarrow\qquad
\boxed{K^\ast = \tfrac{1}{2}\sqrt{T\,W}}, \qquad
\boxed{P_{\min} = 32\sqrt{T\,W}\ \text{bytes}.}$$

At the minimum the two terms are equal. This is the standard $\sqrt{W}$ scaling of
Ligero.

The cost of increasing $K$ for the prover is set by the shape of $c(n)$ — a
property of the NTT kernel, not of the protocol. The upper bound on $K$ is set by
the padding loss: the witness is laid into the matrix variable by variable, and a
variable shorter than $\mathrm{ELL}$ leaves an unused row remainder. The loss:

$$\text{loss} \approx \frac{n_{\text{vars}} \cdot \mathrm{ELL}}{2\,W},$$

where $n_{\text{vars}}$ is the number of witness variables. This loss is a *separate*
constraint on $K$: it is **not** folded into the $P(K)$, $T_{\text{prove}}$, or
verifier expressions above, which all assume rows are fully packed
($\mathrm{ELL}$ payload per row). Treating the padding-loss ceiling and the
fully-packed cost model together (e.g. inflating $W$ by $1/(1-\text{loss})$) is left
to whoever fixes the operating $K$; here it only bounds $K$ from above, it does not
enter the cost of a given $K$.

## 12. Fixed-point parameters

### 12.1. Scale $s$

A dot product of length $k$ sums $k$ products of magnitude up to $(sR)^2$, where
$R$ is the maximum absolute value of the activations. The no-wraparound condition:

$$\boxed{k\,s^2 R^2 < \frac{p}{2} = 2^{63}.}$$

A second, independent criterion is model accuracy as a function of $s$.

**Consequence — the field size is forced (this is the derivation behind §2.1).**
Taking $\log_2$ of the overflow condition, the modulus must be wider than

$$\boxed{b \;=\; \log_2 k \;+\; 2\log_2 s \;+\; 2\log_2 R \;+\; 1\ (\text{sign}).}$$

For the model's largest dot length and activation range this exceeds the width of
any 31-bit field (M31, BabyBear) — and still does even after dropping $s$ to the
smallest value accuracy tolerates — so no 31-bit field admits the required scales.
This forces a 64-bit field. Among 64-bit primes with a large 2-adic valuation
(several exist), Goldilocks is selected for its cheap reduction (§2.1); the
64-bit *width* is forced, the specific prime is a performance choice. (Raising the
soundness target past the point where the fold floor binds is the one reason to
move to a degree-2 extension $|F^2|\approx 2^{128}$; see §9.4.)

### 12.2. Bit-width $w$

A range-check confines values to $\pm 2^{w-1}$; the representable real range is
$R = 2^{w-1}/s$, whence:

$$\boxed{w = 1 + \log_2 s + \log_2 R_{\max}.}$$

An activation exceeding $R_{\max}$ causes the proof to be rejected (REJECT); $w$ is
chosen from the maximum activation of the specific model.

### 12.3. LogUp table size

A table on $[-x_{\max}, x_{\max}]$ has step $2 x_{\max} / T_{\text{tbl}}$. The
condition "the table step is no coarser than the fixed-point step $1/s$":

$$\boxed{T_{\text{tbl}} \ge 2\,s\,x_{\max}.}$$

The table's second parameter — the clip bound $x_{\max}$ — affects accuracy
independently of $T_{\text{tbl}}$.

---

## 13. Parameter summary (analysis)

Each parameter, its governing constraint, and the resulting decision — formulas
only; the numeric substitution for the demonstrated run is in the companion
notebook.

| parameter | governing constraint | decision |
|---|---|---|
| field $p$ | overflow $k\,s^2R^2 < p/2$ (§12.1) | must be 64-bit (31-bit fields fail the overflow width); Goldilocks $2^{64}-2^{32}+1$ chosen among 64-bit high-2-adic primes for cheap reduction |
| $\rho$ | power of two; usable per-column soundness (§9.1) | $\rho = 4$ — $\rho = 2$ is legal but needs ~$2\times10^5$ columns; $\rho = 8$ is a prover-unfavorable trade-off (§9.2) |
| $T$ | $T \ge s/\log_2(4/3)$ (§9.3), useful only up to $T_{\max}$ (§9.4) | $T$ is **not** an optimum — it is fixed only once a target soundness is named. The construction caps grinding-resistant soundness ~1 bit below the floor (§9.4); under Fiat–Shamir (§3.3) the levers past that are lookup-challenge repetition and field extension, not larger $T$ |
| $\mathrm{pad}$ | ZK lifetime $\mathrm{pad} \ge T\,Q_c$ (§6, §11.1) | per block: activations $\mathrm{pad} = T$ ($Q_c = 1$); weights by the recommit trade-off. A single global $\mathrm{pad}$ over-pays $\lambda$ on the activation blocks |
| $K$ | $\sqrt{W}$ optimum $K^\ast = \tfrac12\sqrt{TW}$ (§11.2), ceilings from $c(n)$ and padding loss | raise toward the optimum; the practical ceiling is set by the padding loss and by the NTT kernel's $c(n)$ shape, not by the protocol |
| $s$ | overflow (§12.1) and accuracy — two independent criteria | fixed by both; increasing it past the accuracy plateau only reduces overflow headroom |

**Where the prover time goes.** $T_{\text{prove}}$ is dominated by the
geometry-independent term $4\,T_{\text{wit}}$ (§10.1), so tuning the geometry
($K$, $\mathrm{pad}$, $\rho$) shrinks the proof size and the verifier work but not
the prover time. Reducing prover time is a separate, orthogonal optimization
(streaming the claims) that complements the geometry choice.

**Open items.**

- $n_{\text{vars}}$ (§11.2) — the padding-loss coefficient; computing it exactly
  requires a witness-tape sweep and fixes the precise $K$ ceiling.
- Relaxing the Ligero proximity condition $e < d/3$ to $e < d/2$ in the
  interactive setting would raise the per-column soundness to
  $-\log_2(1/2)\cdot\frac{\rho-1}{2\rho}$ and lower $T$; this requires revisiting
  the proximity lemma.
- The constant $\kappa$ (transforms per quadratic row, §10.1) is calibrated to the
  measured quadratic-fold cost rather than counted from `core.py`.

## Appendix: notation

| symbol | definition | introduced in |
|---|---|---|
| $F$, $p$, $\lvert F\rvert$ | Goldilocks field, its modulus and size, $p = 2^{64} - 2^{32} + 1$ | §2.1 |
| $s$ | fixed-point scale | §2.2 |
| $W$ | witness length | §2.3 |
| $Q$, $L$ | number of quadratic and linear constraints | §2.4 |
| $M$ | number of LogUp table accesses | §2.6 |
| $n_{\mathrm{mm}}$ | number of matrix multiplications | §2.5 |
| $h$ | BLAKE3 compression time | §3.1 |
| $K$, $\mathrm{ELL}$, $\mathrm{pad}$ | row length; payload; masking elements; $K = \mathrm{ELL} + \mathrm{pad}$ | §4.3 |
| $\rho$, $N$, $d$ | inverse rate; codeword length $N = \rho K$; distance $d = N - K + 1$ | §4.2 |
| $m$ | number of rows, $W/\mathrm{ELL}$ | §4.3 |
| $\lambda$ | $K/\mathrm{ELL}$, the geometric cost multiplier | §4.3 |
| $T$ | number of opened columns (`T_QUERIES`) | §5 |
| $Q_c$ | number of proofs per commitment | §6 |
| $\varepsilon$, $s$ (bits) | cheating probability; soundness in bits, $\varepsilon = 2^{-s}$ | §8, §9.3 |
| $c(n)$ | NTT time per element at length $n$ | §4.2 |
| $\kappa$ | transforms per quadratic row | §10.1 |
| $n_{\mathrm{vars}}$ | number of witness variables (padding loss) | §11.2 |
