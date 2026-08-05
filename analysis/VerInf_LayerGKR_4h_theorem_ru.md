# Layer-GKR-LF: теоретический протокол доказательства 400B-инференса за четыре часа

Дата: 2026-08-04  
Целевая модель: Llama-4 Maverick, 48 слоёв, 24 MoE-слоя, 128 experts,
top-1, `S=1000` (500 скрытых prompt tokens и 500 скрытых continuation tokens).  
Машина модели времени: один DGX Spark/GB10, 121 GB unified memory.  
Цель: online prover не более 14 400 секунд после независимого от запроса
enrollment постоянной модели.

## 1. Теорема

Пусть выполнены следующие условия.

1. Утверждённый GGUF, его exact bytes-to-field mapping, claim-DAG, таблицы,
   scales, tie-break rules и 48 layer-weight roots закреплены доверенным manifest.
2. Используется точная integer-семантика VerInf: raw accumulator, затем
   deterministic rescale/range; существующие SiLU, softmax, RMSNorm, RoPE,
   routing и one-sided UI relations не меняются.
3. Primitive-cost model равен Appendix A.5:
   `A_c=4.2`, `A_f=3.4`, `A_x=4.2`, `D=0.5`, `C=15`, `B=0.6` ns; один exact
   semantic forward стоит не более 3609 секунд; streamed factorized coefficient
   generation, включая quantized decode и secret-padding PRG, имеет
   `E<=3 ns/weight-or-padding-slot`.
4. Для переноса primitive model на новый prover принимается явная performance
   hypothesis `kappa<=1.5`. Она консервативнее multiplier
   `10/7.784=1.285`, полученного из верхнего края самой paper-модели 8--10 h,
   но не объявляется статистической tail-гарантией.
5. Proof transport имеет не менее 1 Gbit/s, RTT не более 20 ms, binary dump
   не медленнее 108 MB/s; оба consumer сохраняют эти rates одновременно с GPU
   L5 opening encode. Формат chunk-major/framed, queue bounded. Post-hoc
   materialized JSON не входит в canonical protocol.
6. Manifest-certified maximum resident compressed layer shard, включая
   quantization scales/zero-points, alignment и loader staging, не превышает
   11.5 GB; common row-chunk/NTT/allocator workspace не превышает 83.89 GB.
   Shard удерживается от semantic compute до L5 opening.

Тогда описанный ниже публично проверяемый, zero-knowledge протокол
`Layer-GKR-LF`:

- доказывает тот же exact inference/UI statement;
- имеет ошибку soundness не больше intended security profile `T=40`
  (это сравнение параметров безопасности, а не использование headline run как
  performance baseline);
- использует менее 121 GB peak memory;
- для canonical streamed binary proof имеет online prover bound

\[
T_{prove}\le 14\,200\text{ s}=3.95\text{ h}.
\]

- допускает отдельную оценку streaming GPU verification при дополнительных
  throughput assumptions; эта оценка не входит в основную prover theorem.

Это даёт ускорение `8/3.95>2.02` даже относительно нижней границы paper range
8--10 h.

Условие на модель времени можно записать без выбора `kappa`:

\[
T_{prove}<4\text{ h}\quad\Longleftarrow\quad \kappa\le1.53.
\]

## 2. Почему это не Ligero над уменьшенным trace

Текущий VerInf коммитит плоский trace размера

\[
W\simeq8.88\cdot10^{11}
\]

и выполняет commit encode, linear fold и opening re-encode над этим trace.
`Layer-GKR-LF` вообще не RS-кодирует внутренние matmul, attention, SiLU,
Hadamard или RMS intermediates. Их корректность доказывает layer-local
tensor-GKR.

RS/Ligero остаётся только как linear-functional commitment к четырём видам
границ:

1. постоянные веса;
2. input/output hidden state слоя;
3. lookup boundary tuples, которые обязаны быть связаны до LogUp challenge;
4. скрытые route-sort records и маленький ZK mask tape.

Таким образом, Ligero является только input PCS. Основной arithmetic IOP,
MoE reduction, composition и zero knowledge устроены иначе.

## 3. Enrollment и statement

### 3.1 Manifest

Для каждого слоя `l` manifest фиксирует:

- `layer_id`, тип слоя и shapes;
- field/scaling/range metadata;
- ordered ids корней весов;
- output-major RS row layout: одинаковые input/padding positions всех output
  coordinates выровнены для codeword projection;
- canonical input/output row layout и padding;
- digests public lookup tables;
- predecessor/successor interface ids.

Super-root

\[
R_{model}=H(\text{GGUF digest},\text{DAG digest},R_{W,0},\ldots,R_{W,47},
\text{metadata})
\]

является trusted verifier input. Claims, roots и transcript digests никогда не
берутся из недоверенного proof как policy.

Каждый `R_{W,l}` -- hiding RS/Merkle commitment с
`ELL=8192,K=16384,N=65536`, fresh secret enrollment padding и отдельным
opening ledger. Encoded matrix не хранится.

Если embedding и LM head физически tied, enrollment создаёт две логические
output-major RS views: `(V,d)` и её `(d,V)` transpose. Cold permutation-link
доказывает, что это один и тот же GGUF tensor. Online model считает обе
projection orientations; одна layout не может поддержать обе same-column
checks.

### 3.2 Lifetime

Layer proof открывает `q=54` columns. Поэтому один weight root используется не
более

\[
\left\lfloor\frac{8192}{54}\right\rfloor=151
\]

proofs до refresh. Амортизированные recommit+link costs включены в модель как
43 секунды на proof. Public constant padding seeds запрещены.

## 4. Layer-local protocol

Superlayer 0 включает hidden-token validity, canonical transcript-hash circuit,
embedding и transformer block 0. Superlayer 47 включает block 47, final
RMSNorm, LM head, output-token selection и one-sided surprisal/UI epilogue.
Соответствующие embedding/head/final-gain weights входят в roots 0 и 47.
Начальный token root и внешний transcript digest закреплены verifier, а hash
circuit доказывает их равенство. Hidden embedding реализуется как matmul
committed one-hot token vector на embedding matrix и использует тот же
project-before-sumcheck seam; LM head является обычным последним projected
matmul. Поэтому ни hidden vocabulary index, ни output selection не оставлены
внешним oracle.
Для `l=0,...,47` выполняется следующий протокол.

### Stage L1: compute and commit

Input root слоя равен output root предыдущего слоя byte-for-byte. Prover один
раз вычисляет exact layer semantics и, пока local values живы, коммитит:

- canonical output hidden state `R_out,l`;
- lookup-boundary vector `R_lk,l`;
- hidden stable-route-sort records `R_sort,l` для MoE;
- small affine-mask tape `R_mask,l`.

`R_out,l` становится `R_in,l+1`; equality roots является composition link, а
не отдельным недоказанным copy claim.

### Stage L2: first public coins

После всех L1 roots текущего слоя verifier выдаёт domain-separated randomness
для:

- output evaluation;
- fanout batching;
- route permutation fingerprints/products;
- output/Freivalds points и local LogUp tuple compression `beta_l`;
- masked tensor sumchecks.

После `beta_l` prover коммитит derived compressed fingerprints/table values
`R_cmp,l`. Только после этого verifier выдаёт reciprocal point `alpha_l`;
prover строит inverses и продолжает GKR. Таким образом порядок lookup равен

\[
R_{raw}\to\beta_l\to R_{cmp}\to\alpha_l\to
R_{inverse}/GKR.
\]

Каждый verifier challenge -- `beta`, `alpha`, все sumcheck points, terminal и
root batching coefficients, columns -- получается отдельным coin flip:
verifier предварительно коммитит nonce, prover фиксирует предыдущее message и
свой nonce, verifier открывает nonce, challenge хешируется из transcript и XOR
nonces. Offline-вариант использует
последовательную Fiat--Shamir chain
`H(manifest,layer,all previous roots/messages)`; constant или prover-supplied
seeds недопустимы. NIZK-ZK для offline path условна в random-oracle model.

### Stage L3: local tensor-GKR

Один tagged ragged sumcheck батчит relations слоя. Every prover polynomial
предшествует следующему challenge. Fanout одного canonical tensor node `v`
редуцируется после фиксации всех child claims:

\[
A_v(z)=\sum_{e\in out(v)}\beta_{v,e}\,eq(r_e,z).
\]

`node_id,edge_id,port_id,shape` входят в domain tag; два operand ports в
`x*x` не склеиваются.

Affine/residual/reshape/RoPE relations переписывают evaluation claims.
Hadamard, booleanity и raw-rescale brackets являются degree-2 sumchecks.
Dense matmul и attention используют structured contraction: случайный output
MLE claim редуцируется к child activation и weight MLE claims без
`S^2*d` gate trace.

Load-bearing causal seam устроен как **project-before-sumcheck**. Как только
output point `rho` конкретного matmul зафиксирован, но до первого challenge его
input-dimension sumcheck, prover одним проходом строит projected weight

\[
P[j]=\sum_i\chi_\rho(i)W[j,i]
\]

и коммитит полный projected RS codeword `R_P`. После этого contraction
sumcheck использует маленький `P`, а не исходный `W`. Это закрывает causal
counterexample `W=(w0,w1)`: polynomial до challenge строится уже из committed
`P`; post-challenge обращение к 400B `W` не требуется.

Lookup boundaries и sort records не раскрываются: GKR заканчивается их
linear-functional claims, которые L5 связывает с L1 roots.

### Stage L4: projected-codeword and terminal commitments

Все hidden non-weight terminal factors, masked-claim values и mask products
коммитятся в маленьком RS block. Для binary-gate normalization degree-3
relation содержит две multiplication boundaries. Global `q_quad/p0` доказывает
mask products; `q_lin` доказывает affine mask recurrences.

Manifest выравнивает RS rows разных output coordinates одного tensor по одним
и тем же message/padding positions. Поэтому prover коммитит не только `P`, но
буквально линейную комбинацию полных codewords

\[
F_P=\sum_i\chi_\rho(i)F_{W_i},
\]

включая secret padding halves. Tied embedding/LM-head segment получает два
независимых projected roots. Совокупный размер projected messages менее 60M
fields; их encode/root cost включён в reserve.

### Stage L5: LF proof and opening

Non-weight terminal claims на roots текущего слоя агрегируются random
coefficients после commitments. Terminal claims к projected `P` проверяются
обычным local LF proof над маленькими `R_P`; full-weight `q_lin` отсутствует.
Все local `q_lin`, `q_quad`, projected-root IRS polynomials и roots фиксируются
до column challenge.

В тех же 54 columns verifier открывает persistent `F_W` и corresponding `F_P`
и проверяет codeword projection equality. Если roots являются codewords, но
message `P` не равно требуемой проекции, difference -- ненулевой RS polynomial
degree `<K`, и probability пройти все columns не больше `(K/N)^54=(1/4)^54`.
Если какой-либо fresh `P` root не codeword, local IRS даёт `(3/4)^54`.
Projection openings используют ровно те же columns, что и `W`, поэтому не
добавляют независимой линейной утечки о persistent padding.
Работа verifier этой equality пропорциональна числу открытых RS row-values,
`q*N_pad/ELL`, а не `q*P`: каждый opened row-value уже кодирует 8192 message
positions. Она входит в 3.99B-value verifier corollary ниже.

Verifier выбирает 54 columns; prover re-encode/open local values, пока они ещё
resident, и persistent layer weights из удерживаемого compressed shard. Поэтому
ни terminal evaluation, ни opening encode не требуют нового чтения модели.
После ACCEPT слоя освобождается всё, кроме raw output buffer и `R_out,l`.

Это ровно один semantic realization на слой. `A_f` и `A_x` -- proof folding и
opening encode, а не повторный inference.

## 5. MoE без `S*E*K`

### 5.1 Hidden stable sort

Source record использует уникальную позицию `t`, не token id:

\[
R=(layer,kind,t,e_t,j,x_{t,j}).
\]

Sorted records коммитятся до permutation challenges. Stable order проверяется
строгим ростом

\[
k_q=e'_q(S+1)+t'_q.
\]

Permutation проверяется характеристическим произведением

\[
\prod_{src}(z-\chi_\beta(R))=
\prod_{dst}(z-\chi_\beta(R')).
\]

Layer/kind/position/expert/coordinate входят в fingerprint. При несовпадающих
formal multisets difference polynomial ненулевой, поэтому ошибка не больше
`n/p`.

### 5.2 Delimiters and scan

Hidden sequence содержит `E+1` delimiters `0,...,E`. Delimiter 0 начинает
expert 0; delimiter `e>0` выпускает accumulator expert `e-1` и сбрасывает его.
Для challenge `tau_t`:

\[
h_{q+1,j}=(1-d_q)
\left(h_{q,j}+a_q\tau_{t'_q}x'_{q,j}\right),
\quad d_q,a_q\in\{0,1\},\ d_q+a_q=1.
\]

Consecutive delimiters дают нулевой accumulator для empty expert. Если все
tokens идут одному expert, остальные segments пусты и те же equations остаются
полными. Power-of-two padding публично расположен после logical prefix и
исключён из products.

Порядок delimiters и принадлежность token rows закрепляются явным counter:

\[
c_0=0,\quad c_{q+1}=c_q+d_q,\quad c_{final}=E+1,
\]

\[
d_q(label_q-c_q)=0,\qquad
a_q(e'_q-(c_q-1))=0.
\]

Для каждого coordinate `j` ведётся отдельная lane; strict key применяется к
парам `(kind,j)` независимо, поэтому повторение `(e,t)` при разных `j` не
нарушает strict order. Delimiter emission
`(layer,kind,c_q-1,j,h_{q,j})` связывается tagged RAM/permutation equality с
fixed-order tensor `A[e,j]`. Это запрещает переставить или переметить segment.

### 5.3 Expert matmul

Для output challenge `rho_i` положим

\[
A[e,j]=\sum_{t:e_t=e}\tau_t X[t,j],\qquad
P[e,j]=\sum_i\rho_i W[e,j,i].
\]

Тогда выбранный matmul проверяется одним scalar identity

\[
\sum_{t,i}\tau_t\rho_iY[t,i]
=\sum_{e,j}A[e,j]P[e,j].
\]

Weight terminal `P` связывается L5 с `R_W`. Gate/up используют одну sort order;
SiLU и Hadamard pointwise сохраняют её; down использует тот же route, после
чего output inverse-permuted к canonical token order.

Стоимость вместо naive

\[
24S E(2d+d_{ff})=56.623\text{ B cells}
\]

составляет

\[
24S(2d+d_{ff})=442.368\text{ M segmented cells}
\]

плюс менее 492 M permutation cells.

## 6. Lookup binding

LogUp challenge нельзя выдавать до binding query/table tuples. Иначе prover
подгоняет lookup witness под sampled reciprocal point.

Pass L1 коммитит distinct raw query operands, paired outputs и локальные
multiplicities. После `beta_l`, но до `alpha_l`, отдельный `R_cmp,l` коммитит
compressed fingerprints и table-side values. Безопасная верхняя оценка
совокупности raw и compressed binding slots с дублированием shared
public-table multiplicities по 48 local arguments:

\[
K_{lookup}=37\,267\,138\,496\text{ field slots}.
\]

Все table ids получают public domain tags. После roots текущего слоя verifier
выбирает local tuple compression `beta_l` и reciprocal point `alpha_l`; inverse
wires затем доказываются GKR до освобождения слоя. Одна repetition на слой
достаточна для профиля T40; суммарный degree numerator по всем слоям:

\[
\epsilon_{lookup}\le
\frac{2(23.979840\cdot10^9)+4.059\cdot10^9}{p}
\approx2^{-28.5}.
\]

Она уже значительно меньше leading IRS error. Дополнительные repetitions нужны
для более сильного профиля, но не для сопоставимости с текущим T40.
Route-sort construction не использует `P` как dynamic lookup table keyed by
private route. `R_P` служит только weight seam для subsequent contraction;
поэтому lookup transcript не добавляет ещё одну weight projection.

## 7. Zero knowledge

### 7.1 Affine masked claims

Каждый secret scalar claim несётся как public masked value

\[
\widehat x=x+\mu_x,
\]

где mask handle закреплён pre-challenge mask root.

Для degree-`d` sumcheck round с carried mask `mu` prover использует `d` fresh
tape fields `u_1,...,u_d` и определяет

\[
h(X)=a_0+\sum_{k=1}^d u_kX^k,\qquad
a_0=\frac{\mu-\sum_{k=1}^d u_k}{2}.
\]

Тогда `h(0)+h(1)=mu`; prover посылает `g+h`, а после challenge `r` новый mask
равен `h(r)`. Отображение fresh tape fields в свободные coefficients masked
round polynomial является triangular full-rank. Поэтому каждый transcript
polynomial равномерен в affine space, заданном только предыдущим masked claim.

### 7.2 Products

Для `z=xy`, `X=x+a`, `Y=y+b`, `Z=z+c` проверяется

\[
Z-c=XY-Xb-Ya+ab.
\]

`q_lin` проверяет affine identity, `q_quad` -- mask products. Binary-gate
decomposition даёт не более двух mask products на cubic GKR terminal; общий
трёхсекретный cubic без decomposition требует четыре и здесь не используется.

При не более 3500 sumcheck epochs mask tape содержит менее 100 000 fields и
менее 14 000 scalar products. Стоимость и proof size пренебрежимы.

Soundness получается вычитанием authenticated masks: любой accepting masked
transcript преобразуется в accepting ordinary GKR transcript. Simulator
выбирает masked claims и free round coefficients равномерно; hiding RS proofs
симулируют terminal qlin/qquad view. Routes, multiplicities, activations и
weights не раскрываются. Binding требует collision resistance Merkle hash;
computational hiding commitment и offline simulation формулируются в
random-oracle model с secret RS padding.

## 8. Composition and soundness

### 8.1 Layer composition lemma

Manifest фиксирует semantic type/layout каждого interface. Verifier требует
exact root equality

\[
R_{in,l+1}=R_{out,l}.
\]

По binding RS/Merkle оба local proofs относятся к одному message vector.
Следовательно, accepting inconsistent adjacent layers требует либо false local
GKR/LF proof, либо commitment collision. Canonical metadata исключает splice
root от другого layer/shape/scale.

### 8.2 Root batching and q=54

Все roots одного layer батчатся random root coefficient после commitments и
проверяются одним IRS event. Cancellation bad syndromes добавляет `1/p`.
Across 48 layers:

\[
48(3/4)^{54}=8.61\cdot10^{-6}\approx2^{-16.83}
<(3/4)^{40}.
\]

Output root открывается как output и как следующий input максимум в 108
distinct columns; это намного меньше hiding budget 8192.

### 8.3 Union bound

Для Goldilocks field:

\[
\begin{aligned}
\epsilon\le{}&48[(3/4)^{54}+(1/4)^{54}+(3/8)^{54}+2^{-54}
+65536/p+1/p]\\
&+(47.959680\text{ B}+4.059\text{ B})/p\\
&+245.760\text{ M}/p\\
&+56.623\text{ M}/p\\
&+14000/p+48/p+\epsilon_{BLAKE3}.
\end{aligned}
\]

Leading term равен `2^-16.83`; LogUp около `2^-28.5`, route permutation
`2^-36.13`, RAM compaction `2^-38.25`, GKR/field terms меньше `2^-40`.
Итог не хуже intended comparable T40 security profile.

## 9. Efficiency model

### 9.1 Reference 8--10-hour model

Appendix A.5 для `S=1000` даёт:

\[
\begin{aligned}
W(S)&=4.00\cdot10^{11}+4.48\cdot10^8S+40320S^2
      =8.8832\cdot10^{11},\\
L(S)&=1.19\cdot10^8+1.50\cdot10^8S+12480S^2
      =1.62599\cdot10^{11},\\
Q(S)&=5.93\cdot10^7+1.54\cdot10^8S+19200S^2
      =1.732593\cdot10^{11}.
\end{aligned}
\]

\[
4T_{wit}=4.000h,
\quad(A_c+A_f+A_x)W=2.912h,
\quad DW=0.123h,
\quad CQ=0.722h,
\quad BL=0.027h.
\]

Known subtotal `7.784h`; coefficient/auxiliary residual даёт заявленный paper
range `8--10h`. Даже верхняя paper calibration соответствует multiplier лишь
`10/7.784=1.285` над известным subtotal; принятый для новой схемы `kappa=1.5`
оставляет ещё 16.7% относительного запаса сверх этой верхней calibration.

### 9.2 New counted work

Используется safe lookup volume, `E=3ns/slot`, одна local LogUp repetition на
слой и q=54.

| Component | Seconds |
|---|---:|
| one semantic forward | 3 609 |
| projected weight codewords: `E*(P+N_pad)` | 2 902.1 |
| persistent-weight opening encode: `A_x*N_pad` | 2 371.5 |
| selected local GKR: `C*Q_sel` | 632.6 |
| lookup boundary + local multiplicity RS | 570.2 |
| stable-sort RS + segmented/permutation | 20.5 |
| output/emitted roots | 7.3 |
| embedding/token/UI edge superlayers | 50 |
| projected-root encode/IRS/LF reserve | 12 |
| other structured GKR | 105 |
| explicit fold reserve | 100 |
| general orchestration reserve | 100 |
| linear/mask products | 20 |
| radix-sort traffic reserve | 29 |
| amortized weight refresh/link | 43 |

Primitive-priced proof compute excluding semantic forward therefore не больше

\[
T_{comp}\le6995\text{ s}.
\]

В частности, edge-superlayer reserve 50 s покрывает менее `1B` дополнительных
structured cells для token/embedding/LM-head/UI/hash boundary даже при цене
`C=15ns` (15 s), оставляя более чем трёхкратный запас.

Каждый weight tensor проецируется ровно один раз после фиксации его output point
и до challenges его contraction sumcheck. Из-за output-major alignment число
padding slots не равно `P`, а считается по фактической row geometry:

\[
N_{pad}=8192\sum_t c_t n_{out,t}
\left\lceil\frac{n_{in,t}}{8192}\right\rceil.
\]

Для всех Maverick families, включая обе tied embedding/head orientations,

| Family | Count | `n_in` | `n_out` | P, fields | N_pad, fields |
|---|---:|---:|---:|---:|---:|
| Attention QKVO | 192 | 5120 | 5120 | 5,033,164,800 | 8,053,063,680 |
| Dense gate/up | 48 | 5120 | 16384 | 4,026,531,840 | 6,442,450,944 |
| Dense down | 24 | 16384 | 5120 | 2,013,265,920 | 2,013,265,920 |
| MoE expert gate/up | 6144 | 5120 | 8192 | 257,698,037,760 | 412,316,860,416 |
| MoE expert down | 3072 | 8192 | 5120 | 128,849,018,880 | 128,849,018,880 |
| MoE shared gate/up | 48 | 5120 | 8192 | 2,013,265,920 | 3,221,225,472 |
| MoE shared down | 24 | 8192 | 5120 | 1,006,632,960 | 1,006,632,960 |
| Router | 24 | 5120 | 128 | 15,728,640 | 25,165,824 |
| Embedding view | 1 | 202048 | 5120 | 1,034,485,760 | 1,048,576,000 |
| LM-head view | 1 | 5120 | 202048 | 1,034,485,760 | 1,655,177,216 |
| Gains | 97 | 5120 | 1 | 496,640 | 794,624 |


\[
P=402{,}725{,}114{,}880,
\quad N_{pad}=564{,}632{,}231{,}936,
\]

\[
E(P+N_{pad})=2902.072\text{ s},\qquad
A_xN_{pad}=2371.455\text{ s}.
\]

Упаковывать slack разных output coordinates в одну row нельзя: один opened
functional `L_eta(m)` не позволяет получить
`rho_1 L_eta(m|B1)+rho_2 L_eta(m|B2)` для независимых `rho_1,rho_2`.
Следовательно, padding slack здесь посчитан, а не скрыт. Последующий GKR
работает только с менее 60M projected fields. Ни pre-challenge GKR scan
исходного `W`, ни post-challenge full-weight `q_lin/A_f` больше нет.

Opening volume обязан считать полную row capacity, а не только nonzero model
entries. После замены `P` на `N_pad` безопасная occurrence bound равна
`603.91B` capacity fields. При
q=54 raw binary openings:

\[
P_{raw}=\frac{8\cdot54\cdot603.91B}{8192}\le31.85\text{ GB}.
\]

После добавления не более 0.29 GB local polynomials, Merkle paths и manifest
metadata полный binary proof ограничен 32.14 GB. Эти байты не добавляются после
`A_x`: opening encoder производит их за 2371.5 s, то есть не быстрее
13.5 MB/s, тогда как serializer/disk обслуживает 108 MB/s, а 1 Gbit/s network
-- 125 MB/s. Для bounded queue producer/consumer lemma даёт

\[
T_{opening+dump+send}=\max(T_{A_x},T_{dump},T_{net})+T_{tail}=T_{A_x}+T_{tail},
\]

поскольку обе drain rates строго выше producer rate. Metadata tail ограничен
10 s. 3500 GKR RTT и две дополнительные local-LogUp challenge epochs на слой
дают менее 76 s. Refresh/link входит в primitive bucket как 43 s до умножения
на `kappa`.

Canonical frame имеет вид
`(layer_id,root_id,row_range,54 opened values,auth fragment)`. После каждого
row chunk encoder немедленно enqueue frame; disk и network читают один tee
параллельно, verifier ведёт 54 incremental column-hash states. Merkle paths и
не более 0.29 GB metadata завершаются в tail. Никакой последующий prover
polynomial не зависит от полного opening file, поэтому следующий layer coin
может быть выдан до окончания verifier arithmetic; итоговый ACCEPT всё равно
требует успешной проверки всех предыдущих frames. Queue cap 1.26 GB превышает
один encoder burst и входит в memory ledger.

### 9.3 Calibrated bound

Point identity даёт около 2.97 h. Чтобы не повторять ошибку floor-as-runtime,
умножаем **все proof-compute primitive terms** на явно принятую calibration
hypothesis 1.5:

\[
T_{binary}\le3609+1.5(6995)+76+10
=14\,187.5\text{ s}<14\,200\text{ s}=3.95\text{ h}.
\]

Break-even multiplier:

\[
\kappa_{max}=\frac{14400-3609-76-10}{6995}\approx1.530.
\]

Для достаточного округлённого условия используется `kappa<=1.53`. Legacy JSON,
материализованный после завершения proof, не является частью четырёхчасового
режима; при необходимости он строится как отдельный archival artifact.

### 9.4 Peak and verifier

Peak закрывается phase-liveness. `B_common=83.89 GB` -- консервативный cap
существующего unified-memory workspace, включающий row-chunk/NTT allocator.
`B_W,max<=11.5 GB` -- условие теоремы на полный compressed layer shard:

| Phase | Live set | Bound |
|---|---|---:|
| semantic compute/root stream | measured common workspace, shard уже входит в old path | <=83.89 GB |
| route radix | `B_common+5.77 raw+17.37 ping-pong+1.36 tables+B_W,max` | <=119.89 GB |
| LogUp/GKR | `B_common+5.77 raw+4.34 inverses+1.36 tables+B_W,max` | <=106.86 GB |
| L5 all openings | `B_common+B_W,max+5.77 lookup+1.0 sort/meta+0.96 projected+1.26 opening/queue` | <=105.38 GB |

Compressed shard удерживается через все четыре фазы; повторного disk read и
semantic recompute нет. Radix ping-pong освобождается до reciprocal/GKR phase;
raw lookup/sort messages удерживаются до L5 columns, открываются и только затем
освобождаются. Никакой 213 GB global lookup vector не создаётся. Следовательно
maximum 119.89 GB меньше 121 GB.

Streaming GPU verifier читает не более 32.14 GB и проверяет около 3.99B opened
field values. При отдельной гипотезе arithmetic throughput не хуже
`15ns/value`, input не хуже 108 MB/s и 60-second scheduling reserve его
projected time меньше 420 s. Это conditional corollary; существующий CPU
verifier этой оценки не достигает и в prover theorem не используется.

## 10. Real deployment policy

Четырёхчасовая теорема относится к online proof для уже enrolled model. Это
обычный режим многократной верификации одной 400B модели. Cold enrollment
создаёт model manifest и weight roots один раз независимо от prompt/output.
Его amortized refresh/link cost включён.

Если требуется единственный cold proof, включая первичное построение всех
weight roots, этот setup должен быть добавлен отдельно; данная теорема такого
условия не заявляет.

Verifier policy также фиксирует:

- retry/abort budget;
- cumulative opened-column ledger;
- expected model/DAG/table/transcript digests;
- exact interactive или sequential-FS challenge derivation;
- canonical binary serialization;
- отсутствие локального GPU-address side-channel в threat model либо
  oblivious implementation для route-dependent memory access.

## 11. Что именно улучшено

1. Flat Ligero trace заменён layer-local tensor-GKR.
2. Four semantic witness sweeps заменены одним.
3. `S*E*K` private MoE tensor заменён hidden stable sort + segmented
   contractions.
4. Коммитятся только lookup boundaries, необходимые для challenge ordering.
5. Exact output root является next input root, поэтому нет IVC splice.
6. Affine-mask compiler закрывает recursive ZK, включая nonlinear mask terms.
7. `q=54` выводится из union bound 48 local arguments, а не переносится
   механически из одного Ligero proof.
8. Runtime использует paper's 8--10h analytical model и явную hypothesis
   `kappa<=1.5`, а не 14.26h headline как baseline.

## 12. Реестр семейств подходов

| Семейство | Статус | Точный результат аудита |
|---|---|---|
| Parameter-only Ligero: coset NTT, BabyBear, spill removal, retuning | rejected | Не устраняет четыре semantic/encode sweeps и `S*E*K`; Amdahl bound недостаточен для 2x от 8 h. |
| Project-once + private dynamic expert lookup | blocked as whole family | Алгебра требует `rho -> commit W^rho -> beta -> commit fingerprints -> alpha` и большой private-table inverse argument. Выбран только codeword-projection seam; route lookup заменён sort/segment. |
| Transparent multilinear PCS / Basefold / Orion | blocked | Может заменить LF-Ligero, но в репозитории нет спецификации hiding, proximity и operation bound через уже калиброванные primitives. Такой путь просто переименовывал отсутствующую theorem-level implementation lemma. |
| Pure tensor-GKR с Merkle root весов | rejected | Terminal arbitrary MLE weight claim не аутентифицируется Merkle root без PCS-equivalent argument; sampling raw leaves имеет ошибку почти 1. |
| Designated-verifier secret weight sketches | rejected for target | Даёт простой sub-4h protocol, но меняет public-verifier trust model: verifier/custodian должен знать веса и держать secret challenges. |
| Global GKR + global LogUp | blocked | Один `alpha` после всех roots требует удерживать более 200 GB lookup boundaries либо второй full semantic pass. |
| **Layer-local tensor-GKR + codeword projection + route sort** | **selected** | Local challenges разрешают освобождать слой; route sort устраняет `S*E*K`; project-before-sumcheck закрывает causal seam, а same-column RS equality аутентифицирует projection без full-weight qlin. |

Adversarial register, проверенный для выбранного пути: policy/model-root binding;
external transcript anchoring; commit-before-challenge для Freivalds, LogUp и
IRS; field wrap/range/rescale; zero padding; fanout/IVC splice; lookup address
binding; empty/all-one-expert routes; duplicate token ids; recursive ZK masks;
cumulative opening leakage; shared-column union bound; HBM/allocator peak;
storage/network/RTT; retry/abort policy; route-dependent address side channel.

## 13. Локальные исходные опорные данные

- `analysis/paper.md`, Appendix A.5--A.6: 8--10 h model, `W,L,Q` и primitive
  coefficients;
- `analysis/appendix-moe-routing.md`, B.7: cubic MoE contraction floor и
  граница materialized selector products;
- `analysis/full-model-hidden-run-archive.md`: `S=1000`, one-pass witness time,
  memory high-water и serialization throughput;
- `prover/core.py` и `verifier/src/verify.rs`: RS/Ligero polynomial ordering,
  persistent roots и необходимость fresh post-root challenges.

Эти данные используются как параметры математической модели; headline
artifact с заранее известными seeds не используется как доказательство
soundness или как performance baseline.
