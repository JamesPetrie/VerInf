# Related work: candidate citations for the paper

Working document for `paper.md` §3 (Related work) and for the citations the rest of the paper needs (Ligero, LogUp, Freivalds, and the currently unattributed Hyrax, BLAKE3, Goldilocks, and Reed-Solomon mentions). Compiled 2026-07-19 from public sources; every entry marked VERIFIED was checked against the abstract or paper text on that date, with the supporting quote or figure noted. Entries marked from-survey were taken from an earlier literature pass and not independently re-verified; UNVERIFIED flags a specific fact we could not confirm. Per the research-integrity rules, do not cite a claim from this document without the verification status supporting it.

The paper's §3 TODO asks for exactly this: additional zero-knowledge inference work beyond zkLLM, positioned against **scale**, **statement proven**, and **commitment assumptions**. Those three axes organize the tables below.

## 1. The headline comparison, verified

**Frontier as of July 2026: no published system demonstrates an end-to-end cryptographic ZK proof of LLM inference above 13B parameters.** A targeted search for January-July 2026 work found no new ZK inference proof above 1B: the 2026 entries at real scale all relax the trust model (sampling spot-checks, statistical re-execution, or TEEs), and the true-ZK 2026 entries are toy-scale. Systems claiming larger architectures (ZK-DeepSeek, 671B) are capability claims benchmarked per-operation, not demonstrated runs. This makes the paper's "roughly thirty times" gap against zkLLM safe, and arguably understated if measured against demonstrated end-to-end runs.

| System | Demonstrated scale | Statement proven | Commitments | Post-quantum |
|---|---|---|---|---|
| zkLLM (CCS 2024) | LLaMA-2-13B, seq 2048, 803 s on A100 | quantized integer circuit with tolerance-band lookups | Hyrax (Pedersen variant) on BLS12-381, transparent | no |
| zkGPT (USENIX Sec 2025) | GPT-2 (~124M), 21.8 s on 32 CPU threads | quantized integer circuit, untrusted-advice + lookups | Hyrax on BN254, transparent | no |
| zkPyTorch/Expander (ePrint 2025/535) | Llama-3 8B, 150 s/token, single CPU thread | quantized circuit (M61 field), ~99.3% output cosine similarity | GKR pipeline; PCS for this run UNVERIFIED | plausibly (unconfirmed) |
| ZKTorch (arXiv 2025) | GPT-J 6B in 20 min, 64 threads (from-survey) | compiled quantized circuit, accumulation scheme | Mira-based accumulation (from-survey) | not stated |
| ZK-DeepSeek (arXiv 2025) | 671B claimed; only per-op benchmarks (one matmul ~57 h) | quantized per-operation circuits, recursively composed | Kimchi/Pickles on Pasta curves | no |
| **VerInf (this paper)** | Llama-4-Maverick 400B, 1000 tokens end to end, accepted proof | bound on unexplained information; unique witness upstream of logits | hash-based only (Ligero + BLAKE3 Merkle) | plausibly |

Apples-to-apples caution for §3: zkLLM's 803 s (~13.4 min) is a single forward pass at sequence length 2048, which matches how our 14.3 h figure is measured (a single 1000-token forward pass), so the comparison is fair; do not compare against zkLLM's autoregressive-generation extrapolations.

## 2. Zero-knowledge proofs of LLM inference (§3, first paragraph)

**Must cite.**

- **zkLLM** (already cited). Haochen Sun, Jason Li, Hongyang Zhang. "zkLLM: Zero Knowledge Proofs for Large Language Models." ACM CCS 2024. DOI 10.1145/3658644.3670334; arXiv:2404.16109. VERIFIED: 13B at seq 2048 in 803 s on an A100 (Table 1); abstract: "for LLMs boasting 13 billion parameters, our approach enables the generation of a correctness proof for the entire inference process in under 15 minutes"; the paper states it uses "Hyrax (Wahby et al., 2018), a variant of the Pedersen commitment ... as an instantiation of the polynomial commitment scheme" over BLS12-381. Proves quantized computation with bounded-error lookups, not bit-exact float inference (reports an L1 output error near 1e-2 and separate quantized-vs-original perplexities). https://arxiv.org/abs/2404.16109
- **zkGPT.** Wenjie Qu, Yijun Sun, Xuanming Liu, Tao Lu, Yanpei Guo, Kai Chen, Jiaheng Zhang. "zkGPT: An Efficient Non-interactive Zero-knowledge Proof Framework for LLM Inference." USENIX Security 2025; ePrint 2025/1184. VERIFIED: "our scheme can prove GPT-2 inference in less than 25 seconds" (21.8 s, 32 threads, Table 3); Hyrax over BN254 (~100-bit security), GKR + Lasso lookups; quantized 16-bit integer inference. The current speed frontier at GPT-2 scale. https://eprint.iacr.org/2025/1184
- **zkPyTorch / Expander.** Tiancheng Xie, Tao Lu, Zhiyong Fang, Siqi Wang, Zhenfei Zhang, Yongzheng Jia, Dawn Song, Jiaheng Zhang. "zkPyTorch: A Hierarchical Optimized Compiler for Zero-Knowledge Machine Learning." ePrint 2025/535 (no venue found). VERIFIED scale/time: "150 seconds per token for Llama-3 inference" (8B, single CPU thread; abstract). Quantized circuits over the M61 field via the GKR-based Expander prover. UNVERIFIED: which polynomial commitment backs the Llama-3 run (the PDF was inaccessible; Expander materials mention both hash-based and KZG), so do not assert its post-quantum status in the paper without checking; also no published end-to-end GPU per-token number as of mid-2026, only component-level GPU acceleration claims in press. https://eprint.iacr.org/2025/535

**Cite if space allows.**

- **ZKTorch.** Bing-Jyue Chen, Lilia Tang, Daniel Kang. "ZKTorch: Compiling ML Inference to Zero-Knowledge Proofs via Parallel Proof Accumulation." arXiv:2507.07031, 2025. From-survey: GPT-J 6B in 20 minutes on 64 threads via an extended Mira accumulation scheme; re-verify before citing figures. https://arxiv.org/abs/2507.07031
- **ZK-DeepSeek.** Yunxiao Wang. "Zero-Knowledge Proof Based Verifiable Inference of Models." arXiv:2511.19902, 2025 (preprint only, single author). VERIFIED caveat: the 671B DeepSeek-V3 support is an architecture-compatibility claim; experiments benchmark individual operations only (embedding 4,823 s; one matmul 204,138 s, roughly 57 h), with no end-to-end or per-token time. Kimchi (PLONKish) with Pickles recursion on the Pasta curves; not post-quantum. If cited, cite it as a claims-versus-demonstrations contrast, and do so carefully. https://arxiv.org/abs/2511.19902
- **Artemis.** Hidde Lycklama, Alexander Viand, Nikolay Avramov, Nicolas Küchler, Anwar Hithnawi. "Artemis: Efficient Commit-and-Prove SNARKs for zkML." arXiv:2409.12055. VERIFIED: shows commitment-consistency checks are the zkML bottleneck ("for the VGG model, we reduce the overhead associated with commitment checks from 11.5x to 1.1x"); venue UNVERIFIED (dblp lists only CoRR). Relevant when discussing why committing the witness dominates cost. https://arxiv.org/abs/2409.12055
- **Peng et al. survey.** Zhizhi Peng, Chonghe Zhao, Taotao Wang, et al. "A Survey of Zero-Knowledge Proof Based Verifiable Machine Learning." arXiv:2502.18535, 2025. From-survey: covers ZKML June 2017 to August 2025. Useful as the single pointer for the long tail. https://arxiv.org/abs/2502.18535

**2026 items found and why they do not change §3.** NanoZK (arXiv:2603.18046, ICLR 2026 VerifAI workshop) is true ZK but toy-scale (transformers up to d=128). Anchuri et al. (arXiv:2603.19025) reach Llama-2-7B but explicitly trade soundness for efficiency (Merkle spot-checks of the trace, not a ZK proof). VeriAttn (arXiv:2606.16352) is TEE-based. All VERIFIED from abstracts. None demonstrates cryptographic ZK inference above 1B.

## 3. The pre-LLM zkML lineage (§3, compact mention or footnote)

All VERIFIED from abstracts unless noted. One sentence in §3 with two or three of these suffices; zkCNN and ZKML are the strongest picks.

- **zkCNN.** Tianyi Liu, Xiang Xie, Yupeng Zhang. CCS 2021; ePrint 2021/673. VGG16 (15M params) in 88.3 s; GKR with a transparent PCS.
- **ZKML.** Bing-Jyue Chen, Suppakit Waiwitlikhit, Ion Stoica, Daniel Kang. EuroSys 2024. DOI 10.1145/3627703.3650088. "the first framework ... to produce ZK-SNARKs for realistic ML models, including ... a distilled GPT-2"; halo2-based.
- **Mystique.** Chenkai Weng, Kang Yang, Xiang Xie, Jonathan Katz, Xiao Wang. USENIX Security 2021; ePrint 2021/730. ResNet-101 inference in 28 minutes; VOLE-based, designated verifier.
- **ZEN.** Boyuan Feng, Lianke Qin, Zhenfei Zhang, Yufei Ding, Shumo Chu. ePrint 2021/087 (preprint). R1CS-friendly quantization; small CNNs.
- **vCNN.** Seunghwa Lee, Hankyung Ko, Jihye Kim, Hyunok Oh. ePrint 2020/584. Pairing-based SNARK with trusted setup; MNIST/VGG16.
- **Kang et al.** "Scaling up Trustless DNN Inference with Zero-Knowledge Proofs." arXiv:2210.08674. From-survey: first ImageNet-scale ZK-SNARK inference proof.
- Training-side (only if §3 mentions training): **Kaizen** (Abbaszadeh, Pappas, Katz, Papadopoulos, CCS 2024, ePrint 2024/162; VGG-11 10M params, 15 min per gradient-descent iteration) and **Garg et al.** (CCS 2023, ePrint 2023/1345; logistic regression only). VERIFIED. No verifiable training at LLM scale exists.

## 4. Verified inference with a trusted verifier (§3, last paragraph)

The paper already cites Rinberg et al. 2025 and Karvonen et al. 2025. Full citations, plus neighbors worth one collective sentence.

- **Rinberg et al.** Roy Rinberg, Adam Karvonen, Alexander Hoover, Daniel Reuter, Keri Warr. "Verifying LLM Inference to Detect Model Weight Exfiltration." arXiv:2511.02620, 2025. VERIFIED: recomputation-based verification; "we characterize valid sources of non-determinism in large language model inference and introduce two practical estimators"; reduces exfiltratable information to under 0.5% on MOE-Qwen-30B. https://arxiv.org/abs/2511.02620
- **Karvonen et al. (DiFR).** Adam Karvonen, Daniel Reuter, Roy Rinberg, Luke Marks, Adrià Garriga-Alonso, Keri Warr. "DiFR: Inference Verification Despite Nondeterminism." arXiv:2511.20621, 2025. VERIFIED: token- and activation-level comparison against a seed-synchronized trusted reference. Note for §3: the two are companion works with overlapping author teams; the text should not present them as independent lines. https://arxiv.org/abs/2511.20621
- **TOPLOC.** Jack Min Ong et al. "TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference." arXiv:2501.16007, 2025. VERIFIED: locality-sensitive hashes of top-k activations, robust to hardware variation; attestation-style, for decentralized compute. https://arxiv.org/abs/2501.16007
- **Verde.** Arasu Arun et al. (incl. Joseph Bonneau). "Verde: Verification via Refereed Delegation for Machine Learning Programs." arXiv:2502.19405, 2025. VERIFIED: refereed-delegation arbitration plus RepOps, bitwise-reproducible floating-point operators; the make-it-deterministic alternative to tolerating nondeterminism. https://arxiv.org/abs/2502.19405
- **SVIP.** Yifan Sun et al. "SVIP: Towards Verifiable Inference of Open-source Large Language Models." arXiv:2410.22307. VERIFIED: secret-based protocol against model substitution. Optional. https://arxiv.org/abs/2410.22307
- **Model Equality Testing.** Irena Gao, Percy Liang, Carlos Guestrin. arXiv:2410.20247. VERIFIED abstract (statistical two-sample test identifying which model an API serves); ICLR 2025 acceptance UNVERIFIED, cite as arXiv unless confirmed. Optional. https://arxiv.org/abs/2410.20247
- **Cankaya.** Naci Cankaya. "Bit-Exact AI Inference Verification Without Performance Tradeoffs." arXiv:2606.00279, 2026. VERIFIED: argues GPU inference is deterministic but not invariant, demonstrates bitwise recomputation across GPU variants, explicitly motivated by AI governance. Directly relevant to §1's "how easily a given serving stack could be made bit-exact is difficult to judge from outside": this is the strongest published counterpoint, so consider engaging with it rather than only citing it. https://arxiv.org/abs/2606.00279
- **He / Thinking Machines.** Horace He and Thinking Machines Lab. "Defeating Nondeterminism in LLM Inference." Connectionism blog, Sep 2025, DOI 10.64434/tml.20250910. VERIFIED: identifies batch invariance as the dominant cause of serving nondeterminism; the standard citation for why bit-exact serving is hard today. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

## 5. AI-governance and compute-verification context (§1; the companion [cite] neighborhood)

The two [cite] placeholders are the companion framework paper and are not resolvable from public sources here. Around them, if §1 or a discussion section wants grounding:

- **Shavit.** Yonadav Shavit. "What does it take to catch a Chinchilla? Verifying Rules on Large-Scale Neural Network Training via Compute Monitoring." arXiv:2303.11341, 2023. VERIFIED. The foundational compute-verification proposal; its training-transcript proofs are the training-side analogue of proving inference.
- **flexHEG reports.** Part I: James Petrie, Onni Aarne, Nora Ammann, David Dalrymple. "Flexible Hardware-Enabled Guarantees for AI Compute." arXiv:2506.15093, 2025 (VERIFIED). Part II: James Petrie, Onni Aarne. "Technical Options for Flexible Hardware-Enabled Guarantees." arXiv:2506.03409 (VERIFIED; canonical source for the interlock design). Part III: arXiv:2506.15100 (author list UNVERIFIED). The hardware layer that would produce §2.4's recorded digests.
- **Wasil et al.** "Verification methods for international AI agreements." arXiv:2408.16074, 2024. VERIFIED.
- **Scher and Thiergart.** "Mechanisms to Verify International Agreements About AI Development." arXiv:2506.15867 (MIRI report 2024). VERIFIED.
- **Baker et al.** Mauricio Baker, Gabriel Kulp, Oliver Marks, Miles Brundage, Lennart Heim. "Verifying International Agreements on AI: Six Layers of Verification..." arXiv:2507.15916, 2025 (RAND WR-A4077-1). VERIFIED.
- Optional broader framing: Sastry et al., "Computing Power and the Governance of Artificial Intelligence," arXiv:2402.08797, 2024 (VERIFIED).

## 6. Cryptographic building blocks (citations the body already needs)

- **Ligero** (cited as Ames et al. 2017). Scott Ames, Carmit Hazay, Yuval Ishai, Muthuramakrishnan Venkitasubramaniam. "Ligero: Lightweight Sublinear Arguments Without a Trusted Setup." ACM CCS 2017, pp. 2087-2104. DOI 10.1145/3133956.3134104. Journal version: Designs, Codes and Cryptography, 2023, DOI 10.1007/s10623-023-01222-8. VERIFIED.
- **LogUp** (cited as Haböck 2022). Ulrich Haböck. "Multivariate lookups based on logarithmic derivatives." ePrint 2022/1530. Preprint only; cite the ePrint. VERIFIED.
- **Freivalds.** Rūsiņš Freivalds. "Probabilistic Machines Can Use Less Running Time." Information Processing 77 (IFIP Congress), pp. 839-842, 1977. VERIFIED (distinct from his 1979 MFCS paper; cite the 1977 one for the matmul check).
- **Hyrax** (mentioned in §3, currently uncited). Riad S. Wahby, Ioanna Tzialla, abhi shelat, Justin Thaler, Michael Walfish. "Doubly-efficient zkSNARKs without trusted setup." IEEE S&P 2018; ePrint 2017/1132. VERIFIED. §3's discrete-log characterization of Hyrax matches zkLLM's own description of its commitment.
- **Sumcheck and GKR** (if §3's "sumcheck-based argument" gets a citation). Lund, Fortnow, Karloff, Nisan. "Algebraic Methods for Interactive Proof Systems." J. ACM 39(4), 1992. Goldwasser, Kalai, Rothblum. "Delegating Computation: Interactive Proofs for Muggles." STOC 2008; J. ACM 62(4), 2015. VERIFIED. Optional: Thaler, "Time-Optimal Interactive Proofs for Circuit Evaluation," arXiv:1304.3812, 2013, for prover-efficiency context.
- **BLAKE3** (used in §5, currently uncited). Jack O'Connor, Jean-Philippe Aumasson, Samuel Neves, Zooko Wilcox-O'Hearn. "BLAKE3: one function, fast everywhere." Technical specification, 2020, github.com/BLAKE3-team/BLAKE3-specs. No peer-reviewed venue; cite as a technical report. VERIFIED.
- **Goldilocks field** (used in §4/§5, currently uncited). Canonical source is the Plonky2 whitepaper: Polygon Zero Team, "Plonky2: Fast Recursive Arguments with PLONK and FRI," 2022, in the plonky2 repository. PARTIALLY VERIFIED (repo and field confirmed; exact title page not re-fetched). A pre-ZK alternative is Solinas primes (Solinas 1999).
- **Reed-Solomon.** Irving S. Reed, Gustave Solomon. "Polynomial Codes over Certain Finite Fields." J. SIAM 8(2), pp. 300-304, 1960. VERIFIED (DOI 10.1137/0108018 standard but not re-fetched).
- Optional for §9's proof-size discussion (hash-based alternatives with better asymptotics): **Brakedown** (Golovnev, Lee, Setty, Thaler, Wahby, CRYPTO 2023, ePrint 2021/1043; VERIFIED), **BaseFold** (Zeilberger et al., CRYPTO 2024, ePrint 2023/1705; from-survey), **DeepFold** (Guo et al., USENIX Security 2025, ePrint 2024/1595; VERIFIED, includes GPT-2-dimension matmul benchmarks).

## 7. Recommended citation set for §3, summarized

Minimal (keeps §3 near its current length): zkLLM; zkGPT; zkPyTorch/Expander; one lineage sentence citing zkCNN and ZKML with the Peng et al. survey for the long tail; Rinberg et al. and Karvonen et al. (as companion works), with one sentence adding TOPLOC and Verde as the attestation-style and make-it-deterministic corners of the trusted-verifier design space; Hyrax cited where it is named.

Additions worth considering: ZKTorch (6B accumulation-based, strengthens the scale ladder); ZK-DeepSeek (only with the claims-versus-demonstrations caveat); Cankaya 2026 (engage the bit-exactness counterpoint explicitly, likely in §9 or wherever the deployment-modification argument recurs); Artemis (if §8's commitment-cost discussion wants an external anchor).

## 8. Open items

- The companion framework [cite] placeholders remain unresolved in this document.
- zkPyTorch's polynomial commitment for the Llama-3 run: check the ePrint PDF before making any post-quantum claim about it.
- Artemis venue, flexHEG Part III authors, Model Equality Testing's ICLR status, Plonky2 whitepaper title page: verify before camera-ready if cited.
- The paper has no references.bib yet; when one is created, this document is the source for entries and keys.
