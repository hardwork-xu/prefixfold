# Research rationale and algorithm

[简体中文](../zh/RESEARCH.md) · [Home](../../README.md)

I chose **PrefixFold: shared-prefix decode attention on CPUs** to make a specific systems question inspectable on ordinary development hardware: when several requests attend to identical prefix keys and values, can grouping their queries improve operator latency without changing the attention result? I want this repository to remain small enough to explain and maintain, while preserving the evidence that would justify replacing its implementation later.

## Candidate selection

| Candidate | Research and engineering opportunity | Cost and uncertainty | Decision |
|---|---|---|---|
| Speculative decoding | Acceptance distributions, verification scheduling, useful-token throughput | A convincing comparison needs compatible models and a representative generation workload; draft quality can dominate systems effects | Defer |
| KV quantization | Storage layout, dequantization cost, error propagation | Real quality claims need model-level evaluation; a low error on random tensors is insufficient | Defer |
| Shared-prefix decode attention | Data reuse, matrix layout, stable reduction, latency/memory tradeoffs | CPU operator experiments are feasible with no model download; benefits depend strongly on batch and prefix length | Select |

This is an attention-operator library and an experimental artifact for inference engineers and systems students. It is not a serving platform, a pretrained language model, a tokenizer, or a training framework.

## Research question and falsifiable hypotheses

The central question is whether exposing identical prefix KV data as one matrix and collecting queries across requests produces a useful CPU execution pattern compared with a practical dense attention implementation.

The working hypotheses are: a longer common prefix and larger batch can amortize grouping and reduction overhead; short prefixes, one request, and long private suffixes can remove that advantage; tiling bounds score workspace but may reduce matrix-multiplication efficiency. These are hypotheses, not measured results. The fixed [targets.json](../../research/targets.json) sets a primary target of at least **80% owned KV payload reduction** against physically duplicated dense KV, and a secondary target of at least **1.25× median operator speedup** over PyTorch SDPA, for $B=16,H=4,P=4096,S=128,D=64$, float32, one CPU thread. Correctness requires `atol=3e-5, rtol=3e-5` against float64 across every workload. The protocol uses five warmups and 30 randomized-order repetitions. These are **Target**, not **Measured**. Workloads and results are documented in [Experiments](EXPERIMENTS.md). Changing those targets after observation requires retaining the original target and explaining the change.

## Existing work and contribution boundary

The following primary sources were checked on 2026-09-21. The source register is [sources.json](../../research/sources.json).

| Source | What it establishes | Relationship to PrefixFold |
|---|---|---|
| [Milakov and Gimelshein, online normalizer, 2018](https://arxiv.org/abs/1805.02867v2) | A running maximum can rescale the softmax normalizer during a scan | Numerical basis of online state updates |
| [Dao et al., FlashAttention, 2022](https://arxiv.org/abs/2205.14135v2) | Exact attention can use tiles and rescaled intermediate statistics | Basis of bounded score workspace; its GPU IO bounds do not establish CPU speed |
| [Ye et al., Cascade Inference, 2024](https://flashinfer.ai/2024/02/02/cascade-inference.html) | Shared-prefix attention, private-suffix attention, and state merging can be separated | Direct prior art for the decomposition |
| [Juravsky et al., Hydragen, 2024](https://arxiv.org/abs/2402.05099v2) | Queries from different sequences can be batched over identical prefix KV | Direct prior art for replacing repeated matrix-vector products with grouped matrix multiplication |
| [FlashInfer cascade API](https://docs.flashinfer.ai/api/cascade.html) | Implements merging of attention output and logsumexp states | Established implementation reference; not a dependency or CPU baseline |
| [Wang et al., CoDec, 2026 revision](https://arxiv.org/abs/2505.17694v2) | Addresses irregular shared-prefix trees and GPU workload balancing | Broader related problem outside this version's single-prefix scope |
| [PyTorch SDPA API](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html) | Defines scaling, dropout, and masking semantics | Practical CPU baseline contract; the actual installed version is recorded with each run |

The paper initially called **FlashForge** in arXiv:2505.17694v1 is titled **CoDec: Prefix-Shared Decoding Kernel for LLMs** in v2, revised 2026-03-28. This repository uses the current title and does not transplant its reported GPU performance to CPUs. [Version history](https://arxiv.org/abs/2505.17694).

PrefixFold's contribution is an independently written CPU engineering implementation and a reproducible examination of this established decomposition: explicit reusable prefix storage, grouped NumPy matrix multiplication, tiled stable attention states, suffix merging, validation, and a grouping ablation. NumPy and its BLAS implementation provide matrix multiplication; PyTorch provides the baseline kernel. PrefixFold provides the decomposition, state handling, input contract, and measurement plumbing. There is no claim of a new attention identity, first discovery, SOTA, or general superiority over existing attention systems. No third-party kernel source is copied into this repository.

## Mathematical contract

Consider one decode query per request and head. Let $B$ be the request count, $H$ the head count, $D$ the head width, $P$ the common-prefix length, and $S_i$ request $i$'s live private-suffix length. Let $S$ denote the common suffix length only for uniform benchmark cases. For request $i$ and head $h$:

$$
K_{i,h}=[K^p_h;K^s_{i,h}],\qquad
V_{i,h}=[V^p_h;V^s_{i,h}],\qquad
o_{i,h}=\operatorname{softmax}(q_{i,h}K_{i,h}^{\mathsf T}/\sqrt D)V_{i,h}.
$$

All stored positions are visible to this decode query. There is no future-token position in the supplied cache. The API operates on projected KV arrays; the caller is responsible for establishing that shared arrays really are identical, including positional transforms. Equal text fragments alone do not prove equal KV tensors.

For a nonempty subset $A$ of cache positions, with scores $x_j=qk_j/\sqrt D$, define:

$$
m_A=\max_{j\in A}x_j,\quad
\ell_A=\sum_{j\in A}e^{x_j-m_A},\quad
z_A=\sum_{j\in A}e^{x_j-m_A}v_j.
$$

For disjoint $A,C$, let $m=\max(m_A,m_C)$, $a=e^{m_A-m}$, and $c=e^{m_C-m}$. Then:

$$
\ell=a\ell_A+c\ell_C,\qquad z=az_A+cz_C,\qquad o=z/\ell.
$$

This follows by factoring $e^{-m}$ from each sum. It proves by induction that merging all tiles and the suffix gives the same real-arithmetic attention as a single dense softmax. These are the established online-normalization and attention-reduction identities, expressed in the implementation's unnormalized state convention. [Online normalizer](https://arxiv.org/abs/1805.02867v2), [FlashAttention](https://arxiv.org/abs/2205.14135v2).

An empty segment contributes zero mass and zero numerator; initialization must handle it explicitly instead of evaluating $-\infty-(-\infty)$. A request must have at least one visible KV position to define a normalized result. Floating-point addition is not associative, so tile order may change the last bits. “Exact attention” describes the mathematical operation, not bitwise identity or unlimited numerical range.

## Execution and complexity

For each head, group the $B$ queries into $Q_h\in\mathbb R^{B\times D}$. For each prefix tile $K^p_{h,t}\in\mathbb R^{T\times D}$, compute $Q_h(K^p_{h,t})^{\mathsf T}$, form the tile state, and merge it into the running state. Compute each request's private suffix state and merge it into that request's prefix state. Turning grouping off evaluates the same segments separately per request, giving an ablation that preserves the mathematical work.

The attention arithmetic remains $\Theta(HD(BP+\sum_i S_i))$; grouping does not remove query-key dot products. Prefix KV storage is $2HPD$ scalar elements instead of $2BHPD$ elements in a physically duplicated dense cache. For a suffix capacity $C$ per request, owned suffix storage is $2BHCD$, even when some slots are unused. The saved logical KV capacity against a dense cache with the same suffix capacity is therefore $2(B-1)HPD$ elements. These are allocation formulas, not measured resident memory or measured memory traffic.

With prefix tile width $T$, score workspace is $O(BHT)$ if all heads are processed together, plus $O(BHD)$ state. Sequential head processing can reduce the live score workspace to $O(BT)$. Private-suffix processing, public validation, temporary NumPy allocations, and BLAS scratch space must also be counted when interpreting process memory. Tiling cannot promise that the processor loads each KV byte only once: cache behavior, layouts, and the BLAS backend determine physical traffic.

The default tile is 1,024 positions. An explicit 8 MiB scratch estimate limits the arrays managed by the implementation; it excludes vendor BLAS internal workspace. Shared prefix arrays are owned immutable copies. Suffixes use bounded capacity with per-request lengths; append validation must finish before changing any request, so a failed append leaves the batch unchanged.

The practical alternatives are a dense fused CPU kernel, ungrouped state decomposition, a larger untiled grouped computation, and a compiled fused CPU kernel. Dense SDPA is a necessary baseline because fusion can outperform decomposition even when the latter stores less KV data. A compiled kernel would be a separate implementation effort and must earn its complexity through measurement.

## Interpretation and non-goals

The verification target is agreement with an independently computed float64 dense reference within the declared float32 tolerances, including random inputs and adversarial numerical cases. Near-zero reference values require an absolute tolerance; relative error alone is misleading. Nonfinite inputs and score overflow require explicit rejection or a tested numerical policy.

An operator speedup does not establish faster complete language-model generation. Tokenization, projections, feed-forward blocks, model weights, sampling, and KV production can dominate total latency. This version does not implement automatic prefix detection, eviction, paged or hierarchical caches, arbitrary attention masks, dropout, gradients, GQA, distributed serving, CUDA, or model-quality evaluation. Future research can test a workload-based dispatch rule or a compiled fused implementation, but neither is a result of this release.
