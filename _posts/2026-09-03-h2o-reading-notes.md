---
layout: post
title: 'Reading Notes of H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models'
date: 2026-09-03 22:15:24
description: 'Given current token, use the query to compute attention with keys in previous tokens. Evict n tokens with least attention scores. Any token that is not relevant to current position is removed (but never recovered). Information loss will hurt the model if the evicted tokens should be reused in the future. KV cache eviction based on heuristics. Never recovered when being evicted. It''s not very reasonable but it paves the way of KV cache eviction.'
tags: nlp kv-cache
categories: ReadingNotes
thumbnail: assets/img/blogs/h2o.png
toc:
  sidebar: left
related_posts: false
---

Given current token, use the query to compute attention with keys in previous tokens. Evict n tokens with least attention scores. Any token that is not relevant to current position is removed (but never recovered). Information loss will hurt the model if the evicted tokens should be reused in the future. KV cache eviction based on heuristics. Never recovered when being evicted. It's not very reasonable but it paves the way of KV cache eviction.

<!--more-->

> **Reference**: https://arxiv.org/pdf/2306.14048

## Motivation of KV Cache Eviction

To avoid long context degradation:

![image.png](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/imgimg20260903215836.png)

An ideal KV cache policy must satisfy simultaneously:

1. **Small cache size** — actual memory reduction.
2. **Low miss rate** — the retained set must contain whatever the model actually needs, so quality is preserved.
3. **Low-cost eviction policy** — the decision must be cheap enough to run every decoding step, or you trade memory for wall-clock time and gain nothing.

## Prerequisites

The formal setup is worth reading slowly, because the notation for "the evicted entries" is the one subtle piece of the paper.

**Notation.** $$Q \in \mathbb{R}^{n\times d}$$ is the query matrix, $$K \in \mathbb{R}^{n\times d}$$ the key matrix. $$Q_{i,*}$$ is the $$i$$-th row (the query for token $$i$$). $$K_{\le i,*}$$ is the first $$i$$ rows of $$K$$. Let $$k < n$$ be the **cache budget**.

For a subset $$S_i \subseteq [i]$$, the matrix $$K_{S_i,*} \in \mathbb{R}^{i \times d}$$ is defined as: keep row $$j$$ of $$K$$ if $$j \in S_i$$, **and put all zeros in that row otherwise**. Note it is still $$i \times d$$ — the rows aren't deleted, they're zeroed. This matters below.

### Eviction policy

An eviction policy is a map $$g: S_{i-1} \to S_i$$ satisfying

- $$\vert S_i\vert  = k$$ — the cache size never changes over time;
- $$\vert S_i \setminus S_{i-1}\vert  \le 1$$, equivalently $$\vert S_i \cap S_{i-1}\vert  \ge k-1$$ — at most one entry changes per step.

The second condition is the streaming constraint. At step $$i$$ exactly one new token arrives, so at most one new item can enter the cache, and therefore at most one old item can leave. You are not allowed to re-plan the whole cache each step.

### Generative process under an eviction policy

For token $$i$$, with $$S_i \subset [n]$$ the cached token set:

$$
o_i := D_i^{-1}\cdot \exp\!\big(Q_{i,*}(K_{S_i,*})^{\top}\big), \qquad D_i := \Big(\exp\big(Q_{i,*}(K_{S_i,*})^{\top}\big) - \mathbf{1}_{[i]\setminus S_i}\Big)\cdot \mathbf{1}_i
$$

**Why the subtraction is there.** Since the non-selected rows of $$K_{S_i,*}$$ are zero, we get $$Q_{i,*}\cdot 0 = 0$$ in those coordinates, and $$\exp(0) = 1$$ — not $$0$$. So the raw exponential vector contains a spurious $$1$$ at every evicted position. The indicator vector $$\mathbf{1}_{[i]\setminus S_i}$$ (which is $$1$$ at evicted positions, $$0$$ elsewhere) subtracts exactly those spurious ones off before the normalizer is computed.

**Concrete example.** Say $$i = 4$$, budget $$k=3$$, and the policy kept $$S_4 = \{1,2,4\}$$, evicting token 3. Write $$s_j = Q_{4,*}K_{j,*}^\top$$. Then

$$
\exp(Q_{4,*}(K_{S_4,*})^\top) = [\,e^{s_1},\; e^{s_2},\; \underbrace{e^{0}=1}_{\text{spurious}},\; e^{s_4}\,]
$$

$$
-\ \mathbf{1}_{[4]\setminus S_4} = [\,0,0,1,0\,] \;\Longrightarrow\; [\,e^{s_1},\, e^{s_2},\, 0,\, e^{s_4}\,]
$$

$$
D_4 = e^{s_1} + e^{s_2} + e^{s_4}, \qquad o_4 = \tfrac{1}{D_4}[\,e^{s_1},\, e^{s_2},\, 1,\, e^{s_4}\,]
$$

The point: **the softmax is renormalized over the surviving tokens only.** It is not a masked-and-rescaled version of the true attention; the true probability mass that sat on token 3 is redistributed over tokens 1, 2, 4. This is why eviction is not a benign approximation — it changes the distribution, and once done it can never be undone. Replacing $$S_i$$ by $$[i]$$ everywhere recovers standard, exact attention.

**The Goal.** Find an eviction policy such that the output of this constrained generative process is comparable to the unconstrained one.

## The H₂O algorithm

### The score function and the eviction rule

**H₂O Eviction Policy.** Let $$F_{\text{score}}: 2^{[n]} \to \mathbb{R}$$ be a score function. The policy $$g: S_{i-1}\to S_i$$ satisfies the two constraints above, plus the construction rule:

$$
S_i \leftarrow (S_{i-1}\cup\{i\})\setminus\{u\}, \qquad u \leftarrow \arg\max_{v \in (S_{i-1}\cup\{i\})} F_{\text{score}}\big(S_{i-1}\cup\{i\}\setminus\{v\}\big)
$$

In words: form the candidate set of $$k+1$$ items (current cache plus the new token), then **drop whichever single item leaves behind the highest-scoring remaining set**.

**The instantiation.** $$F_{\text{score}}(T) := \sum_{s\in T} o_s$$, the sum of accumulated attention scores over the set.

**A simplification worth making explicit.** Because this $$F_{\text{score}}$$ is *additive*, for the candidate set $$G = S_{i-1}\cup\{i\}$$ we have $$F_{\text{score}}(G\setminus\{v\}) = F_{\text{score}}(G) - \tilde o_v$$. So

$$
\arg\max_{v} F_{\text{score}}(G\setminus\{v\}) \;=\; \arg\min_{v} \tilde o_v
$$

**The rule is simply: evict the token with the smallest accumulated attention score.** The $$\arg\max$$-over-complements phrasing exists so that the definition generalizes to non-additive $$F_{\text{score}}$$, which the theory needs.

### The algorithm, line by line

```
procedure H2_EVICTION(Q, K ∈ R^{n×d}, k ∈ N)
 1:  S₀ ← ∅
 2:  for i = 1 → n do
 3:      if i ≤ k then
 4:          S_i ← S_{i-1} ∪ {i}                       # cache not full: just add
 5:      else
 6:          D_i ← (exp(Q_{i,*}(K_{S_{i-1},*})ᵀ) − 1_{[i]\S_{i-1}}) · 1_i
 7:          o_i ← D_i^{-1} · (exp(Q_{i,*}(K_{S_{i-1},*})ᵀ) − 1_{[i]\S_{i-1}})
 8:          F_score(T) := Σ_{s∈T} o_s                  # accumulated scores
 9:          G_i ← S_{i-1} ∪ {i}
10:          u ← argmax_{v ∈ G_i} F_score(S_{i-1} ∪ {i} \ {v})
11:          S_i ← (S_{i-1} ∪ {i}) \ {u}
12:      end if
13:  end for
```

Lines 6–7: compute this step's attention over the *currently cached* keys, with the correction term for evicted slots. Line 8: the score of a set is the sum of accumulated per-token attention. Lines 9–11: the swap.

The formal version in the appendix makes two things explicit that the pseudocode above hides:

- The accumulation is maintained incrementally: $$\tilde o_i \leftarrow \tilde o_{i-1} + o_i$$. Only one length-$$n$$ vector is needed, reducing the naive $$n^2$$ bookkeeping to $$n$$.
- $$F_{\text{score}}(T) := h\!\left(\sum_{s\in T}\tilde o_{i,s}\right)$$ for some function $$h:\mathbb{R}\to\mathbb{R}$$. The pseudocode uses $$h(z) = z$$ for clarity; the theory wants $$h$$ non-decreasing and **concave**, e.g. $$h(z)=\sqrt{z+1}$$ or $$h(z)=\log(z+1)$$, since composing a concave function with a modular one is what actually yields diminishing returns.

### Worked example

Sentence being processed: *"Children laughed and played in the sunny park."* Cache budget $$k = 3$$.

**Decoding step 4.** Four tokens exist (*Children, laughed, and, played*), one over budget. The running column sums of the attention matrix are approximately

$$
\tilde o = [\,1.4,\;\;1.5,\;\;0.5,\;\;0.6\,]
$$

for *Children, laughed, and, played*. The minimum is $$0.5$$ at "and". Equivalently: dropping "and" leaves a remaining total of $$3.5$$, which is higher than dropping any other candidate. So **the KV embedding of the third token is evicted**, and $$S_4 = \{1,2,4\}$$.

**Decoding step 5.** Token *"in"* arrives with attention row $$[0.03,\ 0.04,\ -,\ 0.02,\ 0.9]$$ over the surviving keys plus itself. Note "and" contributes nothing — it is permanently unreachable. Accumulate:

$$
\tilde o = [\,1.6,\;\;1.8,\;\;0.62,\;\;0.9\,] \quad \text{for } \textit{Children, laughed, played, in}
$$

Again the minimum among the four candidates is evicted, restoring $$\vert S_5\vert  = 3$$.

The paper's figure also shows a third panel — eviction using the *global* statistic, where the full attention matrix including future rows would be available. It is drawn in red and labeled **infeasible**, because it requires attention from tokens not yet generated. The third observation below is the claim that the cheap local version loses essentially nothing.

### The practical deployment: half heavy hitters, half recent

The algorithm as stated would keep only heavy hitters. In the actual system, **the budget is split evenly**: for a parameter $$K$$, the implementation maintains a KV cache whose first $$K$$ entries are heavy hitters and whose last $$K$$ entries are the most recent tokens. So "H₂O with 20% budget" means 10% heavy hitters plus 10% local window.

The recent window is managed as a **circular queue** — at iteration $$i$$, the oldest of the last $$K$$ tokens is at index $$((i-1) \bmod K) - K$$, and it is overwritten by the newcomer. Heavy-hitter eviction happens **independently per attention head** (each head has its own accumulated scores and its own argmin), which the paper emphasizes as a difference from SpAtten.

**Memory-layout detail that matters for speed:** when an entry is evicted, memory is *never* moved or swapped. The new KV is written directly into the vacated slot. The entire KV cache is preallocated. This is what keeps the eviction policy's wall-clock cost near zero and preserves I/O efficiency.

## Why?

Three Observations

1. Attention is extremely sparse, so a small cache *can* suffice.
2. Accumulated attention scores follow a power law: Heavy Hitters exist
3. Local statistics are as good as global ones, so the policy can be cheap

Theory: dynamic submodular maximization

## Experiments

**Setup.** OPT (6.7B–66B), LLaMA (7B–30B), GPT-NeoX-20B. Accuracy on lm-eval-harness (COPA, MathQA, OpenBookQA, PiQA, RTE, Winogrande, 5-shot) and HELM (XSUM, CNN/Daily Mail). Budgets 4%–100% of prompt length, split 50/50 between H₂ and recent. Baselines: Full KV, **Local** (recent only), Sparse Transformer (strided/fixed), Top-K, StreamingLLM, SpAtten. Systems built on FlexGen; compared against FlexGen / DeepSpeed / HF Accelerate on T4 and A100. All speedups end-to-end, including prefill and cache construction.

### Key takeaways

**1. 20% budget ≈ full cache; Local collapses.**

| Method (20% budget) | PiQA | COPA | OpenBookQA | Winogrande |
|---|---|---|---|---|
| Full | 80.09 | 81.00 | 44.80 | 71.51 |
| Local | 57.94 | 56.00 | 28.40 | 51.30 |
| **H₂O** | **79.22** | **85.00** | **43.80** | **71.67** |

5× memory reduction at parity. Gap is largest on long-generation tasks (XSUM, CNN/DM), where Local already collapses at 60% budget while H₂O holds at 20%.

**2. H₂ is a complement, not a competitor — this is the most robust result.** Every static sparsity pattern fails at 20% budget and is rescued by adding heavy hitters (OPT-30B): Local 48.00→84.00 COPA, strided 50.00→83.00, fixed 61.00→76.00, all approaching Full's 85.00. Same holds for Top-K (+up to 2%).

**3. Both components are necessary.** H₂-only or Local-only each lose 2.85%–22.75%; together they recover the baseline. H₂-only consistently beats Local-only, so heavy hitters carry more of the weight.

**4. Throughput: up to 3×/29×/29× over FlexGen/DeepSpeed/Accelerate.** But note the mechanism — freed KV memory enlarges the effective batch (e.g. 80→416) or removes the need for CPU offloading. The clean apples-to-apples number, same system and same batch size, is **1.1–1.9× lower latency**.

**5. Composes with quantization and streams to 4M tokens.** H₂O + 4-bit is as good as or better than either alone (78.80 vs 78.51 PiQA). With position rolling, H₂O streams to four million tokens at lower perplexity than StreamingLLM, and wins clearly on multi-document QA where the answer sits mid-context — precisely where StreamingLLM's fixed "first-few + local" pattern discards the key document.

**6. Negative result the authors report.** Accumulated attention score is **biased toward early tokens** (older tokens appear in more attention rows, so their sums are mechanically larger). The obvious fix — averaging instead of summing — *degraded* performance. They also observed many H₂ at sentence beginnings, i.e. they rediscovered attention sinks without naming them.

## AI Assessment That I Agree

**1. Eviction is irreversible, and the decision is made with past-only information.** This is the core conceptual problem, and it's the reason the method is "not very reasonable" despite working. Once a KV is dropped it is unreachable forever — there is no miss to recover from, unlike a real cache. Worse, importance is scored greedily from accumulated attention *so far*: a token that looks dead at step 500 may be exactly what step 5000 needs. So the method is fundamentally betting that past attention predicts future attention, which is a query-independent assumption about a query-dependent quantity. The paper's own multi-document QA experiment is the regime where this bet fails, and it is tested only lightly in an appendix. Nothing degrades gracefully once the retained set is wrong. Everything after H₂O (query-aware selection, offload-and-retrieve, recoverable caches) is a response to this.

**2. The theory is decorative rather than load-bearing.** Submodularity of attention is *assumed*, never proven. The $$h$$ that would make $$F_{\text{score}}$$ genuinely submodular ($$\sqrt{\cdot}$$, $$\log$$) is not the one deployed — the shipped algorithm uses $$h(z)=z$$, which is **modular**, and for a modular objective under a cardinality constraint greedy is *exactly* optimal, making the $$(1-1/e)$$ factor vacuous. Most importantly, the bound is on a proxy score $$f$$, not on anything anyone cares about: there is no guarantee of the form $$\Vert o_i^{\text{H}_2\text{O}} - o_i^{\text{full}}\Vert  \le \varepsilon$$, and no analysis of how per-step attention error compounds over the autoregressive chain. The drift conditions ($$\theta=\gamma=\alpha/(10n)$$) are never measured. Read the theory as motivation for the design, not as justification.

**3. It doesn't actually fit the systems stack it claims to speed up.** H₂O needs **column sums of the attention probability matrix**, which FlashAttention deliberately never materializes — so the statistic-collection step is at odds with the standard kernel, especially during prefill. Relatedly, peak memory during prefill is *unchanged*: the budget is a fraction of prompt length, and heavy hitters are chosen only after full attention over the prompt has been computed, so the savings are decode-phase only. Add that eviction is decided **per head**, which sits awkwardly with the grouped-/multi-query attention that every modern model uses (and which already compresses the cache 4–8×), and the practical applicability to a 2024+ serving stack is much narrower than the headline numbers suggest.
