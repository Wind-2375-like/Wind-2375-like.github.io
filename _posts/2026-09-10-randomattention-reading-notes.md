---
layout: post
title: 'Reading Notes of Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning'
date: 2026-09-10 23:30:00
description: 'TL;DR: random KV eviction works better than heuristics ones.'
tags: nlp kv-cache
categories: ReadingNotes
thumbnail: assets/img/blogs/randomattention.png
toc:
  sidebar: left
related_posts: false
---

TL;DR: random KV eviction works better than heuristics ones.

<!--more-->

> **Reference**: https://arxiv.org/pdf/2609.03430v1

## 1. Summary

Reasoning LLMs generate chains of thought tens of thousands of tokens long, so the KV cache (the model's memory of every past token) becomes the serving bottleneck. Eviction methods cap it by scoring each cached token's future importance and dropping low scorers. This paper asks whether the score matters at all. **Random Attention** keeps the prompt and drops the rest uniformly at random, independently per attention head. On four reasoning models (Qwen3-4B/14B/32B, Phi-4-reasoning) and six math/science/code benchmarks at ~4× compression it ties the best scored evictor, serves 32–43% more tokens/s in vLLM because it never scores, and controlled experiments explain why: the prompt is fragile, the trace is redundant.

---

## 2. Motivation

Long reasoning traces make KV memory, not compute, the limit on how many requests a GPU can serve, and eviction is the only compression that actually bounds memory. Prior evictors (H2O, SnapKV, R-KV, VaSE, TriAttention) form a lineage of ever-better importance scores, each justified by heuristics correlated with accuracy, each costing an extra pass over the cache. None was tested against a score-free control at matched budget *and* matched prompt protection; the "random" baselines in earlier papers lost badly because they also dropped the prompt. Random Attention supplies that missing null hypothesis. The motivation is clear; what is not argued is why *randomness* specifically, rather than a plain prompt-plus-recency window, should be the default (see §5).

---

## 3. Math

### 3.1 Problem setup

Fix one (layer, KV head) and a decode step $$t$$. The cache holds $$N$$ key–value pairs $$(k_i, v_i)$$, $$k_i, v_i \in \mathbb{R}^d$$, one per past position $$i = 1,\dots,N$$. Positions $$1,\dots,\ell_p$$ are the **prompt** (system prompt + chat template + question); positions $$i > \ell_p$$ are the model's own generated **trace**. The head's output is ordinary attention:

$$
o_t = \sum_{i=1}^{N} \alpha^{(t)}_i\, v_i, \qquad
\alpha^{(t)}_i = \frac{\exp\!\big(q_t^\top k_i / \sqrt{d}\big)}{\sum_{j} \exp\!\big(q_t^\top k_j / \sqrt{d}\big)}.
$$

**Periodic eviction (shared by every method).** Choose a budget $$K$$ and an unscored recent buffer of size $$r \ll K$$ ($$r = 64$$ throughout). Every $$r$$ decode steps the buffer is full and an eviction fires: let $$\mathcal{C}_t$$ be all cached positions except the $$r$$ newest; every $$i \in \mathcal{C}_t$$ receives a real score $$s_i$$ and the head keeps

$$
\mathcal{S}_t = \operatorname{top\text{-}K}_{\,i \in \mathcal{C}_t}\; s_i ,
$$

so the cache returns to $$K + r$$ entries. Evicted pairs are gone permanently, and the decision is made independently in every layer and every KV head. **Methods differ only in $$s_i$$.**

**Baselines as choices of $$s_i$$** (all evaluated at the eviction step $$t$$):

| Method | Score $$s_i$$ | What it must read |
|---|---|---|
| StreamingLLM | no score: keep a few "sink" tokens + recent buffer | nothing |
| H2O | $$\sum_{t'=i}^{t} \alpha^{(t')}_i$$ (attention accumulated since entry) | attention weights |
| SnapKV | $$\sum_{t'=t-w+1}^{t} \alpha^{(t')}_i$$ (attention from last $$w$$ queries, max-pooled over neighbours) | attention weights |
| R-KV | $$\lambda\, s^{\text{Snap}}_i + (1-\lambda)(1-\bar c_i)$$, $$\;\bar c_i = \tfrac{1}{N-1}\sum_{j\neq i}\cos(k_i,k_j)$$ (penalise restated content) | attention + key similarity |
| VaSE | keep the $$n_{\text{large}}$$ positions with largest value range $$\max_j v_{ij} - \min_j v_{ij}$$; fill the rest by sampling $$\propto s^{\text{Snap}}$$ | values + attention |
| TriAttention | $$f(t-i)$$: per-head calibrated trigonometric function of distance, combined with $$\lVert k_i \rVert$$ | calibration data + key norms |
| Recency+prompt | keep prompt + the $$K-\ell_p$$ most recent positions | nothing |

### 3.2 Random Attention (the method)

$$
s_i =
\begin{cases}
+\infty, & i \le \ell_p \quad (\text{prompt: never evicted})\\[4pt]
u_i \sim \mathrm{Uniform}(0,1), & i > \ell_p \quad (\text{trace: i.i.d. random, per KV head, redrawn every eviction})
\end{cases}
$$

Plugged into the top-$$K$$ rule above, that is the whole method: one `rand`, one `topk`, then compaction. No attention weights, no value statistics, no calibration, no tuning.

**Implied age profile (not age-blind).** A trace position survives one eviction with probability

$$
p = \frac{K-\ell_p}{K + r - \ell_p} \approx 0.94 \quad (K{=}1024,\ r{=}64),
$$

so after $$n$$ evictions it remains in a given head with probability $$p^{n}$$: a *soft recency window* with a thin random tail of old tokens. Because heads draw independently, a token is lost from **all** $$H_{kv}$$ heads only with probability $$(1-p^{n})^{H_{kv}}$$ (my derivation; the paper states only $$p^{n}$$).

### 3.3 Evaluation metrics

- **Accuracy.** For a task with problem set $$\mathcal{D}$$ and $$R$$ independently sampled runs,

  $$
  \mathrm{Acc} = \frac{1}{R\,\vert \mathcal{D}\vert }\sum_{\text{run}}\sum_{x\in\mathcal{D}} \mathbb{1}[\text{final boxed answer of } x \text{ is correct}],
  $$

  LiveCodeBench uses pass@1 by executing hidden tests. *Compression factor* $$\approx$$ (mean full-attention trace length)$$/K$$.
- **Significance.** Paired, problem-clustered percentile bootstrap (95% CI) plus an exact sign test; cells significantly below Random Attention are grayed in the paper's tables.
- **Throughput.** Output tokens/s at a fixed offered load, also reported as a multiple of full attention.
- **Planted-fact probe (§4, Part B).** *Retrieval* = fraction of traces whose greedy decode reproduces the planted value. Graded *recall*

  $$
  R = \frac{\sum_i \big(\mathrm{LP}_i - \mathrm{LP}^{\text{del}}_i\big)}{\sum_i \big(\mathrm{LP}^{\text{kept}}_i - \mathrm{LP}^{\text{del}}_i\big)},
  $$

  where $$\mathrm{LP}_i$$ is the log-probability of the correct value in trace $$i$$ under the tested condition, and $$\mathrm{LP}^{\text{kept}}_i$$, $$\mathrm{LP}^{\text{del}}_i$$ are the same with the fact kept in / deleted from every head. $$R=1$$: as good as never evicting; $$R=0$$: as good as deleting.
- **Passcode test.** *Retr.* (fraction of traces reproducing a once-stated passcode) and mean $$\log p$$ of the correct passcode ($$0$$ = certain, $$\approx -18$$ = effectively gone).

---

## 4. Results and analysis

**Organization.** The experiments answer three questions in order, each building on the last. **Part A — does dropping the score cost accuracy?** The main grid at ~4× compression (Tables 1, 5) plus a 2×→16× sweep (Fig. 2). **Part B — why not?** Three controlled experiments that separate the method's two ingredients: prompt protection (Tables 2, 6), per-head random scatter (planted-fact probe, Fig. 3, App. D), and the one case where a score still helps (Table 3). **Part C — what does skipping the score buy?** vLLM serving throughput (Table 4), per-round cost (Table 9), equal-memory batch (Tables 10–11).

**General setup.** Models: Qwen3-4B, Qwen3-14B, Qwen3-32B, Phi-4-reasoning (14B). Tasks: MATH500 (500 problems), GPQA-Diamond (198), AIME 2025+2026 (60, pooled), HMMT (60), LiveCodeBench-v6 medium (383, graded by test execution). Per-head budget $$K$$ = 1024 / 2048 / 4096 / 4096 / 3072 respectively, i.e. ~4× compression of each task's mean full-attention trace (~3× for LCB); buffer $$r=64$$; generation cap 32,768 tokens; released sampling settings (temperature 0.6 for Qwen3, 0.8 for Phi-4; top-$$p$$ 0.95); independent runs 2 / 4 / 16 / 16 / 4, accuracies averaged over runs. Baselines: SnapKV, R-KV ($$\lambda=0.5$$), VaSE ($$n_{\text{large}}=K/4$$), TriAttention (official per-head variant with per-model calibration), all re-implemented in one HuggingFace + FlashAttention-2 engine with per-KV-head physical eviction; full attention (no eviction) is the ceiling. Hardware: NVIDIA H200.

### Part A — Accuracy under eviction

*Setup:* the general setup above. Table 1 covers Qwen3-4B, Phi-4-reasoning, Qwen3-32B; Table 5 adds Qwen3-14B. Fig. 2 varies $$K$$ so compression runs 2×→16× on the four math/science tasks for Qwen3-4B and Phi-4 (LCB is omitted because its prompts alone do not fit the small budgets).

Five-task mean accuracy at ~4× compression (my averages of Tables 1 and 5; bold = best evictor):

| Model | Full (no eviction) | SnapKV | R-KV | VaSE | TriAttention | Random Attention |
|---|---|---|---|---|---|---|
| Qwen3-4B | 0.682 | 0.478 | 0.574 | 0.597 | 0.636 | **0.639** |
| Phi-4-reasoning | 0.689 | 0.489 | 0.650 | 0.532 | 0.658 | **0.669** |
| Qwen3-14B | 0.743 | 0.573 | 0.640 | 0.674 | **0.701** | 0.693 |
| Qwen3-32B | 0.763 | 0.578 | 0.672 | 0.693 | **0.718** | 0.711 |

LiveCodeBench pass@1 (the one task with large gaps):

| Model | Full | SnapKV | R-KV | VaSE | TriAttention | Random Attention |
|---|---|---|---|---|---|---|
| Qwen3-4B | 0.807 | 0.507 | 0.712 | 0.700 | **0.755** | 0.744 |
| Phi-4-reasoning | 0.697 | 0.314 | 0.621 | 0.373 | 0.652 | **0.667** |
| Qwen3-14B | 0.856 | 0.622 | 0.788 | 0.813 | **0.843** | 0.820 |
| Qwen3-32B | 0.886 | 0.609 | 0.779 | 0.797 | **0.834** | 0.806 |

- **Takeaway A1 — a score-free evictor ties the best scored one.** In Table 1's 60 baseline cells, Random Attention is significantly ahead in 31 and significantly behind in 1 (TriAttention on Qwen3-32B code, −2.8 points). On MATH500 and GPQA-D it beats SnapKV and VaSE significantly on every model and no selector beats it; on AIME/HMMT nominal leads run both ways inside ±5-point run-to-run noise. The price of eviction itself is real: ~4–7 points below full attention on the five-task mean.
- **Takeaway A2 — the one systematic gap is the prompt, not the score.** LCB prompts average 557 tokens (6× MATH500) and can consume half of $$K=3072$$. Selectors that leave the prompt to their score fall apart: SnapKV loses 20–35 points, VaSE collapses on Phi-4 (−29), R-KV trails by 3–5 everywhere. TriAttention, the only baseline that also pins the prompt, is the only one on par with or slightly above Random Attention. This is what motivates Part B.
- **Takeaway A3 — tighter budgets widen the margin over scored methods.** At 2× every evictor sits near full attention; by 16× Random Attention and TriAttention stay within a few points of each other (e.g. 0.65–0.76 on Qwen3-4B MATH500) while VaSE drops to 0.39 (MATH500) and 0.12 (GPQA-D). Caveat: at 14B three cells go significantly *against* Random Attention (TriAttention +2.6 on LCB and +2.1 on MATH500; VaSE +2.6 on AIME), so "matches the strongest baseline" means a statistical tie with a slight TriAttention edge at 14B/32B, not dominance.

### Part B — Why the score buys so little

*Setup:* (i) **Matched protection.** Re-run SnapKV, R-KV and VaSE with the prompt force-kept, and Random Attention / a plain recency window with the prompt rule *removed* (Table 2: Qwen3-4B and Phi-4 on MATH500, GPQA-D; Table 6: code, competition math, GPQA-D at 32B). Keep-logs record, round by round, what fraction of the prompt each policy retains (App. D). (ii) **Planted-fact probe** on Qwen3-4B (8 KV heads): a synthetic fact ("Let zq = 4729") is inserted into real MATH500 traces 1,536 tokens (15 evictions) before a question that needs it; the experimenter controls which heads keep the fact; 250–500 traces per condition. (iii) **Passcode test:** a passcode stated once, 57 eviction rounds before the question; every policy retains as it likes.

Table 2 — accuracy with the score alone → with the prompt also protected (Δ in points), and how much of the prompt each score keeps per head (App. D):

| Method | Qwen3-4B MATH500 | Qwen3-4B GPQA-D | Phi-4 MATH500 | Phi-4 GPQA-D | prompt survival / head |
|---|---|---|---|---|---|
| SnapKV | 0.703 → 0.829 (+12.6) | 0.369 → 0.492 (+12.3) | 0.844 → 0.889 (+4.5) | 0.442 → 0.667 (+22.5) | 11–22% |
| VaSE | 0.809 → 0.812 (+0.3) | 0.461 → 0.470 (+0.9) | 0.853 → 0.895 (+4.2) | 0.562 → 0.664 (+10.2) | 20–29% |
| R-KV | 0.810 → 0.812 (+0.2) | 0.482 → 0.471 (−1.1) | 0.909 → 0.902 (−0.7) | 0.636 → 0.655 (+1.9) | 26–67% |
| Recency window | 0.246 → 0.843 (+59.7) | 0.093 → 0.519 (+42.6) | 0.665 → 0.884 (+21.9) | 0.323 → 0.658 (+33.5) | — |
| Random Attention | 0.459 → **0.874** (+41.5) | 0.231 → **0.530** (+29.9) | 0.759 → **0.910** (+15.1) | 0.434 → **0.678** (+24.4) | ≥99% with rule |

Fig. 3a — planted fact, retrieval accuracy by which heads keep it:

| Heads holding the fact | h5 alone | h2 alone | {2,3} | {3,5} | {2,5} | {2,3,5} | all 8 |
|---|---|---|---|---|---|---|---|
| Retrieval | 0.03 | 0.01 | 0.16 | 0.39 | 0.60 | 0.83 | 0.99 |

Table 3 — once-stated passcode, 57 rounds back:

| Policy | Retr. | mean $$\log p$$ |
|---|---|---|
| Random Attention | 0.000 | −18.35 |
| SnapKV | 0.004 | −11.11 |
| TriAttention | 0.016 | −11.11 |
| VaSE | 0.344 | −3.88 |
| R-KV | **0.836** | **−0.71** |

- **Takeaway B1 — the prompt is the fragile part of the cache.** Each baseline gains from the protection rule exactly in proportion to how much prompt its score was losing: SnapKV (worst retainer) +4.5 to +22.5, VaSE only where its retention fails, R-KV (best retainer) ≤ +1.9. With the rule, the three baselines land within 2.2 points of one another and at or below Random Attention in every setting (a residual 4–6-point deficit remains on Qwen3-4B and GPQA-D at 32B). Losing the prompt is catastrophic (recency alone: 0.09; Random Attention without the rule: 0.23–0.76); randomly cutting the trace is not. This also explains why earlier papers' random baselines looked terrible: they dropped the question.
- **Takeaway B2 — the trace protects itself, at two levels.** *Text:* the model restates what it is still using, so a needed value rarely lives at one position. *Heads:* each KV head keeps its own copy and per-head draws lose a token only when every head drops it. A fact held in one head is retrieved 1–3% of the time, in two heads 16–60%, three 83%, all eight 99% (strongly super-additive), and shape is irrelevant (dealt token-by-token across heads: 0.33 vs 0.39 retrieval for an intact sentence; contiguous blocks up to 64 tokens cost nothing on real MATH500). On real traces a *shared* draw (same random set in every head) scores 0.871 vs 0.874 at $$K=1024$$ and 0.788 vs 0.789 at $$K=512$$, i.e. text redundancy alone suffices; cross-head redundancy is the backstop for facts that are never restated.
- **Takeaway B3 — what a score can still buy: a fact stated once, far back.** Random Attention never recovers the passcode. Recovery tracks the attention statistic each score uses: R-KV (attention accumulated over the whole history) 84%, VaSE (sampled attention) 34%, recency-window scores (SnapKV, TriAttention) ≈ 0. This needle skill is uncorrelated with aggregate accuracy (R-KV, best here, is mid-pack in Table 1; TriAttention, best there, fails here) and is rarely exercised on real traces.

### Part C — Efficiency

*Setup:* (i) **Paged serving:** vLLM v0.19.0 + PagedAttention on one H200, bf16, $$K=2048$$, 1k-token prompts, 32k-token generations, 128 concurrent requests (96 on 32B), compression every 64 tokens. Random Attention is added as a one-function selector inside TriAttention's released vLLM plugin, so kernels, paging, scheduler and trigger are identical. Only TriAttention is compared here because only it and R-KV have vLLM ports and TriAttention was already shown faster than R-KV. (ii) **Equal memory:** HuggingFace + FlashAttention-2 without paging, $$K=3072$$, 32k generations, each method at the largest batch that fits 143 GB (TriAttention here is an unfused PyTorch port, so its row understates the method). (iii) **Per-round cost:** single-stream CUDA-event timing of one eviction round at $$K=1024$$.

Table 4 — vLLM throughput (output tok/s, × full attention):

| Model | Full | TriAttention | Random Attention | RA over TriAttention |
|---|---|---|---|---|
| Qwen3-4B | 1296 (1.00×) | 1494 (1.15×) | **2046 (1.58×)** | +37% |
| Phi-4-reasoning | 780 (1.00×) | 1212 (1.55×) | **1737 (2.23×)** | +43% |
| Qwen3-14B | 925 (1.00×) | 1303 (1.41×) | **1819 (1.97×)** | +40% |
| Qwen3-32B | 346 (1.00×) | 700 (2.02×) | **923 (2.67×)** | +32% |

Table 9 — cost of one eviction round (ms, Qwen3-4B / Qwen3-14B): Random Attention 0.30 / 0.29 · SnapKV 0.37 / 0.39 · R-KV 0.58 / 0.61 · VaSE 0.74 / 0.77 · TriAttention 1.47 / 1.64 (0.5% → 2.7% of decode time).

Table 10 — equal-memory serving, Qwen3-4B (batch, tok/s, × full): Full 28 / 178 / 1.0 · SnapKV 186 / 1624 / 9.1 · R-KV 186 / 1223 / 6.9 · VaSE 190 / 1617 / 9.1 · TriAttention* 182 / 670 / 3.8 · **Random Attention 200 / 1779 / 10.0**. Qwen3-14B: Full 20 / 164 → Random Attention 120 / 1436 / 8.8×.

- **Takeaway C1 — under paged serving Random Attention gives 1.6–2.7× full-attention throughput and 32–43% more than TriAttention** on identical kernels. The margin is robust to the operating point: +41%/+42% at capacity, +30–35% at 64 requests, +35–42% at short 8k generations.
- **Takeaway C2 — the margin is the scoring pass multiplied by synchronization, not kernel time.** A round costs 0.30 ms vs 1.47–1.64 ms (< 3% of single-stream decode; at one request the two methods are within 1%). But vLLM compresses some request at nearly every step (~62k times per workload) at a barrier where all 128 requests wait, and content-dependent scores need an extra pass over paged keys (or attention weights that fused kernels never materialise): ≈ 15 ms of whole-batch waiting per compression for TriAttention vs < 1 ms for Random Attention.
- **Takeaway C3 — at equal memory every evictor gains 6–10×, because the small cache admits a ~7× larger batch (28 → 186–200)**; Random Attention fits the largest batch at the smallest peak footprint and reaches 10.0× / 8.8× (28.8× at $$K=1024$$). Important caveat: compression pays only when the cache is the bottleneck. At 8k-token generations the workload is compute-bound and Random Attention serves just 0.52× / 0.70× / 0.96× of full attention on Qwen3-4B / 14B / 32B.

---

## 5. Three biggest limitations (AI assessment)

**L1 — The result is tied to a narrow regime, and the headline claim overreaches it.** Every benchmark has a prompt of a few hundred tokens followed by 5–20k tokens of verbose math-style reasoning that naturally restates its intermediates. In that regime "keep the prompt" is nearly free and "the trace is redundant" is automatically true. The paper's own data show where this breaks: on code (557-token prompts) the pinned prompt eats up to half the budget and Random Attention loses its only significant cell; on the passcode test it recovers 0% of a once-stated fact. Neither ingredient is tested where eviction is most needed in practice (agentic loops, tool outputs, retrieved documents, multi-turn chat), where the "prompt" is thousands of tokens of mixed importance that cannot be pinned whole, and restatement is not guaranteed. The paper defers this as "how to budget long prompts," but it means the abstract's "the selection signal contributes almost nothing" should be read as "in short-prompt math reasoning, what you protect matters far more than how you rank."

**L2 — The efficiency headline is partly an artifact of one integration, and Random Attention's own costs are underweighted.** The 32–43% margin over TriAttention does not come from kernel time (0.3 vs 1.5 ms per round; single-stream the two are within 1%); it comes from TriAttention's vLLM plugin scoring synchronously at a barrier where all 128 requests wait, and the authors concede an asynchronous scorer could shrink it (none is released). The equal-memory comparison uses the authors' own unfused TriAttention port (3.8× vs 10×), which they admit is not the method's real speed, and SnapKV, R-KV and VaSE are never measured in vLLM at all. Meanwhile the costs Random Attention does incur are treated lightly: 4–7 points below full attention on average and 8 points on 32B code; a net throughput *loss* at 8k generations (0.52× full attention on Qwen3-4B); and on Qwen3-32B code it generates 18.4k tokens on average against 10.3k for full attention and 12.8k for TriAttention (Table 7), extra compute that a fixed-32k-length throughput benchmark never charges. "Fastest evictor at equal accuracy" therefore holds for one released stack, at one operating point, with accuracy defined relative to other evictors rather than to the uncompressed model.

**L3 — The mechanism story is thinner than its framing, and the comparison design leaves the key question open.** The cross-head redundancy probe is run on one small model (Qwen3-4B, 8 KV heads) with synthetic facts, and the paper's own shared-draw control shows removing cross-head diversity costs nothing on real tasks, so the mechanism that gets a figure and a section is not what makes the method work; prompt pinning plus text-level restatement is. The natural baseline is then recency+prompt (concurrently published as Prefix Sliding), which Table 2 shows within ~2 points of the best baseline; yet the marginal value of the random tail over recency is never isolated with significance tests, never swept to 8×–16×, and never tested on code or competition math. On the baseline side, every comparison is the authors' re-implementation with author-chosen settings (R-KV at $$\lambda=0.5$$ rather than the recommended 0.1 because 0.1 scored lower; TriAttention with the chat template enabled contrary to its own harness; VaSE at $$n_{\text{large}}=K/4$$), and "matches the strongest evictor" glosses over four significant losses to TriAttention/VaSE at 14B and 32B, the larger models where trends matter most. A reader cannot tell whether a properly tuned score beats "prompt + recency + noise" by a couple of points, which is exactly the claim at stake.

---

## 6. How to reproduce

**Step 1 — Code and environment.** Everything is in <https://github.com/SalesforceAIResearch/Random-Attention> (Apache-2.0, 11 commits, 15 stars as of 12 Sep 2026). `bash setup.sh` builds a Python 3.10 venv (torch 2.4.0 cu121, flash-attn 2.7.3, transformers 5.0.0); `. env.sh` sets `RA_ROOT / RA_ENGINE / RA_DATA_DIR / RA_MODELS_DIR`. The engine lives in `kvcompress/engine/` (every method as an eviction mode in `cache_utils.py`; Random Attention is `random_pp`); the harness `kvcompress/harness/` is adapted from VaSE's repo, and the TriAttention scorer and vLLM benchmark build on TriAttention's repo (provenance in `THIRD_PARTY_NOTICES.md`). The paper's runs used 8× H200 (141 GB); plan on the same class of GPU, since 32k-token generations at batch are memory-heavy.

**Step 2 — Models and data.** Put HuggingFace checkpoints for `Qwen3-4B`, `Qwen3-14B`, `Qwen3-32B`, `phi-4-reasoning` under `$RA_MODELS_DIR/<name>`. Put each task as `$RA_DATA_DIR/<task>/test.jsonl` following `data/README.md`: MATH500 (Hendrycks/Lightman split), GPQA-Diamond, AIME 2025 and 2026 and HMMT from MathArena, and LiveCodeBench-v6 medium (the 383-problem difficulty subset is shipped in `data/`). Use the models' released sampling settings (T=0.6 Qwen3 / 0.8 Phi-4, top-p 0.95), a 32,768-token cap, and 2 / 4 / 16 / 16 / 4 independent runs per task.

**Step 3 — Accuracy cells, grading, statistics.** One cell = (model, task, method, budget): `scripts/run_cell.sh Qwen3-4B math random_pp` ($$K$$ defaults to the task's ~4× point: 1024 / 2048 / 4096 / 4096 / 3072; buffer $$r=64$$). Baseline modes: `attn` (SnapKV), `attn_rkv_l05` (R-KV, $$\lambda=0.5$$), `vase_faithful` ($$n_{\text{large}}=K/4$$), `triattn_ph_memofix` (TriAttention per-head with calibration stats), `dense` (full attention). Then `scripts/grade_cell.sh` (runs a positional integrity check first, then grades by final boxed answer or test execution), `kvcompress/eval/stats_paired.py` for the paired clustered bootstrap + sign test, and `gen_paper_tables.py` for the LaTeX tables. Matched-protection, shared-draw and recency+prompt variants are engine switches documented in the `cache_utils.py` header, not in the README.

**Step 4 — Efficiency.** HuggingFace protocols (fixed-batch, iso-memory max-batch found by bisection on 143 GB, eviction-round CUDA-event timing) are in `scripts/efficiency/`; never take tok/s from the sharded accuracy runs. The vLLM numbers come from `scripts/vllm_rp_bench/`: Random Attention as a selector inside TriAttention's vLLM 0.19 runtime, bf16, $$K=2048$$, 1k-token prompts, 32k outputs, 128 requests (cap 96 on 32B), CUDA graphs and prefix caching off, with the two tooling fixes (benchmark ignoring output length; a dedup guard disabling later compaction) shipped as runtime shims described in that folder's README/RUNBOOK.

**Step 5 — Mechanism studies.** `KEEPLOG=1` produces the retention logs of App. D; `FORCE_KEEP_RANGE` plus `kvcompress/analysis/fork_replay.py` / `fork_autopsy.py` drive the planted-fact head-pinning experiments; `kvcompress/synth/` holds the registered synthetic-retrieval protocol of App. B (23-token prefilled stub, scripted body fed through the true decode path, 16-token fact box, 4-digit values that are exactly four Qwen3 tokens, distances of 3 / 15 / 39 / 57 evictions, 4,000-replicate instance-clustered bootstrap).

**Missing or unverified parts.** I read the README and the paper, not the code, so these remain open: whether the per-instance logs and graded TSVs behind Tables 1–11 are actually shipped (the paper says statistics were recomputed from "released per-instance logs"); the exact per-model TriAttention calibration statistics and how they were produced; the precise chat templates, prompts and seeds; the matched-protection baselines and the shared-draw control are referenced only as engine switches; the Qwen3-14B protected grid was never run; the vLLM bug fixes are described as "bookkeeping changes" without diffs in the paper; R-KV, SnapKV and VaSE were never benchmarked in vLLM; and the AIME 2026 / HMMT 2026 sets must be fetched from MathArena yourself. Expect the full grid to be expensive: 4 models × 5 tasks × 6 methods × up to 16 runs of 32k-token generations.
