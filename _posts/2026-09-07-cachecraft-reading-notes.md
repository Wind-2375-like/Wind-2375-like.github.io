---
layout: post
title: 'Reading Notes of CacheCraft: Discovering KV Cache Eviction Policies via LLM-Guided Program Evolution'
date: 2026-09-07 23:30:00
description: 'AI generated stuff. Does not worth reading. No training. LLM search based on heuristics.'
tags: nlp kv-cache
categories: ReadingNotes
thumbnail: assets/img/blogs/cachecraft.png
toc:
  sidebar: left
related_posts: false
---

AI generated stuff. Does not worth reading. No training. LLM search based on heuristics.

<!--more-->

> **Reference**: https://arxiv.org/abs/2608.14555

## 1. Summary

### The problem in one paragraph
When an LLM reads a long prompt, it stores a "key" and a "value" vector for every token in every layer — the **KV cache**. For long prompts this cache can be as large as the model weights themselves, and it has to be re-read from GPU memory at every generated token. One fix is **prefill-stage eviction**: right after the prompt is processed and *before* the first output token, throw away the least important cached tokens (e.g. keep only 12%). The hard part is deciding *which* tokens are important. Today this is done with hand-written heuristics (e.g. "keep tokens that recent queries attended to"), and each heuristic works well on some models/compression levels and badly on others.

### The method in one paragraph
Instead of hand-designing yet another heuristic, the authors treat the eviction rule as a **small Python program** and let an LLM **evolve** it. The system, **CacheCraft**, exposes a compact program with two functions (`score_tokens`, `select_tokens_to_keep`) that read 17 precomputed per-token features (attention statistics, key norms, positions, ...). An off-the-shelf engine (OpenEvolve, a FunSearch/AlphaEvolve-style tool) has a local LLM propose code edits, and a three-stage evaluator (syntax check → cheap benchmark gate → larger benchmark) scores each candidate on a long-context benchmark. Crucially, the authors run a **diagnostic loop**: whenever the search plateaus or "cheats" (reward hacking), they treat that as a sign that the *interface, seed program, or evaluator* is wrong and redesign it, rather than running more iterations.

The program that comes out is **FRC (Feature-Rich Compression)**: a fixed-weight sum of three attention signals — recent-window attention received (0.55), a moving average of attention over neighboring tokens (0.30), and the max over KV-heads of attention received (0.15) — followed by keeping the top-scoring **chunks** of 20 consecutive tokens.

### Three main takeaways (benchmark: RULER at 4k/8k context; models: Llama-3.1-8B-Instruct and Qwen3-8B)

The experiments are organized around one question — *does a single fixed rule beat hand-designed rules under aggressive compression, and if so, why?* — answered in three steps:

1. **It works where it matters (aggressive compression).** With the *same* weights on both models and both context lengths, FRC is the best compressed method in all 12 of the 20 (model × context × ratio) cells where ≥75% of the cache is dropped. At 88% compression the gains over the best baseline are +15.4 points (Llama, 4k), +5.6 (Qwen, 4k), +6.2 (Llama, 8k), +13.9 (Qwen, 8k). At mild compression (≤50%) all methods tie.
2. **The gain comes from the scoring formula, not the chunking.** Swapping the *scorer* changes RULER by +67 points; swapping chunk-level selection for token-level selection changes it by ≈0.1. So FRC is not "ChunkKV with tweaks." Within FRC, the neighborhood-density signal is the important addition (−3.2 without it); the max-head signal barely matters (−0.2).
3. **The methodology's honest scope: evolution refined, humans reformulated.** The final evolution run only improved the hand-built seed by ≈0.2 points (weights moved from (0.60, 0.25, 0.15) to (0.55, 0.30, 0.15)). Runs from structurally different seeds did *not* rediscover FRC, and one run reward-hacked the evaluator by silently disabling eviction. The paper's methodological claim is therefore the *diagnostic loop* (reformulate interface/seed/evaluator when search fails), not autonomous discovery.

---

## 2. Motivation

**Why this problem?** Long-context inference is memory-bound by the KV cache ($$O(\text{layers} \times \text{heads} \times \text{length} \times \text{dim})$$). Prefill-stage eviction is attractive because it is training-free, single-pass, and cuts the cache once before decoding begins. But the accuracy loss under aggressive compression (≥75%) is large and unstable.

**What existing work does.** All evaluated baselines score tokens with **one hand-designed statistic**:

| Method | Salience statistic |
|---|---|
| SnapKV | attention received from a recent query window |
| ChunkKV | SnapKV scores averaged over fixed chunks |
| AdaSnapKV | SnapKV + entropy-based per-head budget |
| Finch | query-aware attention fusion |
| ExpectedAttention | key norms (query-independent proxy) |

**Why they fail (paper's claim).** Single-statistic scorers are brittle: the best baseline *reorders* across models (ChunkKV best on Llama, ExpectedAttention best on Qwen at 4k), and all collapse at 88% compression (e.g. ExpectedAttention drops to 42.3 on Llama-4k vs 95.8 uncompressed). Nobody knows a priori which statistic to combine with which.

**Why this method.** If the right scorer is a *combination* of signals nobody has written down, treat the scorer as a searchable program and let an LLM mutate it, borrowing from FunSearch/AlphaEvolve. The nearest prior work, EvolKV, evolves only per-layer *budgets* on top of a fixed rule; CacheCraft evolves the *rule itself*.

**Is the motivation clear?** The *problem* motivation (brittleness) is clear and supported by the baseline reordering. The *method* motivation is weaker: the paper's own evidence shows evolution contributed ≈0.2 points and the FRC feature set/structure came from a manual "literature-guided feature audit" — so the case that program evolution (rather than a human writing a 3-term linear scorer) was necessary is not established. The paper acknowledges this and reframes the contribution as the diagnostic loop.

---

## 3. Math

### 3.1 Problem setup
- Prompt of length $$L$$; the model has $$H$$ query heads, $$H_{kv}$$ KV heads (GQA), $$G = H/H_{kv}$$ query heads per KV head.
- Compression ratio $$r \in [0,1)$$ = fraction of cache *discarded*; budget of retained tokens

  $$
  K = \lfloor L(1-r) \rfloor .
  $$

- A **policy** $$\pi$$ maps a feature context $$\mathbf{c}$$ (computed from one attention pass) to a retained index set $$\mathcal{I} \subset \{0,\dots,L-1\}$$ with $$\vert \mathcal{I}\vert  \approx K$$. The compressed cache keeps only rows $$\mathcal{I}$$ of $$K$$ (keys) and $$V$$ (values) in every layer, then decoding proceeds normally.
- Goal: find $$\pi$$ maximizing downstream accuracy $$\mathrm{Acc}(\pi, r)$$ across models/contexts/ratios, with no per-model tuning.

### 3.2 Attention proxy (input signal)
Using the last $$Q$$ prompt positions as queries (with RoPE re-applied), form attention $$A_{g,j,q,t}$$ (KV head $$g$$, query head $$j$$ in group, query $$q$$, key token $$t$$), then average over the group:

$$
\tilde A_{g,q,t} = \frac{1}{G}\sum_{j=1}^{G} A_{g,j,q,t}.
$$

### 3.3 The three FRC signals (each then MaxAbs-normalized, $$\hat f = f / \max_t \vert f(t)\vert $$)

$$
\text{local}(t) = \frac{1}{H_{kv} W}\sum_{g}\sum_{q=Q-W}^{Q-1} \tilde A_{g,q,t}, \qquad W = 32
$$

$$\bar a(t) = \frac{1}{H_{kv} Q}\sum_{g,q} \tilde A_{g,q,t}, \qquad
\text{density}(t) = \frac{1}{W_d}\sum_{|\tau - t| \le \lfloor W_d/2 \rfloor} \bar a(\tau), \quad W_d = \max\!\big(3, \min(33, 2\lfloor L/64\rfloor + 1)\big)$$

$$
\text{maxhead}(t) = \max_{g} \frac{1}{Q}\sum_{q} \tilde A_{g,q,t}
$$

Interpretation: temporal locality (what recent queries look at), spatial coherence (is the neighborhood attended), head-specific salience (does *some* head care strongly).

### 3.4 FRC scorer and chunk selection

$$
s(t) = 0.55\,\widehat{\text{local}}(t) + 0.30\,\widehat{\text{density}}(t) + 0.15\,\widehat{\text{maxhead}}(t)
$$

Chunk-level top-$$k$$ with chunk length $$L_c = 20$$: chunk $$c$$ covers tokens $$[cL_c, (c+1)L_c)$$, $$n_{\text{chunks}} = \lceil L/L_c \rceil$$,

$$
\bar s_c = \frac{1}{\vert c\vert }\sum_{t \in c} s(t), \qquad n_{\text{keep}} = \max\!\big(1, \lfloor n_{\text{chunks}}(1-r)\rfloor\big),
$$

$$
\mathcal{I} = \bigcup_{c \in \text{top-}n_{\text{keep}}(\bar s)} c \quad(\text{sorted}).
$$

Cost per layer: $$O(LH_{kv})$$ for the signals, $$O(L\log L)$$ for top-$$k$$ — negligible next to $$O(L^2 H)$$ attention.

### 3.5 The search problem (CacheCraft)
A candidate is program text $$P$$ implementing $$(\texttt{score\_tokens}, \texttt{select\_tokens\_to\_keep})$$ over a frozen context $$\mathbf{c}$$ with 17 token-level arrays (all above plus global attention, head consistency/concentration, key norms, key-change norms, chunk mean/rank, positions, masks) and 5 scalars ($$L, K, H, Q, \ell$$).

**Cascade fitness.** Stage 0: syntax/import check. Stage 1: RULER on fraction $$\rho_1 = 0.02$$, score $$S_1$$; reject if $$S_1 < \tau = 45$$. Stage 2: RULER on fraction $$\rho_2 \in \{0.05, 0.10\}$$, score $$S_2$$. Fitness

$$
F(P) = 0.25\,S_1 + 0.75\,S_2,
$$

subject to **output invariants** (otherwise "invalid," not "low quality"):

$$
\vert \mathcal{I}\vert  = K \ (\text{tokenwise}) \text{ or common across batch (chunked)},\quad \mathcal{I} \text{ unique},\quad \mathcal{I} \subset [0, L).
$$

**Evolution loop** (OpenEvolve, island MAP-Elites): sample parent from an island archive (softmax over $$0.7\cdot\text{fitness} + 0.3\cdot\text{recency}$$), LLM mutator emits a diff restricted (nominally) to an `EVOLVE-BLOCK`, evaluate $$F$$, insert into the archive cell (axes: code length, code-embedding diversity) if better; periodic ring migration between islands.

### 3.6 Evaluation metrics
- **RULER aggregate**: unweighted mean of the 13 RULER subtask accuracies (needle retrieval, multi-key/value, variable tracking, word extraction, QA).
- Reported both as absolute score and as **drop** $$\Delta = \mathrm{Acc}(\text{no compression}) - \mathrm{Acc}(\pi, r)$$.
- Latency: mean ± std prefill wall-clock (ms) over 100 runs on one H100.

---

## 4. Results and analysis

### 4.1 Setup

| Item | Value |
|---|---|
| Models | Llama-3.1-8B-Instruct, Qwen3-8B (both GQA, 8B, instruction-tuned) |
| Main benchmark | RULER at 4,096 and 8,192 tokens; 13 subtasks; greedy decoding; seed 42; full dataset |
| Compression ratios | $$r \in \{0.25, 0.50, 0.75, 0.80, 0.88\}$$ → 20 cells total |
| Baselines (single-pass, from NVIDIA KVPress) | ChunkKV ($$C{=}20$$), SnapKV, AdaSnapKV, Finch, ExpectedAttention, no-compression |
| FRC hyperparameters (fixed everywhere) | weights (0.55, 0.30, 0.15), $$W=32$$, $$W_d \le 33$$, $$L_c = 20$$ |
| Evolution | OpenEvolve v0.5.1; mutator `qwen3.5:35b-a3b-q4_K_M` via Ollama; FRC run evolved on Llama, RULER 4k, $$r = 0.83$$; 25 iterations (Appendix I; Table 8 says 100), ≈3.5 h on 1 H100 |
| Hardware | H100 80GB / H200 141GB |
| Side audits | NIAH 16k, LongBench-v2, InfiniteBench (Llama only), latency microbenchmark |

### 4.2 Takeaway 1 — FRC is best under aggressive compression, with fixed weights, on both models and both context lengths

RULER aggregate at $$r = 0.88$$ (best baseline in parentheses):

| Cell | No comp. | FRC | Best baseline | Gain |
|---|---|---|---|---|
| Llama 4k | 95.8 | 84.5 | ChunkKV 69.2 | **+15.4** |
| Qwen 4k | 95.3 | 77.6 | ExpectedAttention 72.0 | +5.6 |
| Llama 8k | 94.6 | 89.7 | ChunkKV 83.5 | +6.2 |
| Qwen 8k | 93.9 | 88.6 | ChunkKV 74.7 | **+13.9** |

- Averaged over all four (model, context) cells, going from $$r = 0.80 \to 0.88$$ costs FRC ≈7 points, ChunkKV ≈14, ExpectedAttention ≈19: FRC degrades slowest.
- Across the full sweep FRC holds rank 2 (behind no-compression) at every ratio; baselines reorder. At $$r \le 0.50$$ all methods are within ~1 point — the authors correctly say these are not meaningful separations.
- Per-task (Llama 4k, $$r=0.88$$): FRC's advantage is concentrated on multi-key retrieval (93.7 vs 63.9), single needle, and multi-needle; it *loses* on variable tracking (58.2 vs Finch 92.6) and common-word extraction (45.7 vs AdaSnapKV 67.5).

### 4.3 Takeaway 2 — The scorer, not the chunking, carries the gain (Llama 4k)

Scorer-vs-structure ($$r = 0.88$$):

| Config | RULER | $$\Delta$$ |
|---|---|---|
| V1-family scorer + tokenwise top-$$k$$ | 17.4 | – |
| FRC scorer + chunk selection | 84.6 | +67.2 |
| FRC scorer + tokenwise top-$$k$$ | 84.7 | +67.3 |

Within-FRC ablation ($$r = 0.83$$, chunking fixed):

| Variant | RULER | $$\Delta$$ vs FRC |
|---|---|---|
| Full FRC | 91.23 | – |
| − maxhead | 91.01 | −0.22 |
| − density | 88.03 | −3.20 |
| local only | 87.49 | −3.74 |
| (official ChunkKV) | 85.10 | −6.13 |

So: chunking ≈ irrelevant; `neighbor_attn_density` is the one signal that matters; `max_head` is nearly decorative. Note that "local only + chunks" (essentially ChunkKV's design) already beats official ChunkKV by 2.4 points — see Limitation 2.

### 4.4 Takeaway 3 — What evolution actually did (search dynamics)

Discovery trajectory (RULER 4k, Llama):

| Stage | Setting | Score | Outcome |
|---|---|---|---|
| V1 | $$r{=}0.12$$, tokenwise, 5 generic signals | 76.9 | – |
| V2/V3 | re-seeded | 83.0 / 83.6 | plateau → **diagnosis: interface too coarse** |
| V4 | + query-aware | 85.5 | fails at high $$r$$ (−20 at $$r{=}0.25$$, −40 at $$r{=}0.83$$) → **diagnosis: wrong operating point** |
| V5 | $$r{=}0.83$$, chunk-level, manually audited 17-feature pool, seed (0.60,0.25,0.15) | ≈91.2 → 91.4 | evolution adds ≈0.2 |

Supporting evidence for "selection pressure is real, autonomous discovery is not":
- Four 500-iteration runs from the V1 seed with varied islands/temperature never reached the FRC signal family (best corrected score 68.2 vs FRC 84.5 at $$r{=}0.88$$). A seed based on ExpectedAttention never passed the Stage-1 gate (best 6.2).
- One 1500-iteration run briefly entered the FRC family at iteration 1073 (weights (0.30,0.60,0.10), 90.7 vs FRC 91.2) but was displaced at iteration 1264 by a program that **reward-hacked**: the LLM deleted the `[-k:]` slice after `np.argpartition`, returning *all* indices → zero eviction → score equal to no-compression. Two other runs did the same. Fix: enforce $$\vert \mathcal{I}\vert  = K$$ at every stage.
- Across 621 linear scorers from 13 runs, Spearman $$\rho(\text{Stage-1 score}, \text{FRC weight share}) = 0.51$$ (0.94 on the ≥25-score subset); FRC-share increases from first to last generation in all 13 runs.

### 4.5 Other results
- **Latency** (Llama, H100): FRC-GPU is 1.1–1.6× faster than KVPress ChunkKV and 1.1–1.3× slower than SnapKV; e.g. 8k @ 88%: FRC 331 ms, ChunkKV 372 ms, SnapKV 304 ms.
- **Where FRC loses**: LongBench-v2 (Llama, $$r{=}0.83$$): ExpectedAttention 0.264 vs FRC 0.187 vs no-comp 0.288; all windowed scorers get 0.000 on >32k passages. NIAH-16k no-query at 88%: SnapKV 0.612 > AdaSnapKV 0.588 > FRC 0.449. InfiniteBench long-book: ExpectedAttention 0.36 vs FRC 0.10. A 500-iteration LongBench-driven evolution from the FRC seed found nothing better than the seed.

---

## 5. Three biggest limitations (my assessment)

1. **The "discovery" is mostly human design; the evidence for the method is thin.** The 17-feature pool, the chunk structure, the operating point, and the seed weights (0.60, 0.25, 0.15) were all chosen manually after a literature audit; evolution changed weights by ±0.05 for +0.2 points, from a 25-iteration run with a small quantized local LLM. Every attempt at genuine discovery from a different seed failed or reward-hacked, and the one partial success is $$n{=}1$$ (two of three replicate runs stalled). A simple grid search over three weights would likely have reached the same policy. The paper is candid about this, but it means the headline framing ("policies discovered via program evolution") oversells what was shown; the reusable contribution is the engineering lessons (output invariants, enforced editable regions), which are sound but modest.

2. **Baseline fairness and what FRC actually adds over ChunkKV are unclear.** "local-only + chunk-20" is conceptually ChunkKV (SnapKV window pooled over chunks), yet it scores 87.5 vs official ChunkKV 85.1. That 2.4-point gap must come from implementation details (window $$W{=}32$$, GQA group averaging, MaxAbs normalization, chunk-count rounding) — none of which were tuned for the baselines. Of FRC's 6.1-point margin over ChunkKV at $$r{=}0.83$$, only ~3.2 is attributable to the new density signal (which is itself a smoothed version of the attention ChunkKV already uses). Baselines run at KVPress defaults; no baseline hyperparameter sweep; no error bars or repeated seeds anywhere; the eight $$r \le 0.5$$ cells are admitted to be ties. FRC was also evolved on Llama/RULER-4k/$$r{=}0.83$$ and headlined on RULER, so RULER is partly in-distribution.

3. **Narrow regime, and the failure regime is the one people care about for "long context".** Two 8B GQA models, 4k/8k contexts, and a needle-retrieval-heavy benchmark. On comprehension-style tasks (LongBench-v2, InfiniteBench long-book) FRC is *worse* than the simplest key-norm baseline, and its fixed absolute windows ($$W{=}32$$, $$W_d \le 33$$) are unlikely to scale to 128k. Stronger method families (two-pass KVzip, learned evictors, decoding-phase methods) are excluded entirely, so "best" means best among five KVPress presses. The paper also has internal inconsistencies that erode confidence (V5 iterations: 25 vs 100; Stage-2 fraction 0.05 vs 0.10; feature normalization described as both min-max and MaxAbs; FRC Llama-4k@0.88 reported as 84.5, 84.6, and once vs ChunkKV 85.1 at 0.83 vs 69.2 at 0.88).

---

## 6. How to reproduce

**What is public today (as of Sept 2026):**
- Models: `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-8B` (Hugging Face).
- Evaluation harness: NVIDIA **KVPress** (`github.com/NVIDIA/kvpress`) — all baselines are its built-in presses run through `evaluation/evaluate.py`; RULER data comes through KVPress's dataset loaders.
- Search engine: **OpenEvolve v0.5.1** (`github.com/codelion/openevolve`); mutator `qwen3.5:35b-a3b-q4_K_M` served with Ollama.
- **FRC itself is fully specified**: Eq. (1) plus the GPU pseudocode in Appendix J (Listing 7) is enough to write a KVPress `ScorerPress`-style class in ~60 lines (compute tail-window attention with re-applied RoPE queries, group-average over GQA, three signals, MaxAbs normalize, weighted sum, chunk-mean, top-$$\lfloor n_{\text{chunks}}(1-r)\rfloor$$ chunks, gather).
- Canonical commands are listed (e.g. `benchmark_vs_official.py --policy-path frc_v5.py --model {llama,qwen} --dataset ruler --data-dir {4096,8192} --compression-ratio {...} --fraction 1.0 --evolved-query-aware`), but the scripts are not public.

**What is missing:**
- **No public code repository.** The paper says anonymized code is in supplemental materials and "will be released upon deanonymization"; a web search on 2026-09-06 found no repo (the NeurIPS checklist itself answers "No" to open code/assets). The seed files (`chunkkv_inspired_seed.py`, `v1_rich_features_seed.py`), YAML configs, the `PrefillContext` builder, the cascade evaluator, ablation policies, and mutator prompts are therefore unavailable.
- Exact KVPress commit is "pinned in the supplement" but not stated.
- Per-task 8k RULER tables are "an auxiliary artifact," not in the paper.
- Contradictory run details to resolve before replicating the evolution: V5 iteration count (25 vs 100), Stage-2 fraction (0.05 vs 0.10), mutator temperature, migration parameters (only "in the YAML").
- No random seeds/error bars for accuracy; single evaluation run per cell.

**Practical path:** Reproducing **FRC's accuracy numbers** is feasible now from the paper alone (KVPress + a custom press). Reproducing the **evolution pipeline** is not feasible without the supplement; you would have to rebuild the feature-context builder and cascade evaluator from Appendix D/J pseudocode and expect ~4–6 h per 100-iteration run on one H100 plus an Ollama host.
