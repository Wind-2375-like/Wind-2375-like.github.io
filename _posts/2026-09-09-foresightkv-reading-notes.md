---
layout: post
title: 'Reading Notes of ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution'
date: 2026-09-09 23:30:00
description: 'Something similar with the KVP (and arxived at similar time). Use SFT to train on "oracle" tokens related to future attention. Use RL/GRPO to penalize loss spikes on low-entropy (tokens that should be easy to predict) tokens.'
tags: nlp kv-cache
categories: ReadingNotes
thumbnail: assets/img/blogs/foresightkv.png
toc:
  sidebar: left
related_posts: false
---

Something similar with the KVP (and arxived at similar time). Use SFT to train on "oracle" tokens related to future attention. Use RL/GRPO to penalize loss spikes on low-entropy (tokens that should be easy to predict) tokens.

<!--more-->

> **Reference**: https://arxiv.org/pdf/2602.03203

## 1. Summary

Reasoning LLMs (e.g. Qwen3, DeepSeek-R1) solve math problems by writing very long chains of thought. Every generated token adds a key–value (KV) pair to the attention cache, so memory grows linearly and decoding slows down. "KV eviction" keeps the cache at a fixed budget by throwing away pairs, but existing hand-written rules (keep recent tokens, keep high-attention tokens) throw away things the model needs later. ForesightKV instead *trains* a tiny MLP per attention group to predict how much a KV pair will matter in the **future**. It is trained in two stages: (1) supervised learning from an oracle ("Golden Eviction") that looks at future attention on a full trace, and (2) reinforcement learning (GRPO) whose reward penalizes loss spikes on low-entropy (should-be-easy) tokens. On AIME2024/2025 with Qwen3-1.7B/4B and DeepSeek-R1-Distill-Qwen-7B, it (a) beats SnapKV/H2O/R-KV with **half** their cache budget, (b) keeps 92%/99% of full-cache accuracy at 2K/4K budgets, and (c) gives up to 9.8× throughput at 32K generation.

## 2. Motivation

Long reasoning traces make the KV cache the dominant memory cost (4.5 GB for one 32K Qwen3-4B sequence). Prior eviction methods are either **training-free rules** (H2O: cumulative attention; SnapKV: attention from the last few tokens; R-KV: SnapKV + redundancy) or **trained but one-shot** scorers (Lancucki et al. 2025) that judge importance once. The authors' empirical study shows why these fail: attention heads mix three patterns — *global* (always attended), *position-dependent* (locality), and *semantic-dependent* (block-wise, shifting over time, sometimes becoming permanently irrelevant). Rules based on recent attention miss semantic-dependent pairs that are unimportant *now* but needed *later*. A second finding: eviction hurts **low-entropy tokens** (numbers, symbols, copied entities) far more than high-entropy ones (+147% vs +52% loss on math), and these factual slips corrupt downstream reasoning. Hence: learn future importance (supervised from an oracle) and directly optimize against low-entropy loss spikes (RL). The motivation is clear and well supported; the only soft spot is that the "low-entropy tokens matter most" claim is asserted from loss ratios rather than from a causal accuracy experiment.

## 3. Math

All quantities below live inside one (layer, GQA group); the same scorer architecture is trained independently for every group. Superscripts for layer/head are dropped.

### 3.1 Problem setup (fixed-budget dynamic eviction)

- $$B$$: cache budget (number of KV pairs kept), $$L$$: eviction interval.
- After every $$L$$ generated tokens the cache holds $$B+L$$ pairs. The newest $$L$$ are always kept; among the older $$B$$, evict $$L$$ so the cache returns to $$B$$. Evictions are permanent.
- Eviction step $$t=1,\dots,M$$, with $$M=\lceil (T-B)/L\rceil-1$$ for a trace of length $$T$$.

### 3.2 Scoring model (policy)

For KV pair $$i$$ build the feature

$$
x_i=\mathrm{Concat}(k_i,\;v_i,\;a_i)\in\mathbb{R}^{2D+6g},
$$

where $$k_i,v_i\in\mathbb{R}^D$$ are the cached key/value and $$a_i\in\mathbb{R}^{6g}$$ are hand-crafted attention statistics for each of the $$g$$ heads in the group (attention received from the last 8/16/32/$$L$$ queries, cumulative attention, and 0.9-decayed cumulative attention). The scorer is a 1-hidden-layer MLP (hidden size 16):

$$
\phi_i=\pi_\theta(x_i)=\sigma(x_iW_1+b_1)W_2+b_2 .
$$

**Action = Top-K multinomial eviction.** Take the $$2L$$ lowest-scored candidates, then sample $$L$$ of them to evict with probability $$\propto \mathrm{softmax}(-\phi)$$:

$$
\mathcal{D}_t=\mathrm{Multinomial}_L\big(\mathrm{softmax}(\mathrm{Top}_{2L}(-\Phi))\big),\qquad s_{t+1}=s_t\setminus\mathcal{D}_t .
$$

Pure top-K is brittle to ranking errors; pure multinomial can evict crucial pairs; the hybrid is a middle ground and gives the stochasticity RL needs.

### 3.3 Stage 1 — Supervised training from "Golden Eviction"

**Oracle labels.** Run the frozen LLM on a full trace $$w_{1:T}$$ with full attention and get the $$T\times T$$ attention matrix $$A$$ for each head. Partition the query axis into blocks of $$L$$ starting at position $$B+L$$. The *block score* of KV pair $$i$$ at block $$j$$ is the attention it receives, averaged over the block's queries and over the heads in the group:

$$
\tilde a_{i,j}=\mathrm{mean}_{h\in\text{group}}\;\mathrm{mean}_{q\in\text{block } j}\;A^h_{q,i}.
$$

The **future score** is the largest attention pair $$i$$ will ever receive from now on:

$$
\alpha_{i,t}=\max_{t\le j\le M}\tilde a_{i,j},
\qquad
s_t^{\star}=\mathrm{Top}_{B-L}\big(\{\alpha_{i,t}\}\big).
$$

Justification (App. A): for a single query, evicting a set $$E$$ with attention mass $$\epsilon=\sum_{j\in E}a_j$$ changes the attention output by at most $$2C\epsilon$$ ($$\Vert v\Vert \le C$$), and $$\sum_i\alpha_{i,t}$$ over the evicted set upper-bounds the cumulative future evicted mass. So keeping max-future-attention pairs minimizes a bound on future output drift.

**Loss.** Follow the oracle eviction trace; at each step, ask the scorer to reproduce the oracle's *ranking* with a hinge pairwise ranking loss (margin $$m=0.01$$):

$$
\mathcal{L}_{\text{sup}}(\theta)=\sum_{t}\;\sum_{\alpha_{i,t}<\alpha_{j,t}}\max\big(0,\;m-(\phi_{j,t}-\phi_{i,t})\big).
$$

(Sign convention chosen so that higher future score $$\Rightarrow$$ higher predicted score.)

### 3.4 Stage 2 — RL (GRPO) to close the train/inference gap

Supervised labels come from *full-attention* states, but at inference the LLM's hidden states are already perturbed by earlier evictions. Model eviction as an MDP:

- State $$s_t$$: currently kept cache. Action $$a_t$$: which pairs to keep. Policy: the scorers $$\{\pi_\theta\}$$ of all groups jointly.
- **Reward** (sequence-level). Let $$\ell^{\text{full}}_n,\ell^{\text{evict}}_n$$ be the LM loss of token $$w_n$$ with full vs evicted cache, $$\Delta\ell_n=\ell^{\text{evict}}_n-\ell^{\text{full}}_n$$. Define the "hurt low-entropy" set

  $$
  E=\{\,n:\;H_n\in\text{bottom }80\%\text{ of token entropies},\;\Delta\ell_n>\eta\,\},\qquad \eta=1.5,
  $$

and the reward

$$
R=-\sum_{n\in E}(\Delta\ell_n)^2 .
$$

The square specifically punishes catastrophic spikes.
- **GRPO.** For one trace, sample $$G=8$$ eviction trajectories $$o_1..o_G$$ (via Top-K multinomial), compute rewards, normalize within the group

  $$
  \hat A_k=\frac{R_k-\mathrm{mean}(R)}{\mathrm{std}(R)},
  $$

broadcast $$\hat A_k$$ to every eviction decision of every scorer in that trajectory, and optimize the clipped objective with a KL term toward the supervised model $$\pi_{\text{ref}}$$:

$$
\mathcal{J}(\theta)=\mathbb{E}\Big[\min\big(r_t\hat A,\;\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat A\big)\Big]-\beta\,\mathrm{KL}\big[\pi_\theta\,\Vert \,\pi_{\text{ref}}\big],
\quad r_t=\frac{\pi_\theta(a_t\vert s_t)}{\pi_{\theta_{\text{old}}}(a_t\vert s_t)} .
$$

Only the MLP scorers are updated; the LLM never receives gradients.

### 3.5 Evaluation metrics

- Accuracy: pass@1 on AIME2024/2025 averaged over 32 samples.
- Fidelity: LM-loss ratio $$\ell^{\text{evict}}/\ell^{\text{full}}$$ and cosine similarity of attention outputs before/after eviction.
- Efficiency: max concurrent batch size and tokens/s throughput at fixed GPU memory.

## 4. Results and Analysis

**Organization.** The experiments answer four questions in sequence: **(A)** Is the oracle worth imitating? (loss-ratio study of Golden Eviction) → **(B)** Does the trained scorer beat rule-based eviction on the target task, and does each training stage help? (main AIME results + ablations on reward/inputs/sampling) → **(C)** Does it generalize beyond the training distribution — other domains, long-input tasks, bigger models, other baselines? → **(D)** Is it actually faster? (throughput). *General setup:* models Qwen3-1.7B, Qwen3-4B, DeepSeek-R1-Distill-Qwen-7B; scorers trained on Qwen3-4B-generated STILL math traces (correct, >4096 tokens); budgets $$B\in\{1K,2K,4K\}$$, $$L=256$$, candidate pool $$2L=512$$; sampling temp 0.6 / top-k 20 / top-p 0.95; baselines SnapKV, H2O, R-KV, all re-compressing every $$L$$ tokens.

### Part A — The oracle is far better than rules (Table 2)

*Setup:* Qwen3-4B, loss ratio vs full cache on sampled traces, budgets 1024/2048, intervals 128/256.

- **Takeaway 1:** Looking ahead with future attention almost eliminates eviction damage; rules do not.

| Method | (1024,256) | (2048,256) | (1024,128) | (2048,128) |
|---|---|---|---|---|
| Golden | **1.071** | **1.017** | **1.072** | **1.019** |
| R-KV | 1.410 | 1.161 | 1.475 | 1.181 |
| SnapKV | 1.409 | 1.128 | 1.421 | 1.134 |
| H2O | 1.273 | 1.095 | 1.411 | 1.158 |

The same gap holds on Qwen3-14B/32B (Table 15: Golden 1.04/1.03 vs baselines 1.14–1.27). This justifies distilling the oracle into a scorer.

### Part B — Main results and ablations (Figure 3, Tables 4, 5)

*Setup:* pass@1 (32 runs) on AIME2024/2025; ablations on Qwen3-4B at 1K budget.

- **Takeaway 1: Half the budget, same or better accuracy.** ForesightKV at 1K beats R-KV at 2K (Qwen3-4B AIME24: 54.5 vs 44.8), and at 4K it roughly matches the full model (retains 92% at 2K, 99% at 4K). Baselines collapse hard at 1K.

| Qwen3-4B, AIME24 pass@1 | 1K | 2K | 4K |
|---|---|---|---|
| Full cache | ~71 | ~71 | ~71 |
| R-KV | ~20 | 44.8 | ~62 |
| ForesightKV | **54.5** | **70.2** | **69.2** |

- **Takeaway 2: Both stages matter, but supervised does the heavy lifting.** SFT alone already jumps far above the rules; RL adds a consistent but modest gain (AIME24 51.7→54.5, AIME25 40.9→42.3 at 1K). Training at $$B\le 2K$$ generalizes to 4K unseen budgets.
- **Takeaway 3: Reward and sampling design are not arbitrary.** Naively minimizing overall loss does nothing; optimizing high-entropy tokens *hurts*; squared loss on hurt low-entropy tokens is best. Pure multinomial sampling destroys performance; dropping KV representations from the input roughly halves accuracy.

| Reward (1K, Qwen3-4B) | none (SFT) | $$-\mathcal{L}_{\text{all}}$$ | $$-\mathcal{L}_{\text{low}}$$ | $$-\mathcal{L}_{\text{high}}$$ | $$-\mathcal{L}_{\text{low,large}}$$ | $$-\mathcal{L}_{\text{ours}}$$ |
|---|---|---|---|---|---|---|
| AIME24 | 51.7 | 50.6 | 53.5 | 49.6 | 53.8 | **54.5** |
| AIME25 | 40.9 | 40.0 | 40.4 | 35.4 | **42.3** | **42.3** |

| Input / Sampling (1K) | AIME24 | AIME25 |
|---|---|---|
| Attn+KV / Top-K+MN | **51.7** | **40.9** |
| Attn only / Top-K+MN | 37.5 | 22.9 |
| Attn+KV / MN only | 16.5 | 13.8 |
| Attn+KV / Top-K only | 46.0 | 37.7 |

### Part C — Generalization (Tables 6, 7, 13, 14, 17)

*Setup:* scorers trained only on math traces; tested on GPQA (science), LiveCodeBench-V3 (code), LongBench (long input, one-shot compression after prefill, non-thinking mode), MiniCPM-4.1-8B, Qwen3-14B, and extra baselines RPC, DuoAttention, G-KV.

- **Takeaway 1: Domain transfer without retraining.** On Qwen3-4B, GPQA at 1K: 45.2 vs R-KV 23.0 (full 54.6); LiveCodeBench at 1K: 55.7 vs R-KV 34.4 (full 63.4). Note RL gives ~zero gain here.
- **Takeaway 2: Works on long-input tasks too.** LongBench average 38.95 vs full 39.41 vs R-KV 37.11 — near-lossless with a 1K budget.
- **Takeaway 3: Scales across architectures and beats stronger baselines.** Qwen3-14B at 2K: R-KV 43.6 → SFT 72.9 → +RL 73.8. On DeepSeek-7B AIME24 at 2K: G-KV 47–49, DuoAttention 13.3, ForesightKV 52.9; on Qwen3-4B RPC gets 10.0/36.7 at 1K/2K vs 54.5/70.2.

### Part D — Efficiency (Table 3, App. E.9)

*Setup:* Qwen3-4B, one A800, generation lengths 8K/16K/32K.

- **Takeaway 1:** A fixed budget makes batch size and throughput length-independent: 1K budget gives 96 concurrent sequences and 369 tok/s at 32K vs 11 and 37.7 for full cache (**9.79×**); 2K gives 7.1×, 4K gives 5.1×. Eviction overhead is 2.7% of wall time (R-KV: 8.1%). End-to-end AIME eval on DeepSeek-7B: 8h (1K) vs 17.5h (full).

## 5. Three Biggest Limitations (AI assessment)

**1. Training is still off-policy with respect to generation.** Golden Eviction labels come from the *full-attention* model, and even the RL stage scores a *fixed, pre-generated* trace: hidden states change with the evicted cache, but the tokens do not. Real inference under eviction produces different tokens, different reasoning paths, and different entropy profiles — none of which the reward sees. The entropy buckets themselves are computed on the original model. The RL stage therefore closes only part of the train/inference gap it is motivated by, which may explain why RL gains are small (≈+1–3 points on AIME, ≈0 on GPQA/LiveCodeBench) and why no confidence intervals are reported despite AIME having only 30 problems (32-run averages still have noise on the order of the RL improvement).

**2. The method depends on materialized attention scores, which fights modern attention kernels.** Both the input features $$a_i$$ (windowed and cumulative attention) and the Golden labels (full $$T\times T$$ attention matrices for 32K traces) require explicit attention probabilities. FlashAttention/PagedAttention deliberately never materialize these, so either the inference engine must run a slower attention path or maintain extra score-accumulation kernels; the paper reports 2.7% overhead but never says how the attention statistics are obtained inside its serving stack, and the throughput numbers are for a research setting, not vLLM/SGLang. The oracle side is also expensive (quadratic memory per head per trace), which is part of why training is 16 GPU-hours (SFT) + 48 GPU-hours (RL) *per model*.

**3. Per-model, per-layer, per-group scorers with irreversible eviction and no error recovery.** Every backbone needs its own set of hundreds of scorers trained on traces from that backbone's domain; nothing transfers across models. Eviction is permanent, so a single scorer mistake on a "global" token (attention sink, problem statement number) propagates through the rest of a 32K trace — the exact failure mode the low-entropy analysis documents. The comparison set also flatters the method: the main baselines are all training-free, and the only trained baselines appear as loss ratios or in an appendix table with numbers copied from other papers. Finally, three design choices (bottom-80% entropy, $$\eta=1.5$$, squared reward) are tuned on Qwen3-4B/AIME24 and then evaluated on the same benchmark, so some of the reported ablation ordering may be selection effect.

## 6. How to Reproduce

**Step 1 — Get code and backbones.** Clone https://github.com/RUCAIBox/ForesightKV and download Qwen3-1.7B, Qwen3-4B, and DeepSeek-R1-Distill-Qwen-7B (optionally Qwen3-14B, MiniCPM-4.1-8B). Scorers are 1-hidden-layer MLPs (hidden 16) per layer × GQA group, input dim $$2D+6g$$.

**Step 2 — Build training traces.** Take questions from the STILL dataset (Min et al. 2024), generate reasoning traces with Qwen3-4B at temperature 0.6 / top-k 20 / top-p 0.95, keep only *correct* traces longer than 4096 tokens (max 32K). The paper says all backbones' scorers are trained "on the same datasets independently", which reads as Qwen3-4B traces reused for all models — but it does not state the number of traces, and this reuse is not confirmed.

**Step 3 — Supervised stage.** Run each backbone with full attention on each trace, collect per-head attention matrices, compute block scores ($$L=256$$ stride from position $$B+L$$, $$B=1024$$), future scores via the running max, and Golden eviction traces. Train scorers with the pairwise hinge loss ($$m=0.01$$), batch 8, LR 1e-2, cosine schedule, 1000 steps.

**Step 4 — RL stage.** Initialize policy = reference = SFT scorers. Batch 32, LR 3e-4 with 10 warmup steps, 200 steps, $$G=8$$ trajectories per trace, $$(B,L)=(1024,256)$$ if trace <12K tokens else $$(2048,512)$$, $$\eta=1.5$$, entropy cutoff bottom 80%, clip $$\epsilon=0.2$$, KL $$\beta$$ = 0.01 (Qwen3-4B) / 0.03 (Qwen3-1.7B) / 0.1 (DeepSeek-7B). Budget ≈ 8h×2 H800 (SFT) + 6h×8 H800 (RL) for the 7B model. **Missing:** how $$\pi_\theta(a_t\vert s_t)$$ is defined for a sampled *set* of $$L$$ evictions (sequential without-replacement probability? per-item independent?), and hence how the importance ratio and KL are computed.

**Step 5 — Evaluate.** AIME2024/2025 pass@1 averaged over 32 seeds at $$B\in\{1K,2K,4K\}$$, $$L=256$$, candidate pool 512; baselines SnapKV (8-token window), H2O, R-KV (0.1 attention / 0.9 redundancy) re-compressed every $$L$$ tokens. Extra: GPQA, LiveCodeBench-V3, LongBench (one-shot post-prefill compression, non-thinking). **Missing:** the inference framework used for the throughput table and how per-layer, per-group ragged caches and attention statistics are implemented; exact full-model and baseline numbers behind Figure 3 (only plotted); SCP-116K training details for the science-domain reward experiment.
