---
layout: post
title: 'Reading Notes of RLKV: Which Heads Matter for Reasoning? RL-Guided KV Cache Compression'
date: 2026-09-09 23:00:00
description: 'TL;DR: Some of heads should not do KV cache eviction. This paper predicts which heads to do KV cache eviction and which heads to do full KV cache. MDP process trained with RL/GRPO based on task success - non-sparsity of KV cache eviction heads (more task success with more heads to do KV cache eviction.)'
tags: nlp kv-cache
categories: ReadingNotes
thumbnail: assets/img/blogs/rlkv.png
toc:
  sidebar: left
related_posts: false
---

TL;DR: Some of heads should not do KV cache eviction. This paper predicts which heads to do KV cache eviction and which heads to do full KV cache. MDP process trained with RL/GRPO based on task success - non-sparsity of KV cache eviction heads (more task success with more heads to do KV cache eviction.)

<!--more-->

> **Reference**: https://arxiv.org/pdf/2510.08525

## 1. Summary

Reasoning LLMs (DeepSeek-R1 distills, Qwen3-Thinking) write thousands of chain-of-thought (CoT) tokens, and every token's attention keys/values must be stored in the **KV cache**, which becomes the memory bottleneck at serving time. Existing cache-compression tricks that are harmless on ordinary instruct models break reasoning models. The paper asks: **which attention heads must keep their full cache for reasoning to survive?** RLKV answers this by freezing the model, attaching one scalar gate per KV head that blends "full attention" with "cheap local attention," and training only those gates with reinforcement learning (reward = final answer correct) plus an L1 penalty pushing gates to zero. Heads whose gates stay high are *reasoning-critical* and keep a full cache; the rest keep only a constant 80-token window. On three 4–8B reasoning models over GSM8K/Math500/AIME24/MBPP/MMLU-Pro: (1) 20–60% cache reduction is near-lossless and beats H2O/R-KV/DuoAttention/KVZip by up to ~20 points; (2) up to 2.06× end-to-end speedup in SGLang; (3) reasoning heads are empirically different from the "retrieval heads" prior work protects.

## 2. Motivation

KV cache size scales with generated length, and reasoning models generate 3–10× longer outputs than instruct models, so compression matters more for them. Prior work has two families: **token-dropping** (H2O, R-KV) evicts low-attention tokens from every head; **head-reallocation** (DuoAttention, KVZip, RazorAttention) keeps full cache only on "retrieval heads" found with needle-in-a-haystack recall objectives. Both are stable on instruct models but collapse on their reasoning twins even when uncompressed accuracy is identical (Fig. 1/10), isolating long CoT as the cause. They fail differently: token-dropping deletes intermediate steps → repetitive loops; head-reallocation protects the wrong heads because its selection objective is a static, teacher-forced recall proxy that never observes how compression errors compound during free-running generation → the model wanders and exceeds the length budget. The fix is therefore to *select heads using the only signal that reflects generation-time reasoning quality — on-policy rollout reward*, which RLVR already provides, and the tiny search space ($$L \times H$$ scalars) makes RL cheap. The motivation is clear; the one soft spot is that "reasoning-critical" is defined operationally (high gate after training) rather than mechanistically.

## 3. Math

### Setup
- Frozen reasoning LLM with $$L$$ layers and $$H$$ KV heads per layer (GQA). Query $$q$$, sampled output $$o$$, gold answer $$y$$.
- For head $$(l,h)$$, two attention outputs on the same $$Q,K,V$$:
  - $$\mathbf{a}^{\text{full}}_{l,h}$$: causal attention over all cached tokens.
  - $$\mathbf{a}^{\text{loc}}_{l,h}$$: attention restricted to the first $$n_s$$ *sink* tokens plus the last $$n_w$$ *recent* tokens (StreamingLLM mask). Training: $$n_s=128, n_w=256$$; inference: $$n_s=16, n_w=64$$.
- Gate $$\alpha_{l,h} \in [0,1]$$, $$\boldsymbol{\alpha} \in [0,1]^{L\times H}$$, the **only** trainable parameters.

### Mixed attention (the probe)

$$
\mathbf{a}_{l,h} = \alpha_{l,h}\,\mathbf{a}^{\text{full}}_{l,h} + (1-\alpha_{l,h})\,\mathbf{a}^{\text{loc}}_{l,h}
$$

This defines a policy $$\pi_{\boldsymbol\alpha}(o \mid q)$$. $$\alpha_{l,h}\to 0$$ means "this head is fine with a constant-size cache."

### Reward and advantage (GRPO, no KL term)
Sample $$G$$ outputs $$o_1,\dots,o_G \sim \pi_{\boldsymbol\alpha_{\text{old}}}(\cdot\mid q)$$ ($$G=4$$, temperature 1.0), verifiable reward $$r_i = \mathbb{1}[\text{answer}(o_i)=y]$$, group-normalized advantage

$$
A_i = \frac{r_i - \bar r}{\operatorname{std}(r_1,\dots,r_G)},\qquad \bar r = \tfrac1G\textstyle\sum_i r_i .
$$

### Objective (maximize)

$$
\mathcal{J}(\boldsymbol\alpha)=\underbrace{\frac1G\sum_{i=1}^{G}\min\!\Big(\rho_i A_i,\ \operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i\Big)}_{\text{keep reasoning correct}}
\;-\;\underbrace{\frac{\beta'(\bar r)}{L H}\,\Vert \boldsymbol\alpha\Vert _1}_{\text{make gates sparse}},
\qquad \rho_i=\frac{\pi_{\boldsymbol\alpha}(o_i\mid q)}{\pi_{\boldsymbol\alpha_{\text{old}}}(o_i\mid q)} .
$$

### Two stabilizers
1. **Adaptive penalty weight** (fights the "sparser reward → relatively stronger L1 → more collapse" spiral):

   $$
   \beta'(\bar r)=\mathbb{1}[\bar r>\tau]\cdot\beta\cdot\big(e^{\bar r}-1\big),
   $$

with base $$\beta\in\{10^{-3},10^{-3},2.5\cdot10^{-3}\}$$ and cutoff $$\tau\in\{0.5,0.55,0.5\}$$ for Llama-R1 / Qwen2.5-R1 / Qwen3-Thinking. Penalty is switched off when the group reward falls below $$\tau$$.
2. **Self-distillation sampling**: training set $$\mathcal{D}=\{q:\ \text{full model solves } q \text{ greedily}\}$$, then 3k problems bucketed by the model's own output length (600 / 600 / 1000 / 800 problems for 0–2k / 2–4k / 4–6k / 6–8k tokens) so rewards stay dense.

### Inference (discretize)
Given target sparsity $$s\in\{0.2,0.4,0.6,0.8\}$$ (= fraction of KV heads compressed), keep set

$$
\mathcal{K}_s=\text{top-}\lceil(1-s)LH\rceil \text{ heads by } \alpha_{l,h}.
$$

Heads in $$\mathcal{K}_s$$ store a full KV cache; all other heads store only $$n_s+n_w=80$$ tokens (and attend fully over what they store — no mixing at test time). Per-sequence cache ≈ $$(1-s)\cdot\text{full} + s\cdot 80\cdot LH\cdot d$$.

### Evaluation quantities
- **Accuracy**: Pass@1 with greedy decoding; reported with $$\Delta = \text{acc}_s - \text{acc}_{\text{full}}$$. "Near-lossless" ≈ $$\vert \Delta\vert \le 3$$ points.
- **Speedup** (SGLang continuous batching): throughput ratio vs. full attention; theoretical bound

  $$
  \text{speedup}_{\max}(s)=\frac{1}{(1-s)+s\,W/\bar L},\quad W=n_s+n_w,\ \bar L=\text{avg. sequence length}.
  $$

- **Head sensitivity**: compress the top-$$f$$ fraction of heads ranked by some score (RLKV $$\alpha$$ / DuoAttention retrieval score / random) and plot accuracy vs. $$f$$; a steeper drop ⇒ the ranking found more important heads.
- **Error modes** on samples the full model solves: *repetitive* (loops), *incorrect* (wrong final answer), *overlength* (hits max tokens).
- **Fair baselines**: token-dropping budgets are *dynamic*, $$b(\text{sample}) = s\cdot\vert \,o^{\text{full}}(\text{sample})\vert $$, so every sample is compressed by the same ratio.

## 4. Results and Analysis

**Organization.** The experiments answer four questions in order: **(A)** does the head selection give a better accuracy–sparsity frontier than existing methods, including out of the training domain and at long context? **(B)** does the memory saving turn into real wall-clock speedup? **(C)** which training ingredients are necessary (ablations)? **(D)** are the selected heads actually *reasoning* heads, i.e. different from retrieval heads, and why do the failure modes look the way they do?

**General setup.** Models: Llama-3.1-8B-R1 and Qwen-2.5-7B-R1 (both DeepSeek-R1 distills) and Qwen3-4B-Thinking-2507; all GQA (KV group 4, 7, 4). Training: 3k DeepScaleR math problems (self-distilled), GRPO with $$G=4$$, AdamW lr 0.01, batch 32, 185 steps / 2 epochs, max rollout 8k tokens, 2×A100-80G, 22–40 GPU-hours per model. Evaluation: greedy, Pass@1, max 8192 tokens (16384 for AIME24), DeepSeek-R1 prompt template, HuggingFace Transformers for math/code and the authors' SGLang integration for MMLU-Pro. Sparsity $$s\in\{0.2,0.4,0.6,0.8\}$$. Baselines: H2O and R-KV (token-dropping, dynamic budget, +80 sink/local tokens for parity), DuoAttention and KVZip (head-reallocation).

### Part A — Accuracy–sparsity frontier (Fig. 5, Tables 1, 2, 6–8)
Setup: 4 reasoning benchmarks (GSM8K, Math500, AIME24 = 30 problems, MBPP) + 4 MMLU-Pro subsets (Chem/CS/Law/Phys, ≤500 each) + LongReason 64K-input subset (400 samples, ~70K context).

- **Takeaway A1 — near-lossless at 20–60% reduction, usually $$s=0.4$$.** For most math/code tasks the largest sparsity within ~3 points of full is 0.4 (0.2 for Qwen2.5 on AIME/MBPP; 0.6 for several MMLU-Pro subsets). Trained only on ≤8k-token math, this transfers to code, knowledge QA, and 70K-context reasoning.
- **Takeaway A2 — biggest margin over baselines at $$s=0.4$$–$$0.6$$ (up to ~20 points).** Head-reallocation baselines are close to RLKV at $$s=0.2$$; token-dropping H2O is unusable on reasoning models at any sparsity.
- **Takeaway A3 — nothing survives $$s=0.8$$, and RLKV is not uniformly best there.** All methods lose 30–90 points; R-KV occasionally beats RLKV at 0.8 (e.g. GSM8K on both Llama and Qwen3). The method's value is in the moderate-sparsity regime.

| Model / task | $$s$$ | Full | H2O | R-KV | DuoAttn | KVZip | **RLKV** |
|---|---|---|---|---|---|---|---|
| Llama-8B-R1 / Math500 | 0.4 | 83.0 | 27.0 | 77.8 | 81.6 | 75.4 | **84.6** (+1.6) |
| Llama-8B-R1 / Math500 | 0.6 | 83.0 | 6.0 | 56.8 | 69.0 | 65.4 | **78.0** (−5.0) |
| Llama-8B-R1 / MBPP | 0.6 | 62.6 | 0.0 | 14.6 | 40.4 | 39.6 | **51.8** (−10.8) |
| Qwen2.5-7B-R1 / GSM8K | 0.6 | 89.1 | 3.5 | 67.9 | 57.1 | 66.7 | **83.1** (−6.0) |
| Qwen2.5-7B-R1 / MMLU-Pro CS | 0.6 | 55.4 | 11.2 | 27.3 | 27.3 | 20.0 | **43.7** (−11.7) |
| Qwen3-4B-Th / Math500 | 0.6 | 77.6 | 2.4 | 35.6 | 53.8 | 71.2 | **75.6** (−2.0) |
| Qwen3-4B-Th / AIME24 | 0.4 | 43.3 | 0.0 | 26.7 | 33.3 | 40.0 | **50.0** (+6.7) |
| Llama-8B-R1 / GSM8K | 0.8 | 89.1 | 2.9 | **48.6** | 28.5 | 34.3 | 47.4 (−41.8) |
| Llama-8B-R1 / LongReason 70K | 0.6 | 49.25 | OOM | 0.0 | 35.25 | 36.0 | **45.25** (−4.0) |
| Qwen3-4B-Th / LongReason 70K | 0.6 | 70.25 | OOM | 0.5 | 55.5 | 53.0 | **58.25** (−12.0) |

### Part B — End-to-end efficiency (Table 3)
Setup: Llama-3.1-8B-R1, Math500, SGLang v0.5.2 continuous batching on one A100-40G, window 16+64. Compressed heads live in a fixed-size circular buffer, so freed memory raises the number of concurrent requests.

- **Takeaway B1 — memory saving converts into throughput; $$s=0.4$$ gives 1.56× at zero accuracy cost, $$s=0.6$$ gives 2.06× at −5.6 points.** Measured speedups track the theoretical bound within ~15%; the gap is dual-dispatch kernel overhead.

| Sparsity | Concurrency | Acc. (%) | Throughput (tok/s) | Speedup | Theoretical |
|---|---|---|---|---|---|
| Full | 150 | 79.4 | 1,500 | 1.00× | – |
| 0.2 | 200 | 78.6 | 1,758 | 1.19× | 1.24× |
| 0.4 | 250 | 79.6 | 2,185 | 1.56× | 1.64× |
| 0.5 | 300 | 77.6 | 2,553 | 1.80× | 1.95× |
| 0.6 | 375 | 73.8 | 2,941 | 2.06× | 2.40× |

### Part C — Ablations (Fig. 6, Table 4)
Setup: Qwen-2.5-7B-R1 on Math500 for the three training components; Llama-3.1-8B-R1 on Math500 (SGLang) for the inference window.

- **Takeaway C1 — both stabilizers are load-bearing.** Without adaptive penalty weighting, Math500 accuracy at $$s=0.6$$ falls from ~0.70 to ~0.25 (training collapses, Fig. 4); without self-distillation sampling it falls to ~0.35. $$\beta$$ is sensitive: $$5\times10^{-3}$$ over-compresses (0.6 at $$s=0.4$$), $$2\times10^{-4}$$ never sparsifies enough (cliff at 0.6).
- **Takeaway C2 — a slightly larger sink+local window buys a lot at high sparsity for almost no memory.** Doubling 16+64 → 32+128 turns $$s=0.6$$ from −5.8 into −0.4 and $$s=0.8$$ from −40 into −25.

| Sink+Local (Llama, Math500, full = 80.8) | $$s=0.2$$ | $$s=0.4$$ | $$s=0.6$$ | $$s=0.8$$ |
|---|---|---|---|---|
| 16 + 64 | 78.2 | 79.4 | 75.0 | 40.6 |
| 32 + 128 | 82.2 | 81.8 | 80.4 | 55.6 |
| 64 + 256 | 79.8 | 80.2 | 79.2 | 64.6 |

### Part D — Are these really "reasoning" heads? (Figs. 3, 7, 8, 9)
Setup: rank heads by RLKV $$\alpha$$, DuoAttention retrieval score, or random; compress the top fraction $$f\le0.4$$ and measure Math500 accuracy; categorize failures on problems the full model solves; measure output length on problems solved by both.

- **Takeaway D1 — RLKV's top heads are more sensitive than retrieval heads.** On Llama and Qwen2.5, compressing the top ~20% of RLKV heads collapses accuracy to near zero, whereas the same fraction of retrieval heads or random heads keeps most accuracy; on Qwen3 the two rankings are similarly sensitive but RLKV's selection still yields better high-sparsity accuracy. Gate maps (Fig. 3) show ~a minority of heads stay high, and Qwen2.5 (KV group size 7) is harder to sparsify.
- **Takeaway D2 — different heads ⇒ different failure modes.** Compressing retrieval heads (DuoAttention) mainly yields *overlength* generations (fluent but never terminates); compressing RLKV heads yields *repetitive* and *incorrect* outputs. So the reasoning heads carry CoT consistency and the decision to stop.
- **Takeaway D3 — RLKV does not pay in tokens.** Among correctly-solved samples, RLKV's output length stays near the uncompressed model's, DuoAttention needs noticeably longer CoTs, and token-dropping's "shorter" outputs are an artifact of solving only easy problems.

## 5. Three biggest limitations (AI assessment)

**(1) The "RL is the right probe" claim is never ablated against a non-RL learned gate on the same data.** The head-reallocation baselines (DuoAttention, KVZip) select heads with objectives trained on synthetic recall/long-context data, while RLKV trains on 3k self-generated math CoTs with correctness reward and is then evaluated mostly on math. Part of the gap could therefore be *training-distribution match* rather than *RL vs. teacher-forced objective*. The clean experiment is missing: train the same $$L\times H$$ gates on the same self-distilled CoT trajectories with a supervised objective (e.g., NLL of the model's own correct traces, DuoAttention's output-deviation loss on those traces) and compare. Without it, "RL as a probe" is a plausible narrative, not a demonstrated necessity. The reasoning/retrieval-head distinction also rests on comparing to DuoAttention's masks only, not to other head scorers.

**(2) Evaluation noise and accounting make several headline numbers fragile.** AIME24 has 30 problems, so a ±6.7 delta is two problems, and all results are single greedy runs with no seeds or confidence intervals; "surpasses the full-cache baseline" is then indistinguishable from noise (SGLang non-determinism is even acknowledged). Sparsity is defined as the *fraction of heads* compressed, not measured memory: the 80-token residual and GQA grouping mean "60% reduction" is an upper bound, and the compressed buffer bypasses paged attention/radix cache, so production memory gains are unverified. The dynamic-budget protocol for token-dropping baselines uses each sample's *full-model output length*, which is oracle information no deployed system has, so the baseline comparison is fair-in-principle but not a realistic deployment protocol.

**(3) The selection is static, coarse, and capped at moderate sparsity, with non-trivial per-model cost.** Gates are fixed at training time and per KV head; the paper itself shows the cliff at $$s\ge0.8$$ and that Qwen2.5 cannot sparsify well because of group size 7. Real-world gains are thus ~1.5–2×, similar to what quantization alone gives, and the paper does not test stacking RLKV with quantization or per-head token-dropping, which is where the method would need to live in practice. Each new model requires a rollout-generation pass over ~40k problems plus 22–40 A100-hours of RL, a custom SGLang fork with dual KV pools and Triton decode kernels, and per-model hand-tuned $$\beta,\tau$$ (Qwen3 needed $$2.5\times$$ larger $$\beta$$), which suggests the recipe is not yet turnkey.

## 6. How to reproduce

**Code and assets exist.** Official repo: https://github.com/kurt232/RLKV (includes the SGLang inference integration); project page https://kurt232.github.io/RLKV. Models are public on HuggingFace: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, `Qwen/Qwen3-4B-Thinking-2507`. Training data comes from the DeepScaleR preview dataset (~40k math problems); evaluation uses GSM8K, Math500, AIME24, MBPP, MMLU-Pro (four subsets), and LongReason, all public.

**Step 1 — build the self-distilled training set.** Run each reasoning model greedily on DeepScaleR with the DeepSeek-R1 prompt template (max 8192 tokens), keep problems answered correctly, bucket by output length, and sample 600/600/1000/800 problems from the 0–2k/2–4k/4–6k/6–8k buckets (3k total). The paper does not give the random seed, the exact answer-extraction regex, or whether the curriculum is ordered during training or only used for sampling.

**Step 2 — set up mixed attention in the RL framework.** The authors use AReaL (FSDP backward, block-sparse attention approximation for the local mask) with SGLang v0.5.2 as the rollout engine (mask matrices for prefill, custom Triton kernels for decode). Add $$L\times H$$ scalar gates after SDPA implementing Eq. (1), with sink 128 / local 256 during training. Expect to rely on the repo here: the block-sparse approximation, kernel details, and how the gate is applied within a GQA group are not specified in the paper.

**Step 3 — GRPO training of the gates only.** AdamW, lr 0.01 constant, $$\beta_1=0.9,\beta_2=0.999$$, weight decay 0.017, batch 32, $$G=4$$, temperature 1.0, top-p 1.0 (0.95 for Qwen3), KL penalty disabled, otherwise AReaL's "recommended" GRPO settings, 2 epochs = 185 steps on 2×A100-80G (22–40 GPU-hours). L1 base $$\beta=10^{-3}$$ ($$2.5\times10^{-3}$$ Qwen3), cutoff $$\tau=0.5/0.55/0.5$$. Missing from the paper: the PPO clip $$\epsilon$$, gate initialization, any gate parameterization (e.g., sigmoid vs. clamping), and the learned $$\alpha$$ maps themselves (only heat-maps are shown).

**Step 4 — inference and evaluation.** Rank heads by $$\alpha$$, keep the top $$(1-s)$$ fraction with full cache, give the rest a 16+64 sink/local buffer; evaluate greedy Pass@1 with 8192 max tokens (16384 for AIME24), Transformers for math/code, the SGLang dual-pool fork for MMLU-Pro/LongReason/speed (single A100-40G for Table 3). Note the top-$$k$$ selection rule and tie-breaking under GQA grouping are only described in words.

**Step 5 — baselines.** Re-implement H2O and R-KV with dynamic per-sample budgets $$b=s\cdot\vert o^{\text{full}}\vert $$ (requires a full-model pass first) plus the same 80 sink/local tokens, eager attention; run DuoAttention and KVZip with default settings and flash attention. Baseline forks and configs are not listed in the paper; check the repo.
