---
layout: post
title: 'Reading Notes of KVP: Learning to Evict from Key-Value Cache'
date: 2026-09-03 22:46:08
description: 'Instead of evicting KV cache with checking whether previous tokens contribute to the current one, learn to predict whether past tokens contribute to the future tokens. Prediction returns a sorted list of how possible every past token can contribute to future tokens with respect to the current token. It''s still a heuristic (future attention), but it must be learned because we won''t know future attention when decoding.'
tags: nlp kv-cache
categories: ReadingNotes
thumbnail: assets/img/blogs/kvp.png
toc:
  sidebar: left
related_posts: false
---

Instead of evicting KV cache with checking whether previous tokens contribute to the current one, learn to predict whether past tokens contribute to the future tokens. Prediction returns a sorted list of how possible every past token can contribute to future tokens with respect to the current token. It's still a heuristic (future attention), but it must be learned because we won't know future attention when decoding.

<!--more-->

> **Reference**: https://arxiv.org/pdf/2602.10238

## Motivation

Don't use heuristics to evict KV cache but learn. (but you learn to predict another heuristics... you even don't use success signal as reward?)

Also when you evict, the future should be changed, but you assume future is not changed.

## Policy Training

**Setup.** Cache entries $$\mathcal{X} = \{x_i\}_{i \in [n]}$$ with $$x_i = (k_i, v_i, pos_i)$$, and $$f$$ recorded future tokens $$x_{n+1}, \dots, x_{n+f}$$ taken from the training corpus. Let $$A(x_i, x_j)$$ be attention from query $$x_j$$ to key $$x_i$$, maxed over query heads in the group under GQA.

**Oracle target.** Future utility of a cached token,

$$
u_i = \sum_{j=n+1}^{n+f} A(x_i, x_j)
$$

**Cost of a ranking.** Under permutation $$\sigma$$, budget $$b$$ keeps $$\sigma_1 \dots \sigma_b$$ and evicts the rest, so the per-budget cost is the discarded future attention $$\mathcal{C}^b = \sum_{i>b} u_{\sigma_i}$$. Summing over all budgets gives the area under the cost curve, which collapses because a token at rank $$b$$ is evicted under exactly $$b$$ smaller budgets:

$$
\mathcal{C}(\sigma) = \sum_{b=0}^{n-1} \mathcal{C}^b = \sum_{b=1}^{n} b \cdot u_{\sigma_b}
$$

By rearrangement the optimum is $$\sigma^\star = \operatorname{argsort}^{\downarrow}(u)$$, available in closed form. Normalizing by it,

$$
\mathcal{R}(\sigma) = -\frac{\mathcal{C}(\sigma)}{\mathcal{C}(\sigma^\star)} \le -1
$$

with equality at $$\sigma^\star$$. The reward reads as: being ranked late is penalized in proportion to how important you are.

**Scoring policy.** A per-head MLP $$f(x_i; \theta) = \mathrm{MLP}_\theta(k_i, v_i, pos_i)$$ induces a Plackett-Luce distribution over permutations,

$$
\pi_\theta(\sigma) = \prod_{i=1}^{n} \frac{\exp f(x_{\sigma_i};\theta)}{\sum_{j \ge i} \exp f(x_{\sigma_j};\theta)}
$$

sampled in one shot by Gumbel-Sort, $$\sigma = \operatorname{argsort}^{\downarrow}(f(x_i;\theta) + g_i)$$ with $$g_i \sim \text{Gumbel}(0,1)$$.

**Why sampling.** $$\operatorname{argsort}$$ is piecewise constant in the scores, so $$\partial \mathcal{C} / \partial s$$ is zero almost everywhere and undefined at swaps. Sampling replaces the hard sort with $$J(\theta) = \mathbb{E}_{\sigma \sim \pi_\theta}[\mathcal{R}(\sigma)]$$, which is smooth in $$\theta$$, and the $$K$$ sampled permutations supply the counterfactual contrast that a single deterministic ranking cannot. Gumbel noise specifically is what makes the sampling distribution exactly Plackett-Luce, so $$\log \pi_\theta$$ has a closed form and REINFORCE stays unbiased.

**Gradient.** Terminal reward, REINFORCE with a leave-one-out baseline $$\bar{\mathcal{R}}$$ averaged over the other $$K-1$$ episodes:

$$
\nabla_\theta J(\theta) \approx \frac{1}{K}\sum_{k=1}^{K}\Big[\big(\mathcal{R}(\sigma^k) - \bar{\mathcal{R}}\big) \sum_{i=1}^{n} \nabla_\theta \log \pi_\theta(\sigma^k_i \mid \mathcal{X})\Big]
$$

**Loop.** Collect $$Q, K, V$$ traces once with a single forward pass over the corpus. Then sample a trace $$j$$ and a split point $$n \sim \text{Unif}(n_j)$$, recompute $$A$$ from stored $$Q, K$$, draw $$K$$ permutations, score them, update $$\theta$$. No LLM inference ever runs inside training, because the future is fixed text rather than generated text.

## Inference

**Ranking is equivalent to eviction.** If each $$S_b^\star$$ is unique and nested, $$S_b^\star \subset S_{b+1}^\star$$, then $$S_{b+1}^\star \setminus S_b^\star$$ is a single element and the induced order $$\pi$$ satisfies $$S_b^\star = \{i : \pi(i) \le b\}$$ for every $$b$$. Nestedness holds here because the reward is additive over tokens, so one budget-agnostic score serves all budgets.

**Compression.** Per layer and head, deterministic, no noise:

$$
s_i = \mathrm{MLP}_\theta(k_i, v_i, pos_i), \quad \text{keep} = \operatorname{top-}b(s), \quad \text{evict} = \text{rest}
$$

with the first 4 and last 16 positions pinned. One parallel forward pass plus one sort.

**Schedules.** Short context uses a single compress-after-prefill. Long context uses chunked prefill-compress, where each new chunk is prefilled and the policy is re-applied to the union of the kept cache and the new tokens, holding total size at $$b$$ throughout.

## Results

**Setup.** Qwen2.5-7B-Chat (28 layers, 4 KV heads, so 112 agents at roughly 650K parameters each), with Phi-4 14B as a second architecture. Trained separately on RULER-4k and OASST2-4k, giving KVP$$^R$$ and KVP$$^S$$. Evaluated against attention-based baselines TOVA and SnapKV, attention-free baselines Random, StreamingLLM, LagKV, KeyDiff, K-Norm, and the concurrent learned method JudgeQ. Uniform budget across all heads and layers for every method, and absolute cache size rather than compression ratio on the x-axis.

**Main numbers**

| Benchmark | Result |
|---|---|
| RULER-4k | Leads across nearly the whole range. Hits ~90% at 2000-2500 tokens where baselines need 3000-4000 |
| OASST2-4k PPL | Lowest almost everywhere, LagKV edges it out only at the tightest budgets |
| RULER-128K (32x train length) | Dominates every baseline at every budget. 10.5 vs 8.2 at cache 100, 28.6 vs 24.5 at cache 10000 |
| BoolQ @ cache 20 | KVP$$^S$$ ~79 vs uncompressed 86, SnapKV falls to 63 |
| MMLU @ tightest budget | KVP ~60, SnapKV and K-Norm drop to roughly chance |
| LongBench retrieval EN/ZH | KVP$$^R$$ leads, also beats JudgeQ at every budget |
| Phi-4 14B | KVP stays strong while the relative ranking of heuristics shifts substantially vs Qwen |

**Cost**

| | |
|---|---|
| Decoding overhead | Zero, compression runs once after prefill |
| Prefill FLOPs | 14.00 → 14.15 GFLOPs/token, about 1% |
| Wall clock @ 10K ctx | 0.71ms per layer vs 404ms full prefill, ~570x |
| Training | Under 30 min per agent on one 8xH100 node, trace collection ~1.2 TB |

## Limitations

**Claimed by the paper.** Uniform budget across heads and layers, with adaptive allocation left open even though KVP's per-head rankings would be a natural signal for it. No cross-head modeling, since 112 independent agents cannot represent dependencies, and joint training would require online LLM inference and forfeit the offline advantage. Future attention is a proxy for task-level utility rather than a direct downstream signal. No optimized end-to-end inference pipeline, so only policy-level wall-clock is reported.

**Three issues AI thinks matter most.**

*The training target assumes eviction does not change the future.* The oracle $$u_i$$ comes from attention paid by future tokens that were generated with the full uncompressed cache. Once you evict, hidden states shift, future queries shift, and the true future attention shifts with them. Training optimizes against a counterfactual that deployment immediately violates, and the gap should compound worst under the chunked prefill-compress schedule where the policy is applied repeatedly and each application drifts further from its training assumption. The 128K results suggest this is tolerable in practice, but the assumption is never stated as one, never tested, and never addressed by anything like an iterative trace refresh.

*Uniform weighting over budgets optimizes for the wrong regime.* Summing cost over all $$b$$ from 0 to $$n-1$$ is equivalent to assuming a uniformly random budget, but nobody deploys at $$b$$ close to $$n$$. Real compression lives at $$b \ll n$$, and that is precisely where the AUC devotes the least relative pressure, since large budgets contribute most of the terms and nearly every method already performs fine there. This is a plausible explanation for the one place KVP loses, LagKV at the smallest OASST2 budgets. A truncated or budget-weighted AUC is a cheap variant that goes untested.

*The RL framing oversells what is happening, and the ablation that would settle it is missing.* There is no environment, no state transition, no exploration-exploitation tension, and no real credit assignment. The reward is a closed-form function of the sampled permutation and precomputed scalars, and its optimum is analytically $$\operatorname{argsort}^{\downarrow}(u)$$. This is stochastic optimization of a discrete objective. That matters because the supervised baselines all optimize something other than the actual cost, either rank error or $$u$$-regression error, and the paper never compares against a differentiable soft-sort surrogate that targets $$\sum_b b\,u_{\sigma_b}$$ directly. The stated counterargument, that the right rescaling would be head-dependent, is asserted rather than shown, and the supervised baselines are described as using untuned defaults while KVP received real design attention.

*Two smaller things worth flagging.* The positional feature is never specified as raw index, normalized, or sinusoidal, which directly gates how to read the headline 32 times extrapolation result. And pinning the first 4 and last 16 tokens is itself a hand-crafted sink-and-recency rule inside a paper arguing against them, which matters at the 20-token BoolQ budget where those pinned tokens are the entire cache.
