---
layout: post
title: 'Reading Notes of EvoDS: Self-Evolving Autonomous Data Science Agent with Skill Learning and Context Management'
date: 2026-09-02 23:54:46
description: 'A data science agent (Qwen3-8B backbone) trained end-to-end to (1) synthesize and reuse new skills and (2) decide when to compress its own context, via SFT distillation + multi-role GRPO.'
tags: nlp llm-agents
categories: ReadingNotes
thumbnail: assets/img/blogs/evods.png
toc:
  sidebar: left
related_posts: false
---

A data science agent (Qwen3-8B backbone) trained end-to-end to (1) **synthesize and reuse new skills** and (2) **decide when to compress its own context**, via SFT distillation + multi-role GRPO.

<!--more-->

> **Reference**: https://arxiv.org/abs/2606.03841 (KDD '26, HKUST-GZ)

![image.png](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/imgimg20260902173340.png)

## 1. Architecture: hierarchical multi-agent

All roles share **one backbone and one set of weights**; roles differ only by system prompt and action space (following MATPO).

**Manager (orchestrator).** Action space $$\mathcal{A}_{man}$$:
- Code execution: `bash`, `sql`, `python`
- Delegation: `data_cleaning` / `feature_engineering` / `model_development` / `visualization` / `debugging`, each routes a subtask $$q_j$$ to a sub-agent, $$a^{sub}_i = \text{Invoke}(\pi^i_\theta, q_j)$$
- `context_summarize`: compresses its own history

**Sub-agents** (Cleaner, Featurizer, Modeler, Visualizer, Debugger). Each has a disjoint local skill set $$\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$$ of predefined tools (e.g. `fill_missing_values`, `xgboost_classification`, `plot_heatmap`) plus one `*_tool_creation` action for skill synthesis. Debugger has no tools; it only rewrites code.

**Memory.** Every agent sees a *global memory* (overall task objective $$G$$) and a *local memory* (its subtask + execution log).

## 2. Autonomous Skill Acquisition (ASA)

A skill is $$a = \langle n, d, c \rangle$$ (name, description, code). When a subtask exceeds a sub-agent's skills:

1. **Synthesis**: LLM writes a new $$a_{new}$$ via the `*_tool_creation` action (must follow a fixed `def tool_name(parameters)` template); a separate prompt extracts a JSON tool schema.
2. **Verification**: execute it on the subtask; keep only if it runs and outputs are valid.
3. **Caching**: store in a repository $$\Delta\mathcal{A}_i = \{a_{new} \mid a_{new} \notin \mathcal{A}_i\}$$, not yet in the action space.
4. **Expansion**: count repeated synthesis by *name* $$c(a_{new})$$; promote only when $$c(a_{new}) \ge \tau$$, $$\tau = 3$$:

   $$
   \mathcal{A}_i \leftarrow \mathcal{A}_i \cup \{a_{new} \in \Delta\mathcal{A}_i \mid c(a_{new}) \ge \tau\}
   $$

**EvoDS** = no cross-task skill reuse; **EvoDS-evo** = reuse within the same benchmark. At test time: 279 skills synthesized, 925 invocations, 69% cross-task reuse rate.

## 3. Adaptive Context Compression (ACC) — two levels

- **Sub-agent level (always on)**: on finishing, the sub-agent returns a summary $$\tilde o_t = \phi(o_t \mid G)$$ instead of raw output (success → key outcomes; failure → error causes).
- **Manager level (learned)**: the Manager decides *when* to call `context_summarize`, $$a^{sum} \sim \pi^{man}_\theta(\cdot \mid C, G)$$, then $$C \leftarrow g(C \mid G)$$ via prompt-based summarization. This is what they contrast with threshold-triggered ("passive") compression.

Each summarization splits the Manager trajectory into segments $$\tau^{main} = \bigcup_{k=1}^K \tau^{main}_k$$; each segment is treated as an independent training trajectory.

## 4. Training

**Data**: 8K instances (DataMind-12K, DataScience-Instruct-500K, MatPlotBench, DSBench, MLE-Dojo), GPT-4o extracts Q/A from trajectory datasets, Qwen3-8B filters easy ones.

**Stage 1 — SFT**: DeepSeek-V3.1 acts as teacher *inside the EvoDS framework*, 8 rollouts/instance → 36K trajectories (main + sub). 3 epochs, bs 32, lr $$10^{-5}$$.

**Stage 2 — joint RL (GRPO)**, 300 steps, rollout 8, lr $$10^{-6}$$, turn budget curriculum 4→20, max response 24K, 4×A800.

Rewards:

$$
R^{sub} = \begin{cases} +0.1 & \text{subtask solved} \\ -0.1 & \text{otherwise} \end{cases}
$$

$$
R^{main} = R_{outcome} + \alpha R_{sub} - \beta P_{context} - \gamma P_{turn}, \quad \alpha = 0.2,\ \beta = \gamma = 0.1
$$

with $$R_{sub} = \frac{1}{N}\sum_i R^{sub}_i$$, $$P_{context} = \vert C\vert /C_{max}$$, $$P_{turn} = T/T_{max}$$. $$R^{main}$$ is **broadcast** to all $$K$$ segments.

Advantages: Manager uses group-normalized $$A^{main}_i = \frac{R^{main}_i - \mu}{\sigma + \epsilon}$$; sub-agents use raw $$A^{sub} = R^{sub}$$. Loss = mean clipped surrogate over main trajectories + over sub-trajectories, minus KL to reference.

**Theory (App. A)**: (i) hierarchical tool selection has a strictly lower union-bound error than flat selection (assumes Gaussian scoring noise, $$\sigma^2_S \le \sigma^2$$, $$\sigma^2_M \approx \sigma^2$$); (ii) with reward monotone in $$I(Z;Y)$$ and token cost $$\propto H(Z)$$, the Manager objective is an Information Bottleneck $$\min I(Z;C) - \lambda I(Z;Y)$$.

## 5. Results (avg over DABench / DA-Code / SAB / MLE-Dojo)

| Method | AVG |
|---|---|
| ReAct (o4-mini) | **0.450** |
| EvoDS-evo-8B | 0.424 |
| EvoDS-8B | 0.410 |
| ReAct (DeepSeek-V3.1, teacher) | 0.408 |
| DataMind-14B (best open) | 0.329 |
| Qwen3-8B in EvoDS, untrained ("w/o train") | 0.199 |

- "28.9%" = *relative* gain over DataMind-14B (absolute +9.5). Still below ReAct/o4-mini overall, but beats it on MLE-Dojo (0.311 vs 0.308).
- Ablations (Table 2): removing ACC hurts most among trained variants (0.355) and reintroduces out-of-token failures (Table 3: 0 vs. 7/257, 20/500, 18/102, 3/10). Removing ASA → 0.398; SFT only → 0.363; Manager-only GRPO → 0.399.
- Skill reuse (Table 4): DA-Code 0.341→0.351, SAB 0.108→0.118.
- Failures on DA-Code (50 cases): instruction following 52%, budget exhaustion 18%, coordination 18%, reasoning 12%.

## 6. Three biggest weaknesses

1. **Gains cannot be attributed to the proposed mechanisms.** The SFT data is distilled from DeepSeek-V3.1 *running the EvoDS framework*, so architecture, teacher distillation, and RL are entangled; there is no Qwen3-8B baseline trained on the same data under a plain ReAct/flat setup, and "w/o train" uses the untrained model inside their framework, which says nothing about the base model's real capability. The "beats DataMind-14B with fewer parameters" claim ignores that training data and teacher differ. The paper also concedes it still loses to a prompted o4-mini on average.

2. **Effect sizes for the headline mechanisms are within noise.** MLE-Dojo has 10 instances (one task = 0.1); SAB differences of 0.010 are a single task out of 102. The ASA reuse gains (Table 4, +0.010 on both benchmarks) and EvoDS-evo vs EvoDS are exactly that size, with no seeds or variance reported. DA-Code visualization is scored by LLM-as-judge for EvoDS "since it generates plots via tools," and it is unclear whether baselines were scored identically.

3. **Efficiency and theory are asserted, not demonstrated.** ACC is credited with saving context, but every summarization is an extra LLM call and the paper never reports total tokens or wall-clock vs. baselines, only per-step context length (Fig. 4) and OOT counts, so it is not shown that compression saves compute rather than shifting it. The theorems rest on assumptions that nearly encode the conclusion (lower variance for the sub-agent, monotone reward in mutual information, cost ∝ entropy). Meanwhile the system is full of hand-set heuristics (name-based skill dedup, $$\tau = 3$$, ±0.1 sub-reward, $$\alpha, \beta, \gamma$$, reward broadcasting) with no sensitivity analysis.
