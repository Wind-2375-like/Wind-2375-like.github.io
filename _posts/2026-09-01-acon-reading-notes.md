---
layout: post
title: 'Reading Notes of ACON: Optimizing Context Compression for Long-horizon LLM Agents'
date: 2026-09-01 00:23:14
description: 'I like the story of this paper to optimize cost, but it spends a long time introducing cost optimization objective while it actually uses iterative prompt engineering, which seems to be totally irrelevant.'
tags: nlp llm-agents
categories: ReadingNotes
thumbnail: assets/img/blogs/acon.png
toc:
  sidebar: left
related_posts: false
---

I like the story of this paper to optimize cost, but it spends a long time introducing cost optimization objective while it actually uses iterative prompt engineering, which seems to be totally irrelevant.

<!--more-->

> **Reference**: https://arxiv.org/pdf/2510.00615

## 1. The problem

An LLM agent solving a multi-step task with increasing context which leads to huge memory cost and long context distration:

- **Memory cost.** Transformer inference keeps a KV cache proportional to context length. A 20k-token agent context is expensive to serve, and it caps how many agents you can run concurrently.
- **Distraction.** Long contexts degrade decision quality. Stale API dumps, failed attempts, and irrelevant tool output dilute the signal. This is the well-documented "LLMs are distracted by irrelevant context" effect, and it hits smaller models much harder.

Prior work are limited in dialogue memory, document compression, heuristics, and model training:

|Approach|Why it fails here|
|---|---|
|Dialogue memory (MemGPT)|Optimizes conversational coherence, not state tracking|
|Document compression (LLMLingua)|Assumes single-step reasoning; context discardable after use|
|Heuristics (FIFO, retrieval, hand-written rules)|Brittle, environment-specific, hand-tuned|
|RL-based (MEM1, Context-Folding, AgentFold)|Entangle compression with the agent policy; require weight access and gradients, so unusable with GPT/Claude/Gemini APIs|

ACON's positioning: keep the compressor as a **separate, prompt-only module** whose _instructions_ are optimized automatically, then distill it into a small model so it's cheap to run.

---

## 2. Formal setup

### The task

An agentic task is a POMDP $$E = \langle \mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{R} \rangle$$ with state space $$\mathcal{S}$$, action space $$\mathcal{A}$$, and observation space $$\mathcal{O}$$.

The transition function is **deterministic**.

$$
\mathcal{T}(s, a) \rightarrow (s', o)
$$

The reward $$\mathcal{R}$$ is **terminal only** in $$[0,1]$$.

### The agent

The agent is an LLM $$M$$ with frozen weights $$\theta$$ and a frozen system prompt $$\mathcal{P}_{\text{agent}}$$ containing the environment description, tool schemas, output format, and few-shot examples. At step $$t$$ it sees the interaction history and the latest observation and emits an action.

$$
M(o_t, h_{t-1}; \theta, \mathcal{P}_{\text{agent}}) \mapsto a_t \tag{1}
$$

$$
h_{t-1} = (o_0, a_0, o_1, a_1, \ldots, o_{t-1}, a_{t-1})
$$

Here $$o_0$$ is the task instruction. Each $$a_t$$ is preceded by a reasoning trace. **Both $$\theta$$ and $$\mathcal{P}_{\text{agent}}$$ stay fixed throughout.**

### Cost

Define a per-step cost $$C(h_{t-1}, o_t)$$ that measures the expense of encoding the dynamic context at that step. Total cost over a trajectory is

$$
C(H) = \sum_{t=1}^{T} C(h_{t-1}, o_t), \qquad H = \{h_{t-1}, o_t\}_{t=1}^{T} \tag{2}
$$

---

## 3. The two compression operators

Introduce a second LLM $$f(\cdot\,; \phi, \mathcal{P})$$ with weights $$\phi$$ and a natural-language **compression guideline** $$\mathcal{P}$$.

### History compression

$$
h'_t = \begin{cases} f(h_t; \phi, \mathcal{P}_{\text{hist}}) & \text{if } \vert h_t\vert  > T_{\text{hist}} \\ h_t & \text{otherwise} \end{cases} \tag{3}
$$

The entire history is sent to the compressor again, ignoring the KV cache, which can be inefficient. Also, it invalidates the further KV cache.

### Observation compression

$$
o'_t = \begin{cases} f(o_t, h_{t-1}; \phi, \mathcal{P}_{\text{obs}}) & \text{if } \vert o_t\vert  > T_{\text{obs}} \\ o_t & \text{otherwise} \end{cases} \tag{4}
$$

The same issue. The compression itself is on a new LLM which fails to reuse the KV cache... It also invalidates the further KV cache.

---

## 4. The optimization objective

Let $$\psi \triangleq (\phi, \mathcal{P})$$ be the compressor parameters. Under compression, the trajectory context becomes

$$
H'(\psi) = \{h'_{t-1}, o'_t\}_{t=1}^{T} \tag{5}
$$

$$
C(H'(\psi)) = \sum_{t=1}^{T} C(h'_{t-1}, o'_t) \tag{6}
$$

With the agent frozen, $$\psi$$ induces a trajectory $$\tau(\psi)$$ and terminal state $$s_T(\psi)$$. The learning objective is a Lagrangian.

$$
\max_{\psi} \ \underbrace{\mathbb{E}\big[\mathcal{R}(s_T(\psi))\big]}_{\text{maximize}} \ - \ \lambda \underbrace{\mathbb{E}\big[C(H'(\psi))\big]}_{\text{minimize}}, \qquad \lambda \geq 0 \tag{7}
$$

The paper threw away these equations and said the reason were:

1. **No gold supervision.** There is no ground-truth correct compression of a history, so supervised learning is not directly available.
2. **Sparse, delayed reward.** $$\mathcal{R}$$ fires only at the terminal state. A compression at step 12 that dropped a token is only punished at step 30.
3. **Discrete objects.** Text in, text out. There is no gradient path from $$\mathcal{R}$$ back to $$\mathcal{P}$$.
4. **Indirect credit assignment.** Compression quality is observed only through eventual task success, which also depends on the agent's own competence.

**Yes this is the issue, so that's why you totally bypass all of these above and the rest below is completely irrelevant.**

---

## 5. ACON's answer, which is to optimize $$\mathcal{P}$$ rather than $$\phi$$

The core move is to freeze $$\phi$$ and optimize only the natural-language guideline $$\mathcal{P}$$, using an optimizer LLM to perform textual gradient descent where the gradient is a natural-language critique.

The insight that makes this work is that trajectories under compression give **dense signal**, not just a scalar. If the agent succeeds without compression and fails with it, you have a matched pair of trajectories, and you can diff them. That diff tells you exactly what got lost and where. That is far more information than a binary reward.

### Step $$\overline{\text{UT}}$$, utility maximization, which pushes reward up

On the training set $$\mathcal{D}_{\text{train}}$$, run the agent with no compression to get baseline context $$H$$ and success $$r^{\text{base}} \in \{0,1\}$$, then run it with compression under $$\mathcal{P}^{(0)}$$ to get compressed context $$H'$$ and success $$r$$. Collect the **contrastive subset**.

$$
\mathcal{D}_{\text{cont}} = \{\, i : r^{\text{base}}_i = 1 \ \wedge \ r_i = 0 \,\}
$$

These are the tasks compression broke. For each, query an optimizer LLM.

$$
\text{Feedback} = \text{LLM}(\text{Feedback Instruction}, H, H') \tag{8}
$$

The feedback instruction is an auditor prompt (Prompt D.1 in the paper) that asks for structured JSON containing the first divergence point, root-cause categories (Missing Critical Fact, Lost Variable/State, Incorrect Summary, Unnecessary Re-discovery, Instruction Drift, API Misuse, Premature Completion, Inefficient Looping), quoted evidence spans from both trajectories, and remediation recommendations. It is a fine-grained failure diagnosis, not a vibe check.

This feedback is the **natural language gradient**. Aggregate across tasks and apply one update.

$$
\mathcal{P}^{(1)} = \text{LLM}\Big(\text{Update Inst.}, \ \mathcal{P}^{(0)}, \ \big\vert _{i=1}^{n} \text{Feedback}_i\Big) \tag{9}
$$

where $$\vert $$ denotes concatenation. Concatenating $$n$$ feedbacks into one update is the textual analogue of a **batch** gradient step, since it averages over examples rather than chasing one. Sample $$K = 5$$ candidate prompts $$\{\mathcal{P}^{(1)}_k\}$$, evaluate each on $$\mathcal{D}_{\text{cont}}$$, and keep the best. That is essentially random-restart hill climbing in prompt space.

This step maximizes the first term of Equation 7.

### Step $$\overline{\text{CO}}$$, compression maximization, which pushes cost down

$$\overline{\text{UT}}$$ alone will make guidelines longer and more conservative, since every failure says you dropped something. Nothing pushes back on length. So run a second pass conditioned on the _opposite_ population, meaning tasks that **succeeded with compression**. Ask the optimizer which information was actually used during execution. Whatever was not used is dead weight. This refines $$\mathcal{P}^{(1)} \rightarrow \mathcal{P}^{(2)}$$ and minimizes the second term of Equation 7.

The alternation $$\overline{\text{UT}} \rightarrow \overline{\text{CO}}$$ is a coordinate-descent-flavored heuristic on the Lagrangian, where one stage owns reward and the other owns cost.

The two published variants are **ACON $$\overline{\text{UT}}$$** with one utility step, which is safer and retains facts better, and **ACON $$\overline{\text{UT}},\overline{\text{CO}}$$** with utility then compression, which yields shorter contexts and occasionally lower accuracy. They run exactly one round consisting of one $$\overline{\text{UT}}$$ and one $$\overline{\text{CO}}$$.

---

## 6. Distillation into a small compressor

Calling gpt-4.1 to compress on every threshold crossing is expensive, so distill.

The teacher is $$f(\cdot\,; \phi_T, \mathcal{P}^*)$$, which is gpt-4.1 with the optimized guideline. It generates compressed outputs $$y$$ from inputs $$x$$. Train a student $$\phi_S$$ with sequence-level knowledge distillation (Kim and Rush 2016), which is plain cross-entropy on the teacher's sampled outputs.

$$
\min_{\phi_S} \ \mathbb{E}_{(x,y) \sim \mathcal{D}^{+}_{\text{train}}} \left[ -\sum_{n=1}^{L_y} \log f(y_n \mid x, y_{<n}; \phi_S, \mathcal{P}^*) \right] \tag{10}
$$

The pairs are $$(x, y) = (h_t, h'_t)$$ for history compression from Equation 3, and $$(x, y) = ((h_{t-1}, o_t), o'_t)$$ for observation compression from Equation 4. The set $$\mathcal{D}^{+}_{\text{train}}$$ is filtered to tasks where **the teacher succeeded with compressed context**, so you only imitate compressions that demonstrably worked.

The full pipeline is

$$
f(\cdot\,; \phi_T, \mathcal{P}) \ \xrightarrow{\ \text{prompt optim.}\ } f(\cdot\,; \phi_T, \mathcal{P}^{*}) \xrightarrow{\ \text{distill}\ } f(\cdot\,; \phi_S, \mathcal{P}^{*}), \qquad \vert \phi_T\vert  \gg \vert \phi_S\vert  \tag{11}
$$

Note that the student keeps $$\mathcal{P}^*$$ in its prompt at inference. The guideline is not fully absorbed into weights, it is still supplied.

This decouples decision-making from compression, so an expensive model does the reasoning and a 14B model does the bookkeeping.

---

## 7. Experimental setup

### Benchmarks

|Benchmark|Environment|Split|Metric|
|---|---|---|---|
|**AppWorld** (primary)|9 apps (Spotify, SimpleNote, Amazon, Venmo, Gmail, Splitwise, FileSystem, Todoist, Phone), 457 APIs, roughly 100 simulated users, average of **42.5 API calls per task**|90 train, 168 test-normal (57 easy, 48 medium, 63 hard)|task goal completion score|
|**OfficeBench**|Word, Excel, PDF, Calendar, Email, Shell, Calculator, with 1-app, 2-app, and 3-app tiers|92 train, 95 test (42, 22, 31), random 1 to 1 split, OCR tasks excluded|benchmark accuracy functions|
|**8-objective QA**|Eight NaturalQuestions questions bundled into one task, BM25 over 2018 Wikipedia|100 train, 100 test|mean of EM and F1 over the 8 questions|
|**WebVoyager** (appendix)|Live web pages as accessibility trees|50 train, 70 test|task accuracy|

### Models

The main configuration uses gpt-4.1 as both agent and compressor, specifically `gpt-4.1-2025-04-14` via Azure at temperature 0.0 with seed 42. Other agents tested include gpt-4.1-mini, gpt-5-chat, and a distilled Qwen3-14B. The optimizer LLM is o3 by default, ablated against gpt-4.1 and gpt-5. Student compressors are Qwen3-14B, Qwen3-8B, and Phi-4, plus gpt-4.1-mini used off the shelf without distillation. The agent scaffold is ReAct throughout.

### Metrics

1. **Accuracy, EM, F1**, which are task-specific.
2. **Steps**, the average number of interaction steps per task.
3. **Peak tokens**, the maximum tokens in any single sequence over the trajectory, **excluding system prompts**. This is a proxy for KV-cache memory.
4. **Dependency**, the cumulative reliance of generated actions on prior context.

$$
\text{Dep} = \sum_{t \in [T]} \frac{\big(n_i^{(t)} + 2 n_o^{(t)}\big) \cdot n_o^{(t)}}{2}
$$

Here $$n_i^{(t)}$$ and $$n_o^{(t)}$$ are input and output tokens at step $$t$$. Intuitively, generating the $$k$$-th output token attends over roughly $$n_i + k$$ positions, so summing over generated positions gives a quadratic-flavored quantity. It is the area under the context-growth curve weighted by generation, or approximate cumulative attention FLOPs.

### Hyperparameters

Thresholds are $$T_{\text{hist}} = 4096$$ for AppWorld and OfficeBench and $$2048$$ for 8-objective QA, with $$T_{\text{obs}} = 1024$$ for AppWorld, $$512$$ for OfficeBench, and $$400$$ for QA. The last action-observation pair is always preserved, for ACON and all baselines equally.

Prompt optimization uses $$K = 5$$ candidates per update with the best selected on a training subset, and runs exactly one round of one $$\overline{\text{UT}}$$ plus one $$\overline{\text{CO}}$$.

Distillation uses LoRA with rank 16 and $$\alpha = 32$$, learning rate $$10^{-4}$$, 3 epochs, batch size 4, maximum sequence length 10,000, 5% linear warmup, weight decay 0.01, AdamW, on a single A100 80GB, with greedy decoding at inference. **No hyperparameter tuning** was performed, and the same configuration is used everywhere. One teacher sample is drawn per input.

### Baselines

1. **No Compression**, the full context, which is both the accuracy ceiling and the cost ceiling.
2. **FIFO**, keeping the last 5 turns, tuned to match ACON's average compression rate.
3. **Retrieval**, embedding turns with `text-embedding-3-large` and retrieving the 4 most similar to the current query plus the last turn.
4. **LLMLingua and LLMLingua-2**, extractive token-level compression at a 30% keep rate.
5. **Prompting**, a generic summarize instruction, lightly hand-engineered per compression type.

---

## 10. Results

### AppWorld with gpt-4.1 as agent and compressor, 168 test tasks

|Method|Acc $$\uparrow$$|Steps $$\downarrow$$|Peak ($$10^3$$) $$\downarrow$$|Dep ($$10^6$$) $$\downarrow$$|
|---|---|---|---|---|
|No compression|56.0|16.14|9.93|5.96|
|**History compression**|||||
|FIFO|45.8|28.48|6.73|5.69|
|Retrieval|27.4|33.17|8.39|6.68|
|LLMLingua|39.3|24.42|7.50|6.37|
|Prompting|43.5|24.01|6.93|5.29|
|ACON $$\overline{\text{UT}}$$|51.2|20.92|7.17|4.49|
|**ACON $$\overline{\text{UT}},\overline{\text{CO}}$$**|**56.5**|22.82|7.33|4.69|
|**Observation compression**|||||
|LLMLingua|32.1|18.16|8.17|6.01|
|Prompting|42.3|17.38|6.58|4.09|
|ACON $$\overline{\text{UT}}$$|47.0|16.67|7.62|5.08|
|ACON $$\overline{\text{UT}},\overline{\text{CO}}$$|53.6|18.12|7.43|4.93|

ACON $$\overline{\text{UT}},\overline{\text{CO}}$$ matches and nominally exceeds the uncompressed ceiling at **26% fewer peak tokens** and **21% lower dependency**. Every other compression method loses 10 to 29 accuracy points. Retrieval is catastrophic, which makes sense, because embedding similarity retrieves _topically related_ turns while state tracking needs _causally necessary_ turns, and those two sets barely overlap.

Also note the step counts. All baselines inflate steps massively, with FIFO at 28.5 and Retrieval at 33.2 versus 16.1 uncompressed, because the agent keeps rediscovering things it already knew. ACON holds steps to roughly 21 to 23. Preserved state means less re-exploration.

Broken down by difficulty for ACON $$\overline{\text{UT}},\overline{\text{CO}}$$ on history, easy goes from 80.7 uncompressed to 86.0, medium from 47.9 to 56.2, and **hard from 39.7 down to 30.2**. Compression helps on easy and medium tasks and still hurts on hard ones. The headline average conceals this.

### OfficeBench

|Method|Acc $$\uparrow$$|Steps $$\downarrow$$|Peak $$\downarrow$$|Dep $$\downarrow$$|
|---|---|---|---|---|
|No Compression|**76.84**|11.52|7.27|4.43|
|FIFO|67.37|12.26|**4.02**|2.64|
|Prompting (history)|71.58|10.13|4.40|**1.10**|
|ACON $$\overline{\text{UT}}$$ (history)|74.74|13.13|4.93|3.85|
|ACON $$\overline{\text{UT}},\overline{\text{CO}}$$ (history)|72.63|11.54|4.54|1.91|
|ACON $$\overline{\text{UT}}$$ (observation)|73.68|10.83|6.55|3.85|

Roughly a 32% peak reduction at 74.74 accuracy, about 2 points below the ceiling. The difficulty breakdown shows the gains concentrate on Level 3 three-app tasks, where ACON $$\overline{\text{UT}}$$ reaches 61.29 versus 54.84 uncompressed and 48.39 for FIFO and Prompting. The hardest coordination tasks benefit most, which is the opposite of the AppWorld pattern.

### 8-objective QA

|Method|EM $$\uparrow$$|F1 $$\uparrow$$|Peak $$\downarrow$$|Dep $$\downarrow$$|
|---|---|---|---|---|
|No compression|0.366|0.488|10.35|3.32|
|Prompting (history)|**0.376**|0.478|4.73|1.66|
|ACON $$\overline{\text{UT}}$$ (history)|0.373|**0.494**|4.71|1.57|
|ACON $$\overline{\text{UT}},\overline{\text{CO}}$$ (history)|0.335|0.458|**4.65**|**1.50**|
|ACON $$\overline{\text{UT}}$$ (observation)|0.364|0.475|4.97|1.28|

This is the best case for the method. F1 exceeds the uncompressed baseline at **54.5% fewer peak tokens** and **53% lower dependency**. This benchmark is the most compressible because retrieved Wikipedia passages are mostly noise once you have extracted the answer span.

### Compressor distillation

Students trained on the $$\overline{\text{UT}}$$-optimized teacher's outputs, evaluated with gpt-4.1 as the agent under history compression.

|Compressor|AppWorld|OfficeBench|8-obj QA (EM)|
|---|---|---|---|
|gpt-4.1 teacher|about 51.2|74.74|0.373|
|Qwen3-14B (distilled)|50.00|70.53|0.371|
|Qwen3-8B (distilled)|47.00|75.79|0.386|
|Phi-4 (distilled)|44.60|73.68|0.370|
|gpt-4.1-mini (no distill)|42.30|77.89|0.379|

Qwen3-14B retains 97.7% of teacher accuracy on AppWorld, and several students match or beat the teacher on the easier benchmarks. The claim of over 95% retention holds.

The important ablation is in Table 13, comparing distillation from **naive-prompt** teacher outputs against **optimized-guideline** teacher outputs with the same student and the same recipe.

| |History|Observation|
|---|---|---|
|Fine-tune on naive prompt (Qwen3-14B)|44.6|40.5|
|ACON distilled (Qwen3-14B)|**50.0**|**56.5**|

That is plus 5.4 and plus 16 points. The guideline optimization is doing the work, not the distillation.

### ACON as an equalizer for small agents

With Qwen3-14B **as the agent**, distilled from gpt-4.1 trajectories and paired with a distilled compressor, AppWorld goes from 25.6 to 33.9 for a **32.4% relative gain**, 8-objective QA goes from 0.158 to 0.23 EM for a **45.6% gain**, and OfficeBench goes from roughly 41 to roughly 50 for about **20%**.

These gains come with _lower_ peak tokens, not higher. Small models are disproportionately hurt by long contexts, so removing the noise buys more than it costs. This is the paper's most interesting empirical claim, namely that for small agents compression is not a memory-accuracy tradeoff at all but a free lunch.

The same pattern holds with gpt-4.1-mini on AppWorld, which goes from 35.7 to 42.3 under ACON $$\overline{\text{UT}}$$ history compression with peak tokens dropping from 8.55 to 6.51.

### WebVoyager, generalization to a different environment type

|Method|Acc $$\uparrow$$|Peak $$\downarrow$$|Dep $$\downarrow$$|
|---|---|---|---|
|No compression|35.7|13.28|2.53|
|Prompting (history)|42.9|7.97|1.23|
|ACON (history)|**48.6**|8.04|1.19|
|ACON (observation)|47.1|4.19|0.80|

Plus 12.9 points over uncompressed on verbose accessibility-tree observations. This is the strongest relative result in the paper and the most different environment.

### Costs

Optimization costs under $2 per benchmark for the o3 guideline loop. Trajectory collection is the real expense at roughly $20 for AppWorld's 90 training tasks times 2 rollouts, and it is a one-time cost that can reuse existing logs.

Compressor inference per example drops from $0.045 with gpt-4.1 to $0.014 with gpt-4.1-mini, a 69.2% reduction, and then to $0.0004 with distilled Qwen3-14B, a **99.1% reduction**.

End to end on AppWorld, using a gpt-4.1 agent with a Qwen3-14B compressor for the cost measurement and Qwen3-14B for both in the latency measurement on a local A100 with no network,

|Method|API cost $$\downarrow$$|Latency $$\downarrow$$|
|---|---|---|
|No Compression|$0.331|73.24s|
|ACON (history)|$0.285|**87.68s**|
|ACON (observation)|$0.272|**101.92s**|

Cost drops 14 to 18%. **Latency rises 20 to 39%.** The paper reports this honestly.

### Analysis and ablations

**Optimizer choice** on AppWorld history compression.

|Optimizer|Contrastive|Acc|
|---|---|---|
|o3|yes|51.2|
|o3|no (failures only)|50.6|
|gpt-4.1|yes|47.6|
|gpt-5|yes|50.6|

Optimizer strength matters more, at minus 3.6 for gpt-4.1, than the contrastive signal, at minus 0.6. That is an awkward result for the paper's framing, since contrastive feedback is presented as the central contribution. gpt-5 underperforming o3 goes unexplained.

**Thresholds.** Sweeping $$T_{\text{hist}} \in \{2048, 4096, 8192\}$$ and $$T_{\text{obs}} \in \{512, 1024, 2048\}$$, small thresholds trigger more compressor calls, save more tokens, and degrade accuracy, with a history threshold of 2048 dropping to roughly 37.5%. Large thresholds preserve accuracy at higher cost. Moderate values of 4096 and 1024 sit at the knee. Compression frequency scales roughly linearly with trajectory length, reaching about 10 calls over a 40 to 49 step task at threshold 2048.

**Extra optimization rounds hurt.** ACON $$\overline{\text{UT}},\overline{\text{CO}},\overline{\text{UT}}$$ scores 47.0 versus 56.5 for $$\overline{\text{UT}},\overline{\text{CO}}$$. One round is it.

**History and observation compression together is worse than either alone.** Combined gets the lowest peak at 5.85 to 5.90, but accuracy falls to 45.8 versus 56.5 history-only and 53.6 observation-only. The two operators are not composable.

**When to use which variant.** Verbose, redundant environments such as AppWorld, with duplicated API dumps and distractors, favor $$\overline{\text{UT}},\overline{\text{CO}}$$, because aggressive pruning improves both efficiency and focus. High-fidelity information-seeking environments such as OfficeBench and QA favor $$\overline{\text{UT}}$$, because over-compression removes subtle facts needed for the final answer.
