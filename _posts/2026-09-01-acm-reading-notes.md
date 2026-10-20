---
layout: post
title: 'Reading Notes of ACM: Agentic Context Management for Long Horizon Tasks'
date: 2026-09-01 23:41:57
description: 'I like the idea of insertion / deletion decisions of context management by on policy distillation, but there should be more analysis here. Also, the comparison is not fair. The compute is not saved. The paper just cares about compression, but not compute.'
tags: nlp llm-agents
categories: ReadingNotes
thumbnail: assets/img/blogs/acm.png
toc:
  sidebar: left
related_posts: false
---

I like the idea of insertion / deletion decisions of context management by on policy distillation, but there should be more analysis here. Also, the comparison is not fair. The compute is not saved. The paper just cares about compression, but not compute.

<!--more-->

> **Reference**: https://arxiv.org/abs/2607.23809

## 1. Motivation

Long growing context is verbose and noisy, leading to context limit hit and worse performance.

Three prior directions and why each falls short:

- **Long-context pretraining.** Extends the nominal window but shows measurable degradation inside it.
- **Hybrid attention** (Mamba-style, Jamba). Cheaper per token, still bounded by the same window.
- **Context compression** (ReSum, ACON, Claude auto-compaction). Actually shrinks the working set, but suffers two defects:
    1. **Lossy.** The compressed messages are deleted. If the summary drops a detail that turns out to matter at turn 60, it is gone.
    2. **Externally triggered.** Compression fires when a hand-written monitor sees usage cross a threshold, for example 90% of the window. That instant has nothing to do with where the agent is in its reasoning. It might fire mid-hypothesis, or fire far too late after the noise has already degraded 40 turns of decisions.

ACM: Makes context management an **agentic decision** rather than an externally-triggered one, allowing the agent to decide when and how to compress its context based on its reasoning state. Use on policy distillation to train the agent to learn when to invoke context management and when not to from super teacher models.

## 2. The ACM Framework

### 2.1 Base formulation

Let $$s$$ be the system prompt, $$a_t$$ an agent action (reasoning content plus tool calls), and $$o_t$$ the environment response at turn $$t$$. The accumulated history is

$$
H_t = \{s, (a_1, o_1), \ldots, (a_{t-1}, o_{t-1})\}
$$

and the agent policy $$\pi_\theta$$ together with the environment $$\pi_\gamma$$ generate

$$
a_t \sim \pi_\theta(\cdot \mid H_t), \qquad o_t \sim \pi_\gamma(\cdot \mid H_t \,; a_t)
$$

The episode ends when the agent emits a finish action $$a_T$$, or when $$\vert H_t\vert  \geq L$$ where $$L$$ is the context limit (131,072 tokens in all experiments).

### 2.2 The Summary Agent baseline

An external monitor triggers $$a_{\text{sum}}$$ once $$\vert H_t\vert $$ crosses a threshold. The environment returns $$o_{\text{sum}} \sim \pi_\gamma(\cdot \mid H_t \,; a_{\text{sum}})$$ and the history collapses to

$$
H' = \{s,\, o_{\text{sum}}\}
$$

Everything else is destroyed. This is the lossy, externally-triggered design ACM replaces.

### 2.3 The ACM Agent

Two tools are added to whatever task tools the benchmark provides.

**`manage_context`** takes no arguments. Let $$b_{k-1}$$ be the turn of the previous compression (with $$b_0 = 0$$) and $$b_k$$ the turn of the current call. The compressed segment is

$$
C_k = \{(a_i, o_i)\}_{i = b_{k-1}+1}^{\,b_k - 1}
$$

A summarizer LLM $$\pi_\sigma$$ produces $$z_k \sim \pi_\sigma(\cdot \mid C_k)$$, capped at 4096 tokens, structured as a _Knowledge state_ section (facts with docid citations, live candidates, eliminated hypotheses, open sub-questions) and a _Thoughts_ section (distilled recent reasoning plus one concrete next step). The raw block is written to disk under an identifier,

$$
\mathcal{M} \leftarrow \mathcal{M} \cup \{\,k \mapsto C_k\,\}
$$

and the working context becomes

$$
H_{b_k + 1} = \{s,\, q,\, z_1, \ldots, z_{k-1},\, a_{b_k},\, z_k\}
$$

Note the system prompt and the original question $$q$$ are always preserved, and prior summaries remain in context. Only the raw span since the last boundary is swapped out.

**`query_memory(summary_id, query)`** hands the archived block plus a natural-language query to a querier LLM $$\pi_\rho$$, which returns

$$
o_{\text{qm}} \sim \pi_\rho(\cdot \mid \mathcal{M}[k],\, \text{query})
$$

formatted as compact bullets under three headings (relevant findings, dead ends, open and unresolved), with identifiers preserved verbatim. This is targeted recall, not reloading.

Define the compression ratio of a call as $$\rho_k = \vert z_k\vert  \,/\, \vert C_k\vert $$. In the paper's case study $$\vert C_3\vert  = 158\text{K}$$ and $$\vert z_3\vert  = 2.3\text{K}$$, giving $$\rho_3 \approx 0.015$$.

### 2.4 Why this extends the exploration horizon (derivation)

The paper asserts this but does not derive it. Let $$\bar{c}$$ be the average token cost of one turn.

Under ReAct the history is monotone, $$\vert H_T\vert  \approx \vert s\vert  + T\bar{c}$$, so the episode dies at

$$
T_{\text{ReAct}} \approx \frac{L - \vert s\vert }{\bar{c}}
$$

Under ACM, compressing every $$m$$ turns at ratio $$\rho$$, the persistent floor after $$k$$ cycles is $$\vert s\vert  + k m \bar{c}\rho$$ and the peak inside a cycle is that floor plus $$m\bar{c}$$. Termination requires

$$
\vert s\vert  + k m \bar{c}\rho + m\bar{c} \geq L \quad\Longrightarrow\quad k_{\max} \approx \frac{L - \vert s\vert  - m\bar{c}}{m\bar{c}\rho}
$$

so total turns $$T_{\text{ACM}} = m\,k_{\max}$$ and

$$
\frac{T_{\text{ACM}}}{T_{\text{ReAct}}} \approx \frac{1}{\rho} \quad\text{when } m\bar{c} \ll L
$$

Two consequences follow directly. First, the horizon gain is governed by $$\rho$$, not by $$L$$, so ACM buys reach that a larger window cannot buy cheaply. Second, peak token usage is bounded by floor plus $$m\bar{c}$$ rather than by $$L$$, which is why the paper claims reduced KV-cache pressure. The observed gain in the case study is only about $$1.7\times$$ rather than $$1/\rho \approx 66\times$$, because summaries accumulate monotonically and `query_memory` results add tokens back.

### 2.5 The two claimed properties

1. **Lossless.** Raw messages are archived, not deleted, and are reachable at any later turn.
2. **Agent-initiated.** Compression can fire at any point, so it tracks the reasoning state rather than a token counter, and it can fire _before_ the history reaches its maximum, which is what relieves peak pressure.

Table 1 in the paper positions ACM as the only method satisfying all of {compact, trainable, lossless, agent-initiated, open data}.

## 3. Training Data Generation

### 3.1 Why training is needed at all

Figure 4 is the key evidence. Given the ACM toolset, GPT-5.5 calls `manage_context` 0.1 times and `query_memory` 0.0 times per question. Frontier models simply do not use the tools. Handing an agent a memory API is not enough. The timing policy has to be taught.

This also rules out the obvious data recipe. Distilling successful teacher trajectories fails because a strong teacher solves most problems without ever needing compression, so the demonstrations contain almost no positive examples of the target behavior. The data must come from _student_ rollouts on problems the student cannot solve.

### 3.2 Dual-constraint teacher annotation

**Phase 1, student rollout.** The student runs each task twice, once without ACM tools producing $$H^-$$, and once with them producing $$H^+$$. $$H^-$$ captures ordinary exploration, $$H^+$$ captures untrained tool usage.

**Phase 2, teacher annotation.** A teacher receives prompt $$P^+$$ or $$P^-$$, the matching trajectory, and the gold answer $$A^*$$. The two constraints are applied crosswise.

- **Injection on $$H^-$$** (given $$P^+$$, where compression _should_ have happened). The teacher scans for the earliest turn showing redundant queries, cyclic exploration, or enough accumulated context to warrant compression, and replaces $$a_t$$ with a `manage_context` or `query_memory` call plus a first-person justification.

    Worked example from Figure 2. Original action reads _"I should continue to search for ... "_ followed by `<tool>search</tool>`. The teacher rewrites it to _"I notice my searches are starting to loop around the same broad clue set, road accident ... I should compress what I have"_ followed by `<tool>manage_context</tool>`. The student was about to issue its fifth near-duplicate query. The label converts that into a compression.

- **Refinement on $$H^+$$** (given $$P^-$$, where compression should _not_ have happened). The teacher finds premature calls and replaces them with a productive alternative, under a strict global priority: **commit** (the in-context evidence already answers the question), then **replace with search** (a novel query not already tried), then **replace with get_document** (fetch a docid already visible in a snippet).

    Worked example. Original action reads _"I should continue to compress the history..."_ followed by `<tool>manage_context</tool>`. The teacher rewrites it to _"Instead of compressing now, I should perform a targeted search for the book title, as it can directly address the residence-university aspect"_ followed by `<tool>search</tool>`. The student was compressing reflexively rather than making progress.

In both cases the student then **resumes the rollout from $$a'_t$$**, so the downstream trajectory is on-policy given the corrected decision.

### 3.3 On-policy distillation loss

A stronger teacher from the same family scores each student-generated assistant token with a soft next-token distribution, truncated to the top $$K = 20$$ tokens $$\mathcal{V}$$ and renormalized:

$$
p_T(v \mid s\,; h_{<t}) = \frac{\tilde{p}_T(v \mid s\,; h_{<t})}{\sum_{v' \in \mathcal{V}} \tilde{p}_T(v' \mid s\,; h_{<t})}, \quad v \in \mathcal{V}
$$

The objective is

$$
\mathcal{L}_{\text{ACM}}(\theta) = -\,\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t \in \mathcal{T}^a(\tau)} \sum_{v \in \mathcal{V}} p_T(v \mid s\,; h_{<t}) \, \log \pi_\theta(v \mid s\,; h_{<t})\right]
$$

where $$\tau$$ is a trajectory sampled from the student policy, $$\mathcal{T}^a(\tau)$$ is the set of assistant-token positions, and $$h_{<t} = (a_1, o_1, \ldots, a_{t-1}, o_{t-1})$$ is the interleaved history. System prompt, user input, and tool-output tokens are masked out.

**Derivation of what this optimizes.** Per position, the forward KL from teacher to student is

$$
\mathrm{KL}\!\left(p_T \,\Vert \, \pi_\theta\right) = \sum_{v \in \mathcal{V}} p_T(v)\log p_T(v) \;-\; \sum_{v \in \mathcal{V}} p_T(v)\log \pi_\theta(v) = -H(p_T) + \mathrm{CE}(p_T, \pi_\theta)
$$

Since $$H(p_T)$$ does not depend on $$\theta$$,

$$
\arg\min_\theta \mathcal{L}_{\text{ACM}}(\theta) = \arg\min_\theta \; \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t \in \mathcal{T}^a(\tau)} \mathrm{KL}\!\left(p_T(\cdot \mid s\,; h_{<t}) \,\Vert \, \pi_\theta(\cdot \mid s\,; h_{<t})\right)\right]
$$

So the loss is exactly forward-KL distillation, evaluated at states the student itself visits. Two design points matter. Forward KL is mode-covering, so the student is penalized for putting zero mass anywhere the teacher puts mass, which suits a decision with genuinely multiple acceptable continuations (search, compress, commit). And the states are on-policy, which is what fixes the compounding-error problem that plain behavior cloning on teacher trajectories would have. The gradient is _not_ propagated through the sampling distribution $$\pi_\theta$$ that produced $$\tau$$, so this is distillation on a periodically refreshed on-policy dataset, not a policy gradient.

### 3.4 Quality filtering

1. **Rejection sampling.** Keep only tasks where the student failed _all_ trials, so the model learns from the teacher on genuinely hard problems rather than on ones it already solves.
2. **Content filters.** The teacher's reasoning trace must justify the decision using structural cues only (repeated query keywords, token pressure, no new docids in the last $$N$$ turns) and must never leak $$A^*$$, cite an unretrieved docid, or use coach-framing vocabulary (`feedback`, `advised`, `external`, `told me`). Rationales are length-bounded to roughly 200 to 1500 characters and written strictly first-person.
3. **Stabilizer.** Original student rollouts are resampled and mixed in, STaR-style self-distillation.

## 4. Pseudocode

### 4.1 Inference

```
Input: system prompt s, question q, policy pi_theta,
       summarizer pi_sigma, querier pi_rho,
       context limit L, task tools T_task

H <- [s, q]          # working context, short-term memory
M <- {}              # external memory, summary_id -> raw message block
B <- []              # messages since last compression boundary
k <- 0               # summary counter

loop t = 1, 2, ...
    H_aug <- H + token_usage_hint(len(H), L)
    a_t   <- sample pi_theta( . | H_aug)

    if a_t is FINISH
        return extract_answer(a_t)

    else if a_t is manage_context           # no arguments
        k    <- k + 1
        z_k  <- pi_sigma(B)                 # <= 4096 tok, knowledge state + thoughts
        M[k] <- B                           # lossless offload to disk
        H    <- drop(H, B) + [a_t, "[summary_id: k] " + z_k]
        B    <- []

    else if a_t is query_memory(id, qry)
        o_t <- pi_rho(M[id], qry)           # targeted recall from raw messages
        H   <- H + [a_t, o_t]
        B   <- B + [a_t, o_t]

    else                                    # search / get_document / bash / edit
        o_t <- env(H, a_t)
        H   <- H + [a_t, o_t]
        B   <- B + [a_t, o_t]

    if len(H) >= L
        return FAIL_context_overflow
```

### 4.2 Training

```
Input: tasks D, student pi_S, teacher pi_T, prompts P_plus / P_minus, gold A*

D_sft <- []
for x in D
    H_minus <- rollout(pi_S, x, tools = T_task)              # no ACM tools
    H_plus  <- rollout(pi_S, x, tools = T_task + T_acm)      # with ACM tools

    if solved(pi_S, x, all_trials)
        continue                                             # rejection sampling

    # Constraint 1, inject ACM where it was missing
    (t, a_new) <- pi_T(P_plus, H_minus, A*)
    if a_new != NO_ACTION and passes_content_filter(a_new, A*)
        tau <- H_minus[:t] + [a_new] + resume_rollout(pi_S, from = a_new)
        D_sft.append(tau)

    # Constraint 2, remove ACM where it was premature
    # teacher priority order is commit > replace_with_search > replace_with_get_document
    (t, a_new) <- pi_T(P_minus, H_plus, A*)
    if a_new != NO_REPLACEMENT and passes_content_filter(a_new, A*)
        tau <- H_plus[:t] + [a_new] + resume_rollout(pi_S, from = a_new)
        D_sft.append(tau)

D_sft <- D_sft + resample(original student rollouts)          # stabilizer

for epoch in 1..3
    for tau in D_sft
        loss <- 0
        for t in assistant_token_positions(tau)
            V   <- top_k(pi_T( . | s, h_lt_t), K = 20)
            p_T <- renormalize(pi_T over V)
            loss <- loss - sum over v in V of p_T(v) * log pi_theta(v | s, h_lt_t)
        theta <- theta - eta * grad(loss)     # prompt and tool-output tokens masked
```

## 5. Experimental Setup

**Benchmarks.**

|Benchmark|Role|Split|Tools|
|---|---|---|---|
|BrowseComp-Plus|in-domain search|680 train, 150 eval|`search` over a fixed local corpus, `get_document`|
|DeepSearchQA|out-of-domain search, eval only|17 domains, single-answer and set-answer|live web `search`, `open`|
|SWE-Bench Verified|coding|trained on SWE-Gym|`execute_bash`, `str_replace_editor`, `submit_patch` in a per-instance Modal sandbox at `/testbed`|

Simple tasks are deliberately excluded, since they finish before context pressure exists.

**Models.** Student policy, summarizer, and querier are all Qwen3.5-9B. Teacher is Qwen3.5-397B-A17B. Frontier reference points are Qwen3.5-397B-A17B and Gemini3-Flash. GPT-5.5 is used both as a probe of untrained tool usage and as a distillation source in the ablation.

**Hyperparameters (all that are reported).** Top-$$K = 20$$ for teacher distributions, three distillation epochs, 131,072-token context cap, decoding settings held fixed across all methods. Learning rate, batch size, and optimizer are not stated.

**Metrics.** Pass@1 accuracy, average tool calls per episode, average peak token count $$\max_t \vert H_t\vert $$ per episode. Pass@4 as a proxy for the capability boundary and $$\text{Pass}^4$$ (all four independent trials correct) as a consistency measure. Grading uses the simple-evals judge template for both search benchmarks.

**Baselines.** ReAct (no context management), ReSum and ACON (threshold-triggered summary agents), ACE (cross-task memory playbook). All three are prompting-only, which is why they were chosen. They can be dropped onto the identical Qwen3.5-9B backbone with the same tools and decoding, so any delta is attributable to the mechanism. Mem1, SUPO, and AgentFold are excluded because they require retraining or have unreleased pipelines.

## 6. Results

### 6.1 Main table

|Method|BCP Pass@1|BCP Tools|BCP Peak|DSQA Pass@1|DSQA Tools|DSQA Peak|SWE Pass@1|SWE Tools|SWE Peak|
|---|---|---|---|---|---|---|---|---|---|
|Qwen3.5-397B-A17B|0.653|15.6|51K|0.710|28.3|47K|0.682|58.9|38K|
|Gemini3-Flash|0.733|22.9|72K|0.619|54.3|121K|0.732|66.7|80K|
|ReAct (9B)|0.570|19.5|63K|0.367|47.4|46K|0.489|74.7|59K|
|ReSum|0.608|24.7|68K|0.371|48.6|79K|0.475|75.2|61K|
|ACON|0.614|28.2|65K|0.380|51.3|54K|0.480|76.1|57K|
|ACE|0.589|19.8|71K|0.352|48.2|70K|0.494|75.6|65K|
|ACM Base|0.635|30.8|59K|0.405|88.7|42K|0.508|77.6|46K|
|ACM Post-Trained|**0.727**|46.2|54K|**0.425**|58.8|41K|**0.530**|79.3|50K|

Relative gain over ReAct: $$+27.5\%$$ on BrowseComp-Plus, $$+15.8\%$$ on DeepSearchQA, $$+8.4\%$$ on SWE-Bench Verified.

Three things are worth separating.

**The framework alone already helps.** ACM Base, with no training at all, beats every baseline on all three benchmarks. Agent-initiated compression is doing real work before any distillation.

**Post-training roughly triples the gain on search.** $$0.635 \to 0.727$$ on BCP. The post-trained 9B model reaches 0.727 against Gemini3-Flash's 0.733 and beats the 397B model of its own family (0.653), which is about $$44\times$$ larger. This near-parity is specific to the in-domain benchmark. On DSQA (0.425 vs 0.710) and SWE (0.530 vs 0.732) the frontier gap remains wide.

**Out-of-domain transfer is real but smaller.** DeepSearchQA is never trained on and uses live web search rather than a fixed corpus, so $$0.367 \to 0.425$$ indicates the learned timing policy is not corpus-specific.

**Peak tokens.** Against ReAct the reduction is 11% to 15%. The paper's "around 20%" figure is supported against the summary agents (ReSum on BCP, $$68\text{K} \to 54\text{K}$$, is $$-21\%$$, and on DSQA $$79\text{K} \to 41\text{K}$$) or for ACM Base on SWE ($$59\text{K} \to 46\text{K}$$, $$-22\%$$). Note that summary agents have _higher_ peak usage than plain ReAct on several cells, exactly as the framework predicts. A threshold trigger by construction lets the context reach the threshold before acting.

**Tool calls rise with accuracy.** ACM Post-Trained makes 46.2 calls on BCP against ReAct's 19.5. The interpretation offered is that small models substitute exploration for raw capability, and context management is what makes extended exploration affordable.

### 6.2 Behavior analysis

**Context growth (Figure 3).** ACM trajectories show a sawtooth well below the 128K ceiling, with compression points scattered across the whole range rather than clustered near the limit. ReAct trajectories climb monotonically and terminate near turn 60. ACM trajectories run past turn 100.

**Tool decomposition (Figure 4), per question on BCP.**

|Agent|search|get_document|manage_context|query_memory|
|---|---|---|---|---|
|ACM + GPT-5.5 ($$n = 1 \times 150$$)|18.9|4.2|0.1|0.0|
|ACM + Qwen3.5-9B ($$n = 4 \times 150$$)|24.9|4.1|2.1|0.6|
|ACM Post-Trained ($$n = 4 \times 150$$)|28.7|6.5|6.8|1.3|

The post-trained model leads on _every_ column, not just the memory tools. Compression is not competing with exploration for budget, it is funding it.

**Pass@K (Figure 5), BrowseComp-Plus.**

|Setting|Pass@4|Pass@1|$$\text{Pass}^4$$|Pass@4 minus $$\text{Pass}^4$$|
|---|---|---|---|---|
|ReAct|73.5|57.0|34.1|39.4|
|ACM Agent|78.8|63.5|44.0|34.8|
|Post-Trained Ep.1|80.0|67.7|50.7|29.3|
|Post-Trained Ep.2|82.0|69.8|56.0|26.0|
|Post-Trained Ep.3|82.0|72.7|59.3|22.7|

This is the most informative result in the paper. Pass@4 moves $$+8.5$$ points while $$\text{Pass}^4$$ moves $$+25.2$$. ACM barely expands the set of solvable problems. What it does is make the solutions _reliable_. The natural reading is that accumulated context noise is a variance source, and cleaning the working context removes it. Gains are still increasing at epoch 3, with Pass@4 already saturated.

**Distillation ablation (Table 3), all on Qwen3.5-9B.**

|Config|BCP|DSQA|SWE|
|---|---|---|---|
|Base|0.635|0.405|0.508|
|+ GPT-5.5 distill|0.623|0.381|0.542|
|+ ACM data|0.727|**0.425**|0.530|
|+ Both|**0.734**|0.413|**0.564**|

Generic distillation from a strong teacher actively _hurts_ on search (below the untrained ACM base on both search benchmarks) while helping on coding. ACM data alone is the only configuration that improves everywhere. Combining is best on two of three. The reading offered is that the two signals are complementary, general problem-solving from distillation and timing from ACM data.

**Case study (Figure 6), BCP qid 347.** A five-constraint multi-hop question (restaurant in the acknowledgments of a UC dissertation, author with a B.Tech from IIT BHU and a UCLA master's, co-authored papers in 2020 and 2020 to 2022, restaurant founded 1980 to 1988). Gold answer California Pizza Kitchen. The base model fails 0 for 4.

The trajectory illustrates every mechanism at once. At turn 10 the model reports 54,932 tokens and compresses 21 messages into summary_1. At turn 18 it compresses again at 40,982 tokens, explicitly reasoning about approaching half of its budget. At turn 49 a `query_memory` probe surfaces a lead it then chooses to verify rather than commit to. At turn 58, at 87% usage with 16,606 tokens left, it folds 78 messages whose raw form is 158K into a 2.3K summary. At turns 67 and 79 it confirms two of the five constraints via `get_document`. At turn 82, sitting at 20,325 tokens, it commits.

Totals: 63 searches, 9 document fetches, 7 compressions, 5 memory queries, 83 turns. Peak actual context 98K against a raw history of 222K. Without compression the trajectory would have crossed the 128K ceiling at turn 47, roughly halfway. The `query_memory` calls interleaved with `manage_context` are the direct evidence that the lossless archive is actually being re-read rather than being a write-only sink.

**Exploration diversity (Appendix C).** Consecutive queries $$(q_{t-1}, q_t)$$ are embedded with Qwen3-Embedding-8B, cosine similarity $$s_t$$ is computed, and a pivot is declared when $$s_t < \tau$$. The running pivot fraction is

$$
f_t = \frac{1}{t}\sum_{i=1}^{t} \mathbb{1}\!\left[s_i < \tau\right]
$$

which is bounded in $$[0,1]$$ and converges to the long-run pivot rate regardless of trajectory length. The horizontal axis normalizes token position to $$[0,1]$$ so that short ReAct trajectories do not create survivorship bias. Post-trained ACM holds a higher pivot fraction than both ACM and ReAct at all four thresholds $$\tau \in \{0.3, 0.4, 0.5, 0.6\}$$. Notably, plain ACM tracks ReAct closely, so the tools alone do not induce exploration. Only the training signal does.

**Small models (Appendix D).** Qwen3-4B-thinking collapses every BrowseComp-Plus rollout to exactly two turns with 1.2 searches and 3.4% accuracy, against 19.4 turns, 16.2 searches, and 57.3% for the 9B model. On qid 124 the 4B model uses 23K of 131K tokens, explicitly writes that it could do more searches, then in the next sentence hallucinates _"since I can't do real searches, I have to think"_ and guesses. The failure is a long-horizon planning failure, not a context-budget failure, so the memory tools never get a chance to matter.

## 7. Limitations

### From the paper

1. ACM presupposes a base model with strong long-horizon reasoning and tool use.
2. No prior context-compression baseline had been evaluated on these three benchmarks. Implementation differences may exist.

### AI assessment that I agree

**The "lossless" claim overstates what the agent experiences.** The raw bytes are preserved, but access runs through a three-step chain: the agent must remember the summary exists, must guess the right `summary_id`, and the querier LLM must extract the right span. If the summary text drops the pointer that would have made the agent think to query summary 2, the archived content is unreachable in practice. This is _recoverable-in-principle_, not lossless. A retrieval-recall measurement (how often does the needed fact survive a compress-then-query round trip) is the obvious missing experiment and is absent entirely.

**Only when, not how (much).** `manage_context` takes no arguments, so the system picks the compression span. "How much" is decided by the system (everything before the last compression). The abstract's claim that the agent decides "when and how" is only half true. Also it still leads to growing context because summaries are not cleaned up.

**The headline comparison is confounded.** ACM Post-Trained is a trained policy. ReSum, ACON, and ACE are prompting-only by explicit design choice. I can do ReSum/ACON/ACE with on policy distillation.

**Compute is measured in the wrong currency.** Peak tokens is reported. Total tokens is not. Peak token is not about cost.
