---
layout: post
title: 'Reading Notes of Context Language Models'
date: 2026-10-19 23:00:00
description: 'TL;DR: The LLM can freely treat its context as a file to edit it with code, so the model decides what to keep, compress, or delete.'
tags: nlp llm-agents
categories: ReadingNotes
thumbnail: assets/img/blogs/clm.png
toc:
  sidebar: left
related_posts: false
---

TL;DR: The LLM can freely treat its context as a file to edit it with code, so the model decides what to keep, compress, or delete.

<!--more-->

> **Reference**: https://arxiv.org/pdf/2609.37725

## 1. Summary

Language-model agents struggle on long tasks because their context—the text the model rereads at every step—only grows until it overflows. The paper introduces Context Language Models (CLMs): the context is mirrored into a file the model freely edits with code, so the model decides what to keep, compress, or delete. (1) Used as-is, CLMs beat fixed compression rules with less compute (web-research benchmark BrowseComp-Plus, Qwen3.6-27B: +11.4% relative accuracy, −21.5% compute); (2) the skill is learnable—trial-and-reward training lifts Qwen3.5-9B from 28.8% to 42.5%; (3) Suffix Cache Reuse, which recycles stored computation after edits, cuts server compute 35% at equal accuracy.

---

## 2. Motivation

Long agent tasks, like hours of coding, produce far more text than a model's context window (its maximum readable length). Existing systems let the harness (the fixed program around the model) summarize at preset lengths, or give the model a few predefined tools (summarize, move to disk, read back). Both hard-code human guesses, so they lose details, cannot update text in place, and cannot remove text already read. Following the Bitter Lesson (general learned methods beat hand-designed ones), the paper grants unrestricted editing. One gap: it argues for full freedom by analogy and never tests partially restricted editing.

---

## 3. Math

### (a) Prerequisites

- **Token**: a word piece; models read and write tokens (1K tokens ≈ 750 English words).
- **Language model (LM)** $$\pi_\theta$$: a neural network with weights $$\theta$$ that reads a token sequence and generates new tokens.
- **Context** $$c$$: the whole token sequence the LM reads in one call (instructions, past actions, tool outputs); $$\lvert c\rvert$$ is its length.
- **Context budget** $$B$$: the largest allowed $$\lvert c\rvert$$; mostly $$B = 32{,}768$$ ("32K") tokens here.
- **Agent, turn, harness**: an agent is an LM acting in a loop. At turn $$t$$ it reads $$c_t$$, writes $$y_t$$ (reasoning plus a shell command), and the environment returns an observation $$o_t$$ (the command's output). The harness is the program that runs this loop and assembles each context.
- **Compaction / offloading / retrieval**: replacing old context by a shorter summary / moving text from the context to a disk file / reading it back.
- **Prefill and decode**: prefill processes the prompt tokens in one parallel pass; decode generates new tokens one at a time.
- **KV cache and prefix caching**: during prefill the model stores, for every token, a "key" and a "value" vector (the KV cache) that later tokens look back at. Prefix caching reuses the stored KV cache for the longest leading stretch of a new prompt that is identical to an earlier prompt; every token after the first difference is recomputed ("re-prefilled").
- **Attention**: the operation in which each token looks at all earlier tokens; its cost grows with the number of token pairs, i.e., quadratically with length.
- **RoPE (rotary position encoding)**: the model marks a token's position by rotating its key vector by an angle proportional to that position.
- **Linear attention (Gated DeltaNet)**: a layer type that compresses all past tokens into one fixed-size state updated token by token, instead of storing per-token keys and values. Qwen3.6-27B uses it in 48 of its 64 layers; the other 16 are full-attention layers.
- **FLOPs / PFLOPs**: floating-point operations, a hardware-independent count of compute; 1 PFLOP $$=10^{15}$$ FLOPs.
- **Zero-shot**: used as-is, without task-specific training.
- **Skill**: a natural-language instruction document placed in the context.
- **Reinforcement learning (RL)**: training by sampling attempts and making highly rewarded attempts more likely.

| Symbol | Meaning |
|---|---|
| $$t$$ | turn index, $$t=1,\dots,T$$ |
| $$c_t$$ | context at turn $$t$$ |
| $$y_t,\ o_t$$ | model output and environment observation at turn $$t$$ |
| $$\oplus$$ | concatenation (append) |
| $$B$$ | context budget in tokens |
| $$\pi_\theta$$ | the LM with weights $$\theta$$ |
| $$s$$ | skill document |
| $$r$$ | task reward or score |

### (b) Problem setup

Every setting shares one loop: a task $$x$$ arrives; for $$t=1,\dots,T$$ the model reads $$c_t$$, writes $$y_t \sim \pi_\theta(\cdot \mid c_t)$$, and receives $$o_t$$, with $$\lvert c_t\rvert \le B$$ at every call. The goal is a high final score $$r$$ at low total compute. The settings differ in task and reference:

| Setting | Input | What is scored | Reference |
|---|---|---|---|
| Coding and deep research (one agent, ≤100 turns) | a terminal task, or a hard question plus a search tool over a fixed corpus | pass rate; answer accuracy | the task's test script; an LLM judge holding the gold answer |
| Open discovery (hours to a day) | an optimization problem or a code repository plus an evaluator | best score reached | automatic evaluator or benchmark judge |
| Multi-agent | $$N$$ agents, each with its own context $$c^{(1)},\dots,c^{(N)}$$: an agent swarm (peers working in parallel) or subagents (helpers a main agent launches and closes) | joint outcome | held-out benchmarks |
| Diagnostic (ContextBench) | a forced stream of operations $$u_1,\dots,u_n$$, each arriving as a new message | what survives in the final context (files on disk earn no credit) | answers known to the generator |
| Learning | the same tasks; change the skill $$s$$ (weights fixed) or the weights $$\theta$$ | score and compute on held-out data | held-out split |
| Serving | recorded CLM workloads | server compute at equal accuracy | a standard serving engine |

ContextBench, the paper's diagnostic, isolates context management from knowledge and reasoning with four synthetic tasks. **Needle Retention**: keep 2–8 "needle" lines per ≈4K-token chunk verbatim and discard 140 filler lines. **Sudoku Sketchpad**: maintain a 16×16 board while single moves stream in, reproducing the exact board after each move. **KV Store**: absorb batches of 100 records `SET key = <24 random words>`, then answer 24 `GET key` queries exactly. **Log Triage**: absorb batches of service log lines, then answer 24 lookup and counting questions.

Out of scope: new architectures or pretraining, memory carried across separate tasks, wall-clock latency, and formal safety guarantees for edits.

### (c) Evaluation metrics

**Task score.** Accuracy, pass rate, or best evaluator score (higher is better). It shows whether the agent kept the right information.

**Prefix-reuse FLOPs** (the main cost metric). Shrinking the context saves compute on every later turn, while an edit in the middle destroys the cached computation after it; this metric charges both effects.

*Intuition.* A server with prefix caching pays (i) a fixed per-token cost for every token it must actually process (uncached prompt tokens plus generated tokens) and (ii) an attention cost per token pair: each newly processed prompt token looks at all tokens before it, and each generated token looks at the whole prompt and the earlier generated tokens.

*Formula.* At turn $$t$$ the prompt has $$P_t$$ tokens, of which a prefix of $$R_t$$ tokens is reused from the cache, leaving $$U_t = P_t - R_t$$ to process; the model generates $$G_t$$ tokens:

$$
F_t = C_{\text{tok}}\,(U_t + G_t) + C_{\text{att}}\Big[\tfrac12\big(P_t^2 - R_t^2\big) + G_t P_t + \tfrac12 G_t^2\Big],
\qquad F = \sum_{t=1}^{T} F_t .
$$

The bracket counts token pairs: $$\tfrac12(P_t^2-R_t^2)$$ while prefilling the uncached part, $$G_tP_t + \tfrac12 G_t^2$$ while decoding. The constants: $$C_{\text{tok}} \approx 2\times$$(weights touched per token), because each weight performs one multiply-add (2 FLOPs) per token; and

$$
C_{\text{att}} = 4\,L_{\text{att}}\,h_q\,d_h,
$$

because each token pair costs, in each of the $$L_{\text{att}}$$ full-attention layers, each of the $$h_q$$ query heads, and each of the $$d_h$$ head dimensions, one multiply-add for the attention score and one for mixing the values. For Qwen3.6-27B: $$C_{\text{att}} = 4\cdot16\cdot24\cdot256 \approx 3.93\times10^{5}$$ and $$C_{\text{tok}} = 4.87\times10^{10}$$ (≈24B weights touched per token).

*Tiny example.* Toy constants $$C_{\text{tok}}=10$$, $$C_{\text{att}}=1$$; prompt $$P_t=10$$, output $$G_t=2$$.
- Append-only turn, $$R_t = 8$$: $$10\cdot(2+2) + \big[\tfrac12(100-64) + 20 + 2\big] = 40 + 40 = 80$$.
- Edit at the very start, $$R_t = 0$$: $$10\cdot(10+2) + [50+20+2] = 120 + 72 = 192$$, i.e., 2.4× more.

At real scale (Qwen3.6-27B, $$P_t = 20{,}000$$, $$G_t=500$$), one turn costs $$1.41\times10^{14}$$ FLOPs append-only ($$R_t=18{,}000$$), $$5.74\times10^{14}$$ after a mid-context edit ($$R_t=10{,}000$$), and $$10.81\times10^{14}$$ after an edit at the start ($$R_t=0$$): 7.7× the append-only turn. An edit pays off only when the shorter context saves more on later turns than the recomputation costs now.

**Context pressure** (ContextBench). *Intuition:* how many times the total input would fill the context window. *Formula:*

$$
\rho = \frac{\sum_{i=1}^{n} \lvert u_i \rvert}{32{,}768}.
$$

*Example:* 24 Needle-Retention chunks of ≈4K tokens ≈ 96K tokens, so $$\rho \approx 3$$; keeping everything is impossible and filler must go. Levels with $$\rho < 1$$ are controls where keeping everything fits.

**Geometric-mean speedup** (Software World). *Intuition:* the typical multiplicative speed gain across benchmarks, robust to one extreme value. *Formula:* with $$s_j = \dfrac{\text{CPU instructions executed before}}{\text{CPU instructions executed after}}$$ on benchmark $$j$$ (a broken benchmark counts as $$s_j = 1$$),

$$
S = \Big(\prod_{j=1}^{m} s_j\Big)^{1/m}.
$$

*Example:* $$s_1 = 1.10$$, $$s_2 = 1.00$$ gives $$S = \sqrt{1.10} \approx 1.049$$, a 4.9% speedup. Instruction counts serve as a noise-free proxy for runtime.

**Pareto frontier.** *Intuition:* keep only methods that no other method beats on both accuracy and cost. *Formula:* method $$a$$ dominates $$b$$ if $$\text{acc}_a \ge \text{acc}_b$$ and $$\text{cost}_a \le \text{cost}_b$$, with at least one strict; the frontier is the set of undominated methods. *Example:* $$a = (60\%,\ 6\ \text{PFLOPs})$$ dominates $$b = (53\%,\ 8\ \text{PFLOPs})$$, so only $$a$$ is on the frontier.

For closed models served through paid APIs, cost is reported in USD per task instead of FLOPs.

### (d) The standard approach and why it fails

*Intuition.* A standard LM agent only appends: every action and every tool output goes to the end, and nothing is ever removed.

$$
c_{t+1} = c_t \oplus y_t \oplus o_t .
$$

*Example.* At 1,000 tokens per turn, $$\lvert c_{33}\rvert \approx 33{,}000 > 32{,}768$$: a 100-turn web search overflows by turn 33.

- **F1 Overflow.** The context exceeds $$B$$ and the run must stop or be cut.
- **F2 Rising cost and clutter.** Per-turn attention cost grows with length, and stale outputs (e.g., irrelevant search results) stay forever.

The usual patch is harness-scheduled compaction: once $$\lvert c_t\rvert > \alpha B$$, history is replaced by a summary, $$c_{t+1} = [\text{system prompt},\, x,\, \operatorname{summary}(c_t)]$$ (Codex-style uses $$\alpha = 0.75$$). It fixes F1 and introduces:

- **F3 Lossy rewriting.** Summaries drop or invent details (exact needle lines, exact Sudoku cells).
- **F4 Fixed timing.** Compaction fires at a preset length, not when it is useful.
- **F5 No surgical updates.** Changing one Sudoku cell requires regenerating the whole board.
- **F6 No eviction.** Text that entered the context can be copied to disk but stays in the live context.
- **F7 Frozen strategy.** The policy lives in harness code and cannot improve by learning.

Every editing method also faces:

- **F8 Edits break the cache.** An edit at position $$p$$ forces recomputation of everything after $$p$$ (the 7.7× case above).

### (e) Existing baselines

| Baseline | Who decides *when* | Allowed operation | Fixes | Remaining limitation |
|---|---|---|---|---|
| Base (Mini-SWE-Agent: a minimal agent whose only tool is a shell) | nobody | append only | none | F1, F2 |
| Codex-style Summary (from OpenAI's Codex coding agent) | harness, at 75% of $$B$$ | summarize all but the system prompt and task | F1 | F3, F4, F5, F6 |
| MEM1 | harness, every turn | rewrite one compact state from the previous state plus the new observation | F1, F2 | rewrite every turn; F3, F7 |
| Self-Compact | model, asked every 2 turns once the context exceeds 37% of $$B$$ | whole-history summary | F4 | F3, F5, F6 |
| ACM (Agentic Context Management) | model | replace everything since its previous call with a summary; query the originals later | F4, partly F3 | fixed granularity; F5 |
| Context Folding | model | branch into a sub-task and return only its summary | keeps sub-task clutter out | cannot delete from the main context (F6) |
| RLM (Recursive Language Model) | model, through code | the long input lives as a variable in a code interpreter, read on demand | huge inputs never flood the context | the live history still only grows (F1, F2) |

Down this list the model gains more control, always within an operation set designed by humans.

### (f) The method: Context Language Models

**One-sentence idea.** Mirror the model's live context into an ordinary file and let the model rewrite that file with any code, so context management becomes the model's own behavior rather than harness logic.

**Analogy.** A standard agent writes in a bound notebook with a pen and can only add pages. A CLM works on a whiteboard: it erases dead ends, condenses a messy area into a clean note, keeps a scoreboard it updates in place, and photographs a section into a drawer (disk) before erasing it.

**Component 1: model-controlled transition.**
*Intuition:* the model's output may contain a small program that rewrites its own next context; without one, the system appends as usual.

$$
c_{t+1} = g_t\big(c_t \oplus y_t \oplus o_t\big),
\qquad
g_t =
\begin{cases}
\text{the edit program written in } y_t & \text{if the model edits},\\
\text{identity (plain append)} & \text{otherwise}.
\end{cases}
$$

The paper writes this compactly as $$c_{t+1} = f^{\text{CLM}}_\theta(c_t)$$, versus $$c_{t+1} = c_t \oplus f^{\text{LM}}_\theta(c_t)$$ for a standard LM.
*Example:* the context holds a 200-token task, three 3,000-token search results, and a 100-token note ($$\lvert c\rvert = 9{,}300$$). The model writes a Python loop replacing each irrelevant result by "Searched X: no relevant results" (≈20 tokens), leaving $$\lvert c\rvert \approx 200 + 60 + 100 = 360$$.

**Component 2: context as a file.** The live context sits at a path such as `/tmp/.live_ctx/LIVE_CTX_MAIN.txt`, named in the system prompt. The model edits it with ordinary shell tools (`sed`, `grep`, Python). After each command the harness reloads the file, so the edited text is exactly what the server receives next. Two light supports remain in the harness: a reminder appears 2,048 tokens before the budget (current models estimate their own context length poorly), and on some benchmarks a turn that would overflow is rolled back and retried.

**Component 3: multi-agent through more files.** A swarm starts with one context file per agent; a subagent is launched by creating a context file and ended by deleting it. An orchestrator can keep a compact scoreboard of its workers inside its own file.

**Component 4: steering with a skill.**
*Intuition:* because management is the model's own behavior, plain instructions change it.

$$
y_t \sim \pi_\theta(\cdot \mid c_t,\ s).
$$

*Example:* $$s =$$ "Whenever your context exceeds 24,000 tokens, compact it to about 4,000." The first compaction then occurs at a median of ≈23.6K tokens.

**Component 5: evolving the skill.**
*Intuition:* with the model fixed, search over instruction texts and keep the one that scores best.

$$
s^\star = \arg\max_{s}\ \mathbb{E}_{x\sim\mathcal{D}}\big[\,r(\tau(x;s))\,\big],
$$

where $$\tau(x;s)$$ is the trajectory (full record of turns) the agent produces on task $$x$$ with skill $$s$$, and $$\mathcal{D}$$ is the task distribution. A proposer LM reads rollouts (including a success and a failure on the same instance) and writes candidate skills; each candidate is screened on 6, then 12 more training instances, then scored on a development split; skills that are Pareto-optimal in accuracy and cost are kept; the final pick is tested once on a held-out split. The proposer is either a stronger model (assisted evolution) or the agent itself (self-evolution).
*Example:* on KV Store the empty skill scores 22.3% on the development split and the evolved skill 83.8%, so the evolved skill is selected.

**Component 6: RL with a success-gated efficiency bonus.**

*6a. Outcome advantage.* *Intuition:* in GRPO (Group Relative Policy Optimization), the model makes $$n$$ attempts at the same question, and each attempt is scored relative to its group, so no separate value-estimating network is needed.

$$
A^{\text{out}}_i = \frac{r_i - \operatorname{mean}(r_{1:n})}{\operatorname{std}(r_{1:n})}.
$$

*Example:* rewards $$(1,1,0,0)$$ give mean $$0.5$$, std $$0.5$$, so $$A^{\text{out}} = (1,\,1,\,-1,\,-1)$$.

*6b. Efficiency advantage.* *Intuition:* among successful attempts, reward the cheaper ones; failures earn no efficiency credit, so the model cannot gain by deleting context recklessly.

$$
A^{\text{eff}}_i =
\begin{cases}
\operatorname{clip}\!\Big(\dfrac{\bar c - c_i}{\bar c},\,-1,\,1\Big) & i \in \mathcal{S},\\
0 & i \notin \mathcal{S},
\end{cases}
\qquad
\bar c = \frac{1}{\lvert\mathcal{S}\rvert}\sum_{k\in\mathcal{S}} c_k ,
$$

where $$\mathcal{S}$$ is the set of successful attempts, $$c_i$$ is attempt $$i$$'s prefix-reuse FLOPs, and $$A^{\text{eff}} \equiv 0$$ whenever $$\lvert\mathcal{S}\rvert < 2$$.
*Example:* the two successes cost 2 and 4 PFLOPs, so $$\bar c = 3$$ and $$A^{\text{eff}} = (+\tfrac13,\ -\tfrac13,\ 0,\ 0)$$.

*6c. Combination.*

$$
A_i = A^{\text{out}}_i + w\,A^{\text{eff}}_i, \qquad w = 0.25 .
$$

*Example:* $$A = (1.083,\ 0.917,\ -1,\ -1)$$: both successes are reinforced, the cheaper one more.

*6d. Stepwise update.* *Intuition:* edits rewrite the past, so a trajectory is not one growing sequence; every model call $$j$$ of attempt $$i$$ is trained on the exact context $$c_{i,j}$$ it saw, weighted by its attempt's advantage.

$$
\nabla_\theta J \approx \sum_{i=1}^{n} A_i \sum_{j} \nabla_\theta \log \pi_\theta\big(y_{i,j} \mid c_{i,j}\big).
$$

*Example:* an attempt with $$A_i = 1.083$$ and three model calls makes all three outputs more likely with the same weight $$1.083$$. The implementation adds ratio clipping in $$[0.2,\,0.28]$$ (a cap on how far one update moves each probability), a KL penalty of 0.01 (keeping the model close to its starting point), and applies the efficiency term to context-management tokens.

**Component 7: Suffix Cache Reuse (SCR), a serving co-design.**
*Intuition:* after an edit $$[X\,Y\,Z] \to [X\,Y'\,Z]$$, the span $$Z$$ is unchanged text, so its cached keys and values are reused; only $$Y'$$ is new.

$$
\text{recomputed tokens:}\quad
\underbrace{\lvert Y'\rvert + \lvert Z\rvert}_{\text{standard prefix caching}}
\;\longrightarrow\;
\underbrace{\lvert Y'\rvert}_{\text{SCR}} .
$$

$$Z$$ shifts by $$\Delta = \lvert Y'\rvert - \lvert Y\rvert$$ positions. Because RoPE encodes position as a rotation, moving a cached key is one extra rotation, $$k \leftarrow \mathcal{R}(\Delta)\,k$$.
*Example:* $$X$$ = 10,000 tokens, $$Y$$ = 5,000 tokens of search results replaced by a 200-token note $$Y'$$, $$Z$$ = 3,000 tokens. Standard serving recomputes 3,200 tokens; SCR computes 200 and rotates $$Z$$'s keys by $$\Delta = -4{,}800$$.

SCR is an approximation: $$Z$$'s cached states were computed while $$Z$$ attended to the old $$Y$$. To bound it, at most $$K = 6$$ surviving spans (the longest) are relocated per edit and the rest are recomputed. Linear-attention layers hold no per-token cache, so SCR continues from their pre-edit state snapshot, and $$Y'$$ is processed only in the 16 full-attention layers. The same mechanism serves ordinary chat traffic: reasoning models' chat templates delete earlier hidden "thinking" blocks every turn, which is itself an edit.

**Pseudocode.**

```text
# ---- CLM agent loop (a swarm runs one loop per context file) ----
ctx = File("/tmp/.live_ctx/LIVE_CTX_MAIN.txt")
ctx.write(system_prompt(path=ctx.path) + skill s + task x)
for t in 1..T:
    c = ctx.read()
    if tokens(c) > B - 2048: c += "[near budget: consider editing your context]"
    y = LM(c)                          # reasoning + one shell command
    o = run(y.command)                 # the command may rewrite ctx (sed, python, ...)
    if ctx was not modified:
        ctx.append(y + o)              # default: plain append
    # if it was modified, the edited file itself is the next context
    if y.done: break

# ---- RL: stepwise GRPO with success-gated efficiency ----
for step in 1..70:
    for each of 8 prompts:
        run n = 32 CLM attempts, saving every (c_ij, y_ij)
        r_i = judge(answer_i) in {0,1};   cost_i = prefix-reuse FLOPs of attempt i
        A_i = A_out_i + 0.25 * A_eff_i
    drop groups whose rewards are all equal          # no learning signal
    policy-gradient step on every (c_ij -> y_ij), weighted by A_i

# ---- Serving with SCR ----
on each new prompt of a session:
    diff against the session's previous prompt
    reuse the cached prefix X
    relocate up to K = 6 surviving spans (re-rotate their RoPE keys)
    prefill only new or changed tokens
```

**How the pieces fix each failure.**

| Failure | Fix in CLMs |
|---|---|
| F1 Overflow | the model compresses or deletes before reaching $$B$$; reminder at $$B - 2{,}048$$ |
| F2 Cost and clutter | stale outputs are deleted, keeping the live context small (an orchestrator stays at 6–8K tokens) |
| F3 Lossy rewriting | exact text stays verbatim, backups go to disk, notes record what was removed |
| F4 Fixed timing | the model decides when, and a skill can tell it when |
| F5 No surgical updates | a one-line `sed` changes one Sudoku cell |
| F6 No eviction | any span can be deleted, including input already read |
| F7 Frozen strategy | skills, skill evolution, and RL improve management |
| F8 Edits break the cache | the RL efficiency bonus favors cache-friendly edits; SCR reuses the unchanged suffix |

---

## 4. Results and analysis

**Organization.** The experiments answer three questions in sequence. **Part A** asks whether an existing LM, simply handed an editable context file, already manages context better than human-designed strategies (zero-shot). **Part B** asks whether that management improves through learning, in the context (instructions, evolved skills) or in the weights (RL). **Part C** asks whether serving can recover the compute that edits cost (failure F8). *General setup:* within each comparison every method uses the same LM, the same agent backbone (a minimal shell-only agent; the Pi agent harness for the 24-hour swarm), and the same token budget (mostly 32K, counted with OpenAI's o200k tokenizer), and differs only in context management. Costs in Parts A and B assume standard prefix caching; SCR's extra savings appear only in Part C. *How to read:* most plots put cost (prefix-reuse PFLOPs per question, Section 3c; lower is better) on the x-axis and task score (higher is better) on the y-axis, so the best point sits top-left, and a dashed Pareto frontier joins the undominated methods. In tables, ↑ means higher is better and ↓ lower is better. Numbers marked ≈ are read off figures.

### Part A: Does an off-the-shelf model manage its own context better than fixed strategies?

**Setup.** The comparison climbs three tiers of horizon. *Diagnostic:* ContextBench with GPT-5.4 at 32K, pressure $$\rho$$ from 0.25× to 24×, 4–8 seeds (independent random runs) per level; every method gets the same task instruction plus an equally detailed skill for its own mechanism. *Short horizon (≤100 turns):* Qwen3.6-27B (an open 27-billion-parameter model) at 32K on TerminalBench 2.1 (89 terminal tasks with pass/fail test scripts), TBLite (a lighter terminal benchmark), and BrowseComp-Plus (830 hard research questions answered by searching a fixed document collection and graded by an LLM judge), against the six baselines of Section 3e. *Long horizon:* four math optimization problems (circle packing, min–max distance ratio, Erdős minimum overlap, Heilbronn triangles) with Claude 4.6 Sonnet against OpenEvolve (an evolutionary program-search system modeled on DeepMind's AlphaEvolve) and OpenEvolve-Agent, capped at 100 scored attempts or 5 hours, one run each; EdgeBench-10 (ten real code-optimization tasks, 12 hours each, best of three seeds, judge score 0–100) with Qwen3.6-27B and Claude 4.6 Sonnet against Summary; and Software World, where six GPT-5.6-Sol agents (272K budget) spend over 24 hours speeding up six interdependent Python libraries (`requests`, `urllib3`, and four packages built on them) and are scored by geometric-mean speedup on 17 benchmarks from four downstream packages they never see.

**How to read.** ContextBench plots accuracy against pressure: a flat line at 1.0 is perfect, and a falling line marks where a strategy breaks. Short-horizon plots are accuracy vs prefix-reuse PFLOPs (top-left wins). Long-horizon results report final best score, with compute per trial in PF (prefix-reuse PFLOPs) where measured.

**Takeaway 1: Letting the model edit its own context beats every fixed strategy in score while using less compute, from synthetic diagnostics to 24-hour multi-agent runs.**

- **Diagnostic.** CLM stays at ≈1.0 on all four tasks at every pressure, while the best baseline falls to ≈0.35–0.75 at the highest pressure. The failure modes match Section 3d: summaries drop needles and corrupt Sudoku boards (F3), methods without in-place edits regenerate whole boards (F5), and offloading tools store data on disk without evicting it (F6).
- **Short horizon.** On BrowseComp-Plus, CLM reaches 59.4% (11.4% relative above Summary) with 21.5% fewer PFLOPs than Summary and 28.9% fewer than MEM1. It matches Summary on TerminalBench 2.1 with 29.5% fewer PFLOPs and beats it on TBLite (73.7% vs 67.0%) at 91% of its PFLOPs, lying on the Pareto frontier of all three.
- **Long horizon.** CLM leads all four math problems; on EdgeBench-10 it scores 44.6 vs 42.3 with 59% less compute (Qwen3.6-27B) and 51.0 vs 42.3 (Claude 4.6 Sonnet); in Software World its swarm reaches ≈1.044× vs ≈1.026× at equal spend, a 65% larger gain.
- **Model capability drives the gain.** Qwen3.5-9B edits 1.4 times per TerminalBench task (none in half of them) versus 2.6 for Qwen3.6-27B, and its median peak context is 30.2K vs 17.6K tokens; it still edges Summary on BrowseComp-Plus (39.9% vs 37.7%).
- **Subagents help when there is room.** At 32K, CLM with subagents ties plain CLM on EdgeBench-10 (44.2 vs 44.6); at 128K, subagents lead (50.2 vs 47.3 for CLM and 47.8 for Summary, at 219 / 142 / 222 PF per trial).
- **New behaviors emerge unprompted.** An orchestrator updates a worker scoreboard through 163 in-place edits while staying at 6–8K tokens; the model invents a "notes" chat role; it defines a `compact_turns` helper and reuses it 37 times.

| Tier | Benchmark (horizon) | Model, budget | Strongest baseline | CLM |
|---|---|---|---|---|
| Diagnostic | ContextBench, 4 tasks, pressure up to 24× | GPT-5.4, 32K | ≈0.35–0.75 at top pressure | ≈1.0 at every pressure |
| Short | BrowseComp-Plus (acc ↑) | Qwen3.6-27B, 32K | Summary ≈53.3% (implied by +11.4% relative) | **59.4%**, 21.5% fewer PFLOPs |
| Short | TerminalBench 2.1 (pass ↑) | Qwen3.6-27B, 32K | Summary | same pass rate, 29.5% fewer PFLOPs |
| Short | TBLite (acc ↑) | Qwen3.6-27B, 32K | Summary 67.0% | **73.7%**, 9% fewer PFLOPs |
| Long | EdgeBench-10, 12 h (score ↑) | Qwen3.6-27B, 32K | Summary 42.3 at 437 PF | **44.6** at 179 PF |
| Long | EdgeBench-10, 12 h (score ↑) | Claude 4.6 Sonnet, 32K | Summary 42.3 | **51.0** |
| Long | Software World, 24 h (speedup ↑) | GPT-5.6-Sol, 272K | Summary swarm ≈1.026× | **≈1.044×** at equal USD |

Math optimization (Claude 4.6 Sonnet, 32K, best of run; SA = with subagents):

| Method | Circle packing ↑ | Heilbronn ↑ | Min–max distance ↑ | Erdős overlap ↓ |
|---|---|---|---|---|
| OpenEvolve | 2.541 | 0.03127 | 0.07690 | 0.38123 |
| OpenEvolve-Agent | 2.525 | 0.03053 | 0.07724 | 0.38167 |
| CLM | 2.618 | **0.03653** (+16.8%) | **0.07758** | **0.38094** |
| CLM (SA) | **2.636** | 0.03617 | **0.07758** | 0.38109 |

### Part B: Can context management be taught, in context or in weights?

**Setup.** Three levers. *Talking (Component 4):* Claude 4.6 Sonnet on BrowseComp-Plus, one sentence appended to the task prompt versus no sentence on the same questions: "compact to ≈4K once your context exceeds $$Y$$" for $$Y \in \{16\text{K}, 24\text{K}, 32\text{K}\}$$; "compact only at sub-question boundaries" (sessions of four chained questions); "back up the context file to disk before every edit." *Evolving instructions (Component 5):* on ContextBench at 32K, assisted evolution pairs a Qwen3.6-27B agent that starts with no context-management instruction with Claude Fable 5.1 as skill proposer; self-evolution uses Claude Opus 5 for both roles; cost is PFLOPs per task for Qwen and USD per task for Opus. *Training weights (Component 6):* Qwen3.5-9B trained for 70 GRPO steps on 3,040 OpenResearcher research prompts (8 prompts × 32 attempts per step, efficiency weight 0.25), then tested on all 830 BrowseComp-Plus questions; the Summary harness trained with the same recipe (task reward only) is the control.

**How to read.** For talking, the measured behavior should track the instruction (first compaction near $$Y$$, compaction spiking at boundaries, backups appearing). For evolution, good skills move up (accuracy) and left (cost) from the starting point. For RL, compare before → after accuracy (↑) and PFLOPs per question (↓).

**Takeaway 2: Context management is a learnable skill: one sentence reliably reshapes it, evolved instructions repair weak management, and RL turns a small model from worse than Summary into equally accurate at 39% less compute.**

- **Talking works.** First compaction lands at ≈16.0K / ≈23.6K / ≈30.9K tokens for $$Y$$ = 16K / 24K / 32K; compaction exactly at a sub-question boundary rises from ≈0.34 to ≈0.77; edits preceded by a full backup rise from ≈0.00 to ≈0.68. Harness and weights stay unchanged.
- **Evolving works.** Assisted evolution lifts development accuracy on all four tasks (KV Store 22.3% → 83.8%, Log Triage 0.0% → 100%), and on the held-out KV Store test split from 38.3% to 74.2% (+35.9 points), with evolved skills extending the accuracy–cost Pareto frontier. Under self-evolution, Opus 5 starts at 94–100% and its evolved skills cut cost or raise accuracy further.
- **Training works.** RL lifts CLM from 28.8% to 42.5% (+47.6% relative) while PFLOPs per question fall from 1.52 to 1.34 (−12%). The trained Summary harness reaches 42.1% at 2.19 PFLOPs, so CLM ends 0.4 points ahead with 38.8% fewer PFLOPs. The efficiency bonus lowers cost without a clear accuracy loss.

| Lever | Setting | Before | After |
|---|---|---|---|
| "Compact once you reach $$Y$$" | Claude 4.6 Sonnet, BrowseComp-Plus | no instruction | first compaction ≈16.0K / ≈23.6K / ≈30.9K |
| "Compact at sub-question boundaries" | same | ≈0.34 at the boundary turn (≈0.40 within 2 turns) | ≈0.77 (≈0.88) |
| "Back up before you compact" | same | ≈0.00 of edits | ≈0.68 full backups (+≈0.09 partial) |
| Assisted evolution, development split ↑ | Qwen3.6-27B agent, Fable 5.1 proposer | Needle 97.6 / Sudoku 45.3 / KV 22.3 / Log 0.0 % | 100 / 65.8 / 83.8 / 100 % |
| Assisted evolution, held-out test ↑ | KV Store | 38.3% | **74.2%** |
| RL, accuracy ↑ | Qwen3.5-9B, BrowseComp-Plus | CLM 28.8%; Summary 34.7% | CLM **42.5%**; Summary 42.1% |
| RL, PFLOPs per question ↓ | same | CLM 1.52; Summary 4.01 | CLM **1.34**; Summary 2.19 |

### Part C: Can serving recover the compute that edits cost?

**Setup.** As Section 3c shows, an edit near the start of a 20K-token prompt makes a turn 7.7× more expensive under standard prefix caching. The test serves Qwen3.6-27B CLM runs on all 830 BrowseComp-Plus questions with standard SGLang (an open-source LLM serving engine with prefix caching) and with SGLang plus SCR ($$K = 6$$). A sweep over $$K \in \{1, 2, 3, 6, 12, 64\}$$ on 64 questions checks sensitivity.

**How to read.** Accuracy should stay equal while PFLOPs per question fall. The token breakdown splits every prompt token into served from the prefix cache, reused by SCR, or recomputed; less recomputation means less waste, and "edited turns" isolates the turns right after an edit.

**Takeaway 3: SCR cuts server compute by 35% at identical accuracy, and most of its saving also applies to ordinary reasoning-model serving.**

- **Same accuracy, less compute.** Both systems score ≈60.2%; SCR needs 65.0% of standard SGLang's compute (≈10.98 → ≈7.14 PFLOPs per question).
- **Where the saving comes from.** SCR reuses 7.8% of all prompt tokens beyond prefix-cache hits: 5.3 points after deleted reasoning blocks and 2.5 points after CLM edits. In edited turns, recomputed tokens drop from ≈75.8% to ≈41.8%.
- **Robust to $$K$$.** Accuracy shows no degradation for any $$K$$, and relocated tokens per edit saturate around $$K = 6$$ (≈4K tokens).
- **Remaining waste is fixable.** In edited turns ≈22.7% of prompt tokens are an unchanged prefix that SGLang still recomputes, because it stores linear-attention states only at request boundaries; checkpointing states at message boundaries targets this.

| Serving | Accuracy ↑ | PFLOPs / question ↓ | All turns: cached / SCR-reused / recomputed (% tokens) | Edited turns: cached / SCR-reused / recomputed |
|---|---|---|---|---|
| Standard SGLang | ≈60.2% | ≈10.98 | ≈72.9 / 0 / ≈27.1 | ≈24.2 / 0 / ≈75.8 |
| SGLang + SCR ($$K=6$$) | ≈60.2% | ≈7.14 (65.0%) | ≈73.9 / 7.8 / ≈18.3 | ≈29.9 / ≈28.2 / ≈41.8 |

---

## 5. Limitations

**Unrestricted self-editing removes guarantees about what the model keeps.** A CLM can delete task constraints, overwrite the record of its own actions, or write instructions to itself that persist into later turns (a prompt injection, i.e., text that steers a model toward unintended instructions); the paper cites a documented case of a model planting unauthorized instructions in its own compaction summary. After many edits, the live context no longer serves as an audit trail of what happened. The paper names this risk but evaluates no injection resistance, no faithfulness of edited history, and no safeguards such as write-protected regions or edit logs, and every reported number comes from benign tasks. Settings where instructions must hold, such as security policies or user constraints, need these safeguards before CLMs are deployed.

**The comparisons carry small CLM-specific allowances, and several headline gains rest on few runs.** On BrowseComp-Plus, CLM's editing turns do not count toward the 100-turn cap and an overflowing turn can be rolled back and retried up to six times, allowances stated only for CLM; in RL, CLM likewise gets 80 non-editing turns. ContextBench is the authors' own benchmark, and its prompts explicitly say "delete this entire block" or "offload this whole block," operations that only an editable context can perform. The math results use one run per method, Software World one swarm per method with a 1.8-point absolute difference (1.044× vs 1.026×), and EdgeBench reports best-of-three without spread. Models, judges, and budgets also change between sections: Qwen3.5-9B with CLM scores 39.9% vs 37.7% for Summary on BrowseComp-Plus in the zero-shot comparison, but 28.8% vs 34.7% in the RL setting.

**"CLM" is a harness pattern plus training, and its efficiency claims rely on proxies and approximations.** Zero-shot CLM is an existing LM with a file-sync wrapper, a budget reminder, and rollback; no new architecture is involved, and the benefit shrinks with weaker models (Qwen3.5-9B edits half as often and fills its context to 30.2K of 32K). Learning in the weights is shown for one 9B model, one task family, and 70 RL steps. Cost is theoretical prefix-reuse FLOPs or USD, not wall-clock latency. SCR recovers editing costs only approximately: reused suffix states are stale, linear-attention layers never see the new text, accuracy is checked for one model on one benchmark (64 questions for the $$K$$ sweep), and most of its saving (5.3 of 7.8 points) comes from deleted reasoning blocks rather than from CLM edits.

---

## 6. How to reproduce

**Step 1: Get the code, models, and hardware.** The official repository is `github.com/facebookresearch/context-language-models` (CC BY-NC 4.0, non-commercial). It contains `clm/clm_harness` (CLM inside Harbor, an agent-evaluation framework: `pip install -e .`, then `clm-harbor run -p <harbor-task> -a clm-minimal -m openai/<model> --agent-kwarg api_base=http://localhost:8000/v1`), `clm/clm_icl` (skill evolution), `clm/clm_rl` (RL), and `suffix_cache_reuse` (the SGLang patch); a `pi-clm` plugin for the Pi agent installs with `pi install npm:@lolipopshock/pi-clm`. Open-weight models (Qwen3.6-27B, Qwen3.5-9B) are served with vLLM for evaluation and SGLang for RL rollouts and SCR (both open-source LLM serving engines). Closed models come through paid APIs: GPT-5.4, GPT-5.4-nano (RL judge), GPT-5.6-Sol, Claude 4.6 Sonnet, Claude Fable 5.1, and Claude Opus 5. RL used 16 H200 GPUs for the policy and 48 for rollouts.

**Step 2: Configure the CLM harness.** Use a minimal shell-only agent (Mini-SWE-Agent style; Pi for Software World). Mirror the context to `/tmp/.live_ctx/LIVE_CTX_MAIN.txt`, name the path in the system prompt, reload the file after every command, and append by default. Insert the budget reminder 2,048 tokens before the limit and count tokens with the o200k tokenizer. On BrowseComp-Plus, exclude editing turns from the 100-turn cap and allow up to six rollback-and-retry attempts on overflow; on EdgeBench all methods roll back up to 50 times. Expose BrowseComp-Plus search tools as in-context skills. Subagents are capped at five (40 turns each) for math and five to six for EdgeBench.

**Step 3: Prepare benchmarks and baselines.** ContextBench: GPT-5.4, 32,768-token limit with 2,048 reserved for output, 4 seeds (8 on selected levels), the same task instruction plus a per-method skill. TerminalBench 2.1: 89 tasks, thinking (hidden reasoning) on, 4,096 generated tokens per call, 32,000-token budget, 64 turns, 180 s per command, 4× default time limit; TBLite: same with 2,048 generated tokens per call. BrowseComp-Plus: 830 questions, top-10 snippets of 512 tokens, document reader capped at 8,192 tokens, temperature 0.7, top-p 0.95, 23,560-token budget, Qwen3.5-27B as judge at temperature 0. Math: Claude 4.6 Sonnet, 32,000 tokens, 100 attempts or 5 hours; OpenEvolve with population 60 in four islands; OpenEvolve-Agent with 25 turns per candidate. EdgeBench-10: tasks `ad_placement_optimization`, `apple_incremental_game`, `graph_node_classification`, `grid_turing_robot`, `juliet_vulnerability_analyzer`, `openrct2_theme_park_ai`, `schemathesis_datagen_pipeline`, `triangulation_coloring_optimization`, `vehicle_routing_time_windows`, `wesnoth_tactical_ai`; 3 seeds, 12 hours, no network, temperature 0.7, 8,192 generated tokens, 32,000 budget (128,000 for the extra study). Software World: agents on `requests`, `urllib3`, `nilearn`, `requests-gssapi`, `databricks-sql-connector`, `responses`; held-out `requests-cache`, `cachecontrol`, `zeep`, `awswrangler`; GPT-5.6-Sol at 272K; cost in USD. Baselines: Summary uses the Codex prompts at 75% of budget; Self-Compact its original prompts every two turns past 37%; RLM and ACM their released code; MEM1 the authors' re-implementation.

**Step 4: Compute the metrics.** Apply the prefix-reuse FLOPs formula of Section 3c turn by turn. Qwen3.6-27B constants: 64 layers, hidden size 5,120, MLP width 17,408; 16 full-attention layers with 24 query heads, 4 key–value heads, head dimension 256; 48 Gated DeltaNet layers with 16 key heads, 48 value heads, head dimensions 128. Evaluation matches cached prefixes at message level; RL uses exact token-level matching against the previous turn. Software World speedup uses executed-instruction counts on 17 CPU benchmarks.

**Step 5: Run the learning experiments.** Steering prompts appear verbatim in the paper's Appendix E (thresholds: 30 long questions, 48K budget, 200 turns; boundaries: 189 sessions of four questions, 24K budget; backup: 200 questions, 16K budget). Skill evolution (`clm/clm_icl`) is a GEPA-style loop (reflective prompt evolution): four parallel proposers, each reading at least ten rollouts and writing at least four full rewrites; screening on 6 then 12 training instances; development split of 102 instances per task; a lineage stops after five non-improving proposals; the test split has 102 instances × 3 seeds; the agent runs at 32,000 tokens, 4,096 reserved, 240 turns. RL (`clm/clm_rl`): 3,040 OpenResearcher prompts, 8 prompts × 32 rollouts per step, learning rate $$10^{-6}$$, weight decay 0.1, clipping $$[0.2, 0.28]$$, KL coefficient 0.01, truncated importance sampling (a correction for rollouts produced by a slightly stale model copy), dynamic sampling (dropping groups with identical rewards), 70 steps, temperature 0.7, top-p 0.95; CLM at 28K budget with 2,048 reserved and 80 non-editing turns; Summary compacts at 28,672 tokens with 100 turns; binary reward from GPT-5.4-nano with the DeepSearchQA rubric; $$w = 0.25$$; checkpoint chosen on 500 held-out OpenResearcher questions.

**Step 6: Measure serving with SCR.** Apply the `suffix_cache_reuse` patch to SGLang with $$K = 6$$ and session-private cache slots, serve Qwen3.6-27B CLM runs on all 830 BrowseComp-Plus questions, and compare accuracy, PFLOPs per question, and the token breakdown against unpatched SGLang; repeat on 64 questions for $$K \in \{1,2,3,6,12,64\}$$.

**Missing parts.** ContextBench is marked "coming soon" in the repository, and Software World (library versions, the 17 benchmarks, the wake schedule between work sessions) has no listed release. The full CLM system prompt and per-method skills appear only as excerpts. RL details left open: the exact advantage normalization, the size of the penalties for failed tool calls and malformed outputs, and how the efficiency term is restricted to context-management tokens. Seeds and Qwen3.5-9B FLOPs constants are not listed. Some settings differ across the paper: TerminalBench 2.1 uses 64 turns in the appendix versus a 100-turn cap in the main figure; BrowseComp-Plus is labeled 32K but configured with a 23,560-token budget; EdgeBench subagents are capped at five in the main text and six in the appendix; the BrowseComp-Plus judge is Qwen3.5-27B for zero-shot runs and GPT-5.4-nano for RL. Several experiments require paid access to closed models.
