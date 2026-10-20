---
layout: post
title: 'Reading Notes of SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning'
date: 2026-09-08 23:10:00
description: 'Train an agent to spawn an auxiliary agent from the ongoing trajectory to select blocks of KV cache from history and delete at regular intervals.'
tags: nlp kv-cache
categories: ReadingNotes
thumbnail: assets/img/blogs/sidequest.png
toc:
  sidebar: left
related_posts: false
---

Train an agent to spawn an auxiliary agent from the ongoing trajectory to select blocks of KV cache from history and delete at regular intervals.

<!--more-->

> **Reference**: https://arxiv.org/abs/2602.22603

## 1. Summary

![image.png](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img20260908132706.png)

How to train? Sample all correct trajectories.
1. Auxiliary agent: we can know how far each turn can propagate. So at each turn, those previous turns whose farthest mentions happen before the turn should be deleted. So we will have ground truth. For the thinking, use gpt-oss-120b to complete. Also, random mask half outdated turns and let the ground truth (blocks to be deleted) be the rest of outdated turns
2. Main agent: original trajectories. Normal behavior should not be changed.
3. Training objective is the combination of both.

On two multi-hop web-browsing benchmarks (FRAMES, BrowseComp), the three takeaways are: 
1. peak KV cache and KV memory shrinks with lower accuracy loss 
2. heuristics frequently cause the model to never finish, while SideQuest finishes tasks as reliably as the uncompressed model
3. in a real serving engine (SGLang, one H100) this translates to +84% throughput and −37% total runtime.

## 2. Motivation

Save cost. Save context limit...

## 3. Math

### 3.1 Problem setup (ReAct loop, what a "cursor" is)

- A user query $$q$$ is answered by a policy $$\pi_\theta$$ over turns $$t = 1, 2, \dots, T$$. At turn $$t$$ the model emits $$r_t$$ (a reasoning trace plus either a tool call or the final answer); if it is a tool call, the environment returns a tool output $$o_t$$.
- **Cursor.** In gpt-oss's browser tool, every tool output is stored under an integer ID: `[Cursor c]` is the text block returned by the $$c$$-th tool call (a page of search results, or the content of an opened web page). Later tool calls *refer to* cursors, e.g. `browser.open(cursor=0, id=1)` opens link 1 from search-result block 0, and final answers cite them. So a cursor is simply **an addressable span of tokens = one tool call + its response**, and the paper's eviction unit is the cursor, not the individual token. Let $$\mathcal{C}_t$$ be the set of cursors created up to turn $$t$$ and $$\mathcal{O}_t \subseteq \mathcal{C}_t$$ the *open* (not yet evicted) ones.
- Context at turn $$t$$: $$C_t = [\,q, r_1, o_1, \dots, r_{t-1}, o_{t-1}\,]$$ minus evicted cursor spans. The KV cache holds exactly the tokens of $$C_t$$ (plus the tokens being generated).
- **Goal.** Minimize peak cache $$P = \max_t \vert C_t\vert $$ and total decode reads $$R = \sum_{s} \vert C^{(s)}\vert $$ (cache size summed over every decoded token $$s$$), while keeping task accuracy $$\Pr[\text{final answer correct}]$$ close to the uncompressed model. Only tool outputs $$o_t$$ are eligible for eviction; the model's own reasoning $$r_t$$ is never pruned.

### 3.2 Inference: the auxiliary thread (Algorithm 1, simplified)

With trigger interval $$K$$ and trigger phrase $$p$$ = "**Memory management mode**":

$$
\text{every } K \text{ turns:}\quad
C^{\text{aux}} = C_t \oplus p,\qquad
a = \pi_\theta(\cdot \mid C^{\text{aux}}),\qquad
a = [\,\text{reasoning}\,;\ \{\texttt{del\_cursors}: \Delta\}\,],\ \Delta \subseteq \mathcal{O}_t
$$

The aux thread shares the KV prefix $$C_t$$ with the main thread (no duplication; shared-prefix kernels such as Cascade Inference / FastTree make its marginal cost small). The main thread continues generating $$r_t$$ concurrently. When the aux thread finishes and the main thread's current turn ends:

$$
C \leftarrow C \setminus \{\,o_c : c \in \Delta\,\},\qquad \mathcal{O} \leftarrow \mathcal{O} \setminus \Delta,
$$

i.e. the KV entries of those spans are freed. The aux tokens $$a$$ are discarded, so the main context never contains management text.

### 3.3 Training data: hindsight labeling (Algorithm 2, re-notated)

Run the base model $$\pi_{\text{base}}$$ on browsing tasks; keep only traces $$\tau = (q, r_1, o_1, \dots, r_T)$$ whose final answer is correct.

**Last-use index.** For each cursor $$c$$,

$$
\ell_c = \max\{\,t : r_t \text{ references } c\,\}
$$

(references = a tool call reading from $$c$$, or a citation of $$c$$ in the final answer). Cursor $$c$$ is *expired* at turn $$t$$ iff $$\ell_c < t$$: hindsight tells us it is never used again. Expired set: $$E_t = \{c \in \mathcal{C}_t : \ell_c < t\}$$.

**Auxiliary samples.** For every $$t \in \{k, 2k, 3k, \dots\}$$ ($$k = 4$$ in the paper):

1. Randomly split $$E_t = D_t \sqcup S_t$$, where $$D_t$$ = cursors pretended to be *already deleted* (their tokens are masked out of the input, simulating a partially compressed cache) and $$S_t$$ = expired cursors still open, i.e. the **correct deletion targets**. Open set seen by the model: $$\mathcal{O}_t = \mathcal{C}_t \setminus D_t$$. *(The paper calls these $$\mathcal{C}_{\text{closed}}$$ and $$\mathcal{C}_{\text{open}}$$, which is confusing because "open" there means "to be closed now".)*
2. Ask an annotation model $$\mathcal{M}$$ (gpt-oss-120b, prompt in Appendix A) to write a short justification $$\rho_t$$ given the partial trace $$\tau_{1:t}$$, $$\mathcal{O}_t$$ and the answer $$S_t$$, without mentioning that it was given the answer.
3. Training pair: input $$x_t = \text{mask}_{D_t}(\tau_{1:t}) \oplus p$$, target $$y_t = \rho_t \oplus \{\texttt{del\_cursors}: S_t\}$$.

**Main samples.** The full correct trace $$\tau$$ with the base model's logits $$z = \pi_{\text{base}}(\tau)$$ stored at every position (upsampled $$3\times$$).

**What a training pair concretely looks like (worked example).** Take the Fig. 1 task. Suppose the base model solved it correctly in 20 turns, and the trace begins:

```
[user]     What's the first long weekend after GTC 2026?
[model]    need GTC dates → browser.search("GTC 2026 dates")
[Cursor 0] search results: link0 nvidia.com/gtc, link1 GTC schedule…
[model]    open the second link → browser.open(cursor=0, id=1)
[Cursor 1] GTC 2026 schedule page: March 16–19, San Jose…
[model]    GTC is Mar 16–19; now find 2026 long weekends → browser.search("long weekends 2026")
[Cursor 2] search results: …
```

*Step 1 — hindsight labels.* Because the whole trace exists, we can look up when each cursor was last referenced: Cursor 0 is only ever used by the `open(cursor=0,…)` call in turn 2, so $$\ell_0 = 2$$; Cursor 1 is cited as `[1]` in the final answer at turn 20, so $$\ell_1 = 20$$.

*Step 2 — cut the trace at turn $$t=4$$.* Expired cursors are those with $$\ell_c < 4$$, i.e. $$E_4 = \{0\}$$. This is the correct deletion target — nobody had to annotate it.

*Step 3 — the pair.*

- **Input $$x_4$$** = the trace above (turns 1–4) with the trigger phrase appended: `… [Cursor 2] search results: … **Memory management mode**`
- **Target $$y_4$$** = a short justification written by the annotation model, followed by the command:
  `Open cursors: [0,1,2]. The GTC dates were already obtained from Cursor 1, so the search page Cursor 0 is no longer needed. Cursor 1 will be cited in the final answer; Cursor 2 was just opened. {del_cursors: [0]}`

The model is trained with ordinary cross-entropy to produce $$y_4$$ given $$x_4$$ — exactly like SFT, except the input is an agent trajectory truncated mid-task and the output is a garbage-collection decision. Repeating the cut at $$t = 8, 12, 16, \dots$$ gives one pair per cut, e.g.

| cut $$t$$ | input $$x_t$$ | target $$y_t$$ |
|---|---|---|
| 4 | turns 1–4 + trigger | reasoning + `{del_cursors: [0]}` |
| 8 | turns 1–8 + trigger | reasoning + `{del_cursors: [2,3]}` |
| 12 | turns 1–12 + trigger | reasoning + `{del_cursors: [4]}` |

215 correct traces × ~6 cuts each ≈ the paper's 1,274 auxiliary pairs. Two details: (i) the reasoning text is a "teaching demonstration" written by gpt-oss-120b, which sees the partial trace *and* the answer set $$S_t$$ but is told not to mention that; the `del_cursors` command is the hard label. (ii) The random split $$E_t = D_t \sqcup S_t$$ means that in some pairs a few expired cursors (those in $$D_t$$) are simply erased from the input, so the model also learns to operate on a context that has already been partially compressed.

A **main pair** is much simpler: input = the full original trace with *no* trigger phrase; target = the base model's own next-token distribution at every position, matched with a KL loss. It teaches nothing new — it only pins the model to its original behavior so that, without the trigger, nothing changes. In one line: **aux pairs teach "when triggered, garbage-collect"; main pairs guarantee "when not triggered, behave exactly as before."**

### 3.4 Objective

$$
\mathcal{L}(\theta) =
\underbrace{\mathbb{E}_{(x,y)\in\mathcal{D}_{\text{aux}}}\big[-\log \pi_\theta(y \mid x)\big]}_{\text{learn to delete when triggered}}
\;+\;
\lambda\,
\underbrace{\mathbb{E}_{\tau\in\mathcal{D}_{\text{main}}}\sum_i \mathrm{KL}\!\big(\pi_{\text{base}}(\cdot\mid\tau_{<i})\,\big\Vert \,\pi_\theta(\cdot\mid\tau_{<i})\big)}_{\text{do not change normal behavior}},
\qquad \lambda = 500 .
$$

Cross-entropy on aux traces teaches the new mode; logit distillation on main traces pins the model to its original distribution on ordinary turns (prevents forgetting / spurious mode switching). Only LoRA adapters are trained.

### 3.5 Evaluation metrics

| Metric | Definition | Why |
|---|---|---|
| Peak token utilization | $$P = \max_t \lvert C_t\rvert$$ | worst-case memory per request ⇒ max batch size |
| KV cache memory reads | $$R = \sum_s \lvert C^{(s)}\rvert$$ over decode steps | decode is bandwidth-bound ⇒ latency/throughput proxy |
| Accuracy | fraction of tasks with correct final answer | utility |
| Non-completion rate | fraction of runs ending in unparsable output, context overflow, or turn limit | reliability of the reasoning loop |
| Serving metrics | throughput (tok/s), normalized peak KV occupancy, total benchmark runtime in SGLang | real-system impact |

## 4. Results and analysis

**Organization.** All experiments use one model, **gpt-oss-20b** (natively trained with the cursor-indexed browser tool), at *medium* and *high* reasoning effort, on two multi-hop browsing benchmarks with deterministic local search backends (Qwen3-Embedding-8B dense retrieval): **FRAMES** (424 questions over 6.4M Wikipedia articles, in-distribution because training data came from 400 other FRAMES tasks) and **BrowseComp** (500 questions over the 100k-document BrowseComp-Plus corpus, out-of-distribution). SideQuest's fine-tune: 215 correct traces → 215 main + 1,274 aux samples, LoRA $$r{=}8,\ \alpha{=}16$$ on `gate_up_proj`/`down_proj` of MoE layers 7, 15, 23 only, 3 epochs, lr $$2\times10^{-4}$$, $$\lambda=500$$, quantization-aware training for mxfp4. Baselines: uncompressed full attention, and H2O, SnapKV, R-KV each at 16k and 24k token budgets, all implemented in SGLang. The paper then answers three questions in order: **(A)** does model-driven eviction give a better accuracy-vs-memory trade-off than heuristics? **(B)** why do heuristics lose—do they break the agent loop? **(C)** does the memory saving turn into real serving gains?

### Part A — Efficiency vs. utility (Fig. 3)

Setup: for each of the 4 (benchmark × effort) settings, plot accuracy against peak token usage and against KV reads; SideQuest uses no budget (it sets its own), heuristics are fixed at 16k/24k.

- **Takeaway 1: large memory savings at small accuracy cost.** Peak tokens drop 56–65% and KV reads 53–71% vs. the uncompressed baseline; accuracy drops ≤2 pts on FRAMES and ~5 pts on BrowseComp.
- **Takeaway 2: heuristics are dominated.** At comparable or *smaller* memory, H2O/SnapKV/R-KV lose 5–25 accuracy points; SideQuest sits alone on the Pareto frontier.
- **Takeaway 3: no budget tuning needed.** Task token counts range from a few thousand to >120k (Fig. 2), so any fixed budget is either wasteful or fatal; SideQuest adapts per query by evicting only semantically stale cursors.

Approximate values read off Fig. 3 (peak tokens; heuristic range spans all three methods at both budgets):

| Setting | Baseline acc / peak | SideQuest acc / peak | Heuristics acc (peak 12–25k) |
|---|---|---|---|
| FRAMES, medium | ~65% / ~26k | ~63% / ~11k | 54–60% |
| FRAMES, high | ~66% / ~40k | ~66% / ~15k | 47–58% |
| BrowseComp, medium | ~40% / ~52k | ~33% / ~20k | 16–30% |
| BrowseComp, high | ~41% / ~78k | ~36% / ~27k | 12–22% |

### Part B — Robustness of the reasoning loop (Fig. 4)

Setup: same runs; count runs that never produce a parsable final answer, split by cause (unparsable/non-terminating output, context-length limit, turn limit).

- **Takeaway 1: heuristic pruning causes model collapse.** 15–37% of FRAMES runs and 20–67% of BrowseComp runs under H2O/SnapKV/R-KV end in unparsable output—the pruning removes tokens the model needs for syntactic/logical coherence.
- **Takeaway 2: SideQuest is as reliable as no compression.** Non-completion stays near the baseline (≈0–3% on FRAMES, ≈10% on BrowseComp-high, where the baseline mostly hits the context limit).
- **Takeaway 3: some heuristic "savings" are fake.** Their low peak memory in BrowseComp-high is partly because the runs crash early.

| Method | FRAMES non-completion | BrowseComp non-completion | dominant cause |
|---|---|---|---|
| Baseline | ~0–3% | ~1–11% | context limit |
| SideQuest | ~0–3% | ~1–9% | context limit |
| H2O / SnapKV / R-KV | 15–37% | 20–67% | unparsable output |

### Part C — Serving efficiency (Fig. 5)

Setup: FRAMES, medium effort, 424 tasks, gpt-oss-20b in SGLang on **one H100**, sweeping concurrent batch size 21–39; SideQuest vs. uncompressed only.

- **Takeaway 1: higher concurrency.** The baseline saturates memory at batch 24 and throughput collapses beyond it; SideQuest scales to batch 36.
- **Takeaway 2: throughput and runtime.** Peak throughput +83.9% (828 → 1,523 tok/s); total benchmark time −36.8% (2,356 s → 1,489 s), despite paying for the auxiliary thread.
- **Takeaway 3: memory headroom.** Normalized peak KV occupancy 0.977 → 0.450 (−53.9%), matching the Part A savings in a real engine.

| Metric | Baseline | SideQuest | Change |
|---|---|---|---|
| Peak throughput (tok/s) | 828 | 1,523 | +83.9% |
| Peak KV occupancy (norm.) | 0.977 (BS 24) | 0.450 (BS 36) | −53.9% |
| Benchmark runtime (s) | 2,356 | 1,489 | −36.8% |

## 5. Three biggest limitations (AI assessment)

**1. Generality is untested: one model, one tool format, one eviction granularity.** Everything hinges on gpt-oss's browser tool, which already packages every tool response as an integer-addressed `[Cursor c]` block that the model natively references and cites. That makes both the eviction unit and the hindsight labels ("last turn a cursor is referenced") nearly free. Models or tools without such addressable, explicitly cited outputs (most function-calling APIs, coding agents reading files, multi-modal tools) would need a new interface and a new labeling rule; the paper's "domain-agnostic" claim is untested. Eviction is also coarse (whole cursor or nothing) and excludes the model's own reasoning tokens, which can dominate the context at high reasoning effort, so the ceiling of the method on thought-heavy agents is unknown.

**2. The baselines are set up to fail, and the key design choices are not ablated.** Heuristics are run at fixed 16k/24k budgets on tasks needing up to 120k tokens, so much of their collapse (unparsable outputs, crashes) reflects an impossibly small budget rather than the attention heuristic itself; no budget sweep or "budget = SideQuest's realized peak" comparison is given. More importantly, several cheap, non-learned alternatives that would isolate *what* actually helps are missing: e.g., "delete a search-results cursor once a link from it has been opened," "keep only the last $$N$$ cursors," or an untrained model prompted to do the same job. There is also no sequential (single-thread) SideQuest variant, so the claimed advantages of the parallel architecture (latency, no pollution) are argued, not measured; and no sensitivity study on the trigger interval $$K$$, $$\lambda$$, or the random partition.

**3. Irreversibility and the limits of the evidence for "minimal degradation."** Eviction is permanent: if the auxiliary thread misjudges (its decision is made from a snapshot of the context and applied after the main thread has moved on), the information is gone with no re-fetch or recovery mechanism, and the paper never analyzes eviction precision/recall or what happens on failure cases. The training signal is a simplifying proxy (a cursor is "expired" only if never *explicitly* referenced again, footnote 2) learned from just 215 traces, and the out-of-distribution BrowseComp loss of ~5 absolute points on ~40% accuracy is a ~12% relative drop—non-trivial for a benchmark meant to test hard research. Finally, the auxiliary thread's compute/latency cost is folded into end-to-end runtime rather than measured, so we do not know how much of the gain shared-prefix kernels give versus lose.

**For me, I think the training pipeline does not learn how to continue to reason with KV cache eviction (main agent data never contains KV cache eviction). The ground truth in auxiliary may not be generalized to more scenarios (like coding has no "cursor"). Also, the farthest mention ignores implicit reasoning. Even if we can find the farthest future mentions of the current step, steps after the farthest mention can still be the implicit reasoning of this step.**

## 6. How to reproduce

**Step 1 — Environment.** Serve gpt-oss-20b (medium/high effort) in SGLang with its native browser tool. Build two deterministic retrieval backends exposing `search()`/`open()`: a Wikipedia dump (~6.4M articles, plus the ground-truth articles for each FRAMES question) and the BrowseComp-Plus 100k-document corpus, both embedded with Qwen3-Embedding-8B for dense search.

**Step 2 — Generate training data (Algorithm 2).** Run the base model on 400 FRAMES tasks (disjoint from the 424 evaluation tasks), keep the 215 correct traces, compute each cursor's last-use index from tool calls and final citations, and at every $$k = 4$$ turns randomly split the expired cursors into masked-out vs. to-delete sets. Use gpt-oss-120b with the Appendix A prompt to write the justification, prepend the trigger phrase, and append the delete command (1,274 aux samples). Record the base model's logits over the 215 main traces and upsample them $$3\times$$.

**Step 3 — Fine-tune.** LoRA ($$r=8$$, $$\alpha=16$$) only on `gate_up_proj` and `down_proj` of the MoE layers at depths 7, 15, 23; 3 epochs; lr $$2\times10^{-4}$$; loss = CE on aux samples + $$500\times$$ logit-distillation KL on main samples; quantization-aware training so the merged model runs in mxfp4.

**Step 4 — Inference runtime.** Modify SGLang to (a) fork a request sharing the prefix every $$K$$ turns with the trigger appended, (b) parse `{del_cursors: [...]}`, and (c) free the KV blocks of those cursor spans after the main thread's current turn finishes (prefix caching must tolerate holes). Implement H2O, SnapKV and R-KV in the same engine at 16k/24k budgets.

**Step 5 — Evaluate.** 424 FRAMES and 500 BrowseComp tasks at both reasoning efforts; log per-turn cache size for peak/reads, judge answer correctness, and count non-completions; for serving numbers, sweep batch size 21–39 on one H100 and record throughput, normalized KV occupancy and total runtime.

**Missing for exact reproduction.** No code, checkpoints, training data, or evaluation subsets are released (the paper does not mention any repository). Unspecified: the inference-time trigger interval $$K$$ (only the data-generation $$k=4$$ is given); the random-partition probability; the distillation temperature and training batch size/hardware; the Wikipedia dump date and which 424/500 questions were sampled; the correctness judge (exact match vs. LLM grader); the turn and context limits used in Fig. 4; the SGLang implementation details of the three heuristic baselines with prefix caching; and the annotation model's sampling settings. Expect to reimplement the fork/evict runtime and the baselines yourself.
