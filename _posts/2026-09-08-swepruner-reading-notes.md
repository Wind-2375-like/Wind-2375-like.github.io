---
layout: post
title: 'Reading Notes of SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents'
date: 2026-09-08 23:25:00
description: 'Train a smaller model to predict what lines to be deleted for the read action of coding agent. The pruned part is sent back to the agent. The training pipeline is not clear.'
tags: nlp llm-agents
categories: ReadingNotes
thumbnail: assets/img/blogs/swepruner.png
toc:
  sidebar: left
related_posts: false
---

Train a smaller model to predict what lines to be deleted for the read action of coding agent. The pruned part is sent back to the agent. The training pipeline is not clear.

<!--more-->

> **Reference**: https://arxiv.org/abs/2601.16746

## 1. Summary

For coding agent, most cost occurs from the read action

![image.png](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img20260908222851.png)

The authors trained a smaller model that given the current goal, predicts what lines of code should be deleted. The pruned context is sent to the coding agent. Training data is annotated by teacher models, including the hint as a question and reference lines to answer the question, so the rest should be deleted (should be an enclosing structure).

**The pipeline is not very clear.**

The experiments then claim they answer three questions:
1. does it actually help a real agent? yes
2. is the design better than other compression ideas? yes
3. does the skimmer generalise beyond agents? yes

## 2. Motivation

The problem is chosen empirically: tracing Mini-SWE-Agent on SWE-Bench Verified, read-type tool calls consume 76.1% of tokens (Claude Sonnet 4.5) and 67.5% (GLM-4.6), and these observations accumulate across rounds, so the cheapest lever is to shrink *what the environment returns*, not the agent's own reasoning. Existing prompt-compression methods come from NLP and prune **tokens** by perplexity/self-information (LLMLingua, Selective-Context); on code this destroys syntax (the paper measures 0.3–12% AST validity after such pruning) and the agent cannot read the result. Abstractive summarisation loses exact identifiers and line-level detail needed for edits, and adds an LLM call. Retrieval (RAG) is coarse (function-level chunks) and misses the specific lines. Code-specific compressors such as LongCodeZip keep structure but are **task-agnostic and static**: they compress the same way regardless of what the agent is currently looking for, and were only tested on single-turn tasks. Agent-history compressors (ACON, AgentFold, …) compress *past turns*, a different and orthogonal problem. The gap is therefore: a compressor that (a) is conditioned on the agent's *current* goal, (b) keeps whole lines so structure survives, and (c) is cheap enough to run on every tool call. The analogy is a programmer skimming a file with a question in mind. The motivation is coherent, with one soft spot: the 76% figure shows that reads are *large*, not that they are *wasteful*; that reads contain removable noise is only demonstrated after the fact by the results, and the claim that task-agnostic methods "fail" for agents rests on a single 50-example comparison (Table 3).

## 3. Math

I use one notation throughout and rename two symbols from the paper: the paper's retention threshold and its compression ratio both use $$\tau$$; here the threshold is $$\theta$$ and the compression ratio is $$\rho$$.

### 3.1 Problem setup

At agent round $$t$$ the agent emits a shell command $$a_t$$ and optionally a goal hint $$q_t$$ (a self-contained natural-language question). The environment returns raw text $$C_t$$. A pruner $$P$$ produces the observation the agent actually sees:

$$
o_t = \begin{cases} P(q_t, C_t) & \text{if } q_t \text{ given and } \vert C_t\vert  \ge 500 \text{ chars}\\ C_t & \text{otherwise (bypass)}\end{cases}
$$

Write $$C$$ as tokens $$x_1,\dots,x_n$$ and as lines $$\ell_1,\dots,\ell_m$$; let $$T_j\subseteq\{1..n\}$$ be the token indices of line $$\ell_j$$. $$P$$ must return a **subset of lines** $$\hat C \subseteq \{\ell_1..\ell_m\}$$ in original order (dropped runs are replaced by a `[pruned]` marker), so the agent never sees rewritten or partial lines.

### 3.2 Model (the "neural skimmer")

Backbone: Qwen3-Reranker-0.6B (a decoder-only reranker). Input is the concatenation `[instruction ; q ; C]`.

1. **Multi-layer fusion.** Take hidden states from layers 7, 14, 28 and fuse them:

   $$
   h_i=\mathrm{Attn}\big(\mathrm{SelfAttn}([h_i^{(7)};h_i^{(14)};h_i^{(28)}])\big)\in\mathbb R^{256},\quad i=1..n
   $$

(self-attention block, then 8-head attention, hidden 256). **self-attention twice. first project long vector to dim 256, then self attention, then split to 8 heads and do self attention again? then merge? details unclear.**

2. **Pruning head (linear-chain CRF over tokens).** Labels $$y_i\in\{0,1\}$$ (0 = prune, 1 = retain). Emission logits $$E_i=\mathrm{MLP}(h_i)\in\mathbb R^2$$, transition matrix $$A\in\mathbb R^{2\times2}$$, start/end vectors $$b,c\in\mathbb R^2$$:

   $$
   \mathrm{score}(C,y)=b_{y_1}+\sum_{i=1}^{n}E_{i,y_i}+\sum_{i=2}^{n}A_{y_{i-1},y_i}+c_{y_n},\qquad p(y\mid q,C)=\frac{e^{\mathrm{score}(C,y)}}{Z(C)},\; Z(C)=\sum_{y'}e^{\mathrm{score}(C,y')}
   $$

The transition term is what makes neighbouring tokens (and hence neighbouring lines) tend to be kept or dropped together. **here y is a sequence of labels (y1, ..., yn). Z(C) is the sum of all possible $$2^n$$ sequences.** Use DP to compute it in $$O(n^{2}.2^2)$$ time. since this is a sequence classification, they use CRF.

3. **Reranking head.** The original LM head of the reranker reads the final layer and outputs a document-level relevance $$\hat S=\sigma(\text{logit}_{\text{yes}}-\text{logit}_{\text{no}})\in[0,1]$$, i.e. "is this whole chunk relevant to $$q$$ at all?" **compare this with WHAT???**

Only the last two backbone layers, the fusion module, the MLP and the CRF are trained; the rest is frozen.

### 3.3 Training data (silver labels)

Each example is a quadruple $$(q, C, M, S)$$: hint, code snippet, line mask $$M\in\{0,1\}^m$$, document relevance $$S\in[0,1]$$. Construction:

1. Sample 200K snippets $$C$$ from the GitHub-Code-2025 corpus (195,370 files, 5,945 repos).
2. Teacher LLM (Qwen3-Coder-30B-A3B-Instruct, $$T{=}0.7$$, top-$$p$$ 0.9) writes a hint $$q$$ for one of **9 task types** (summarise, refactor, find-relevant-part, optimise, locate, explain, debug, feature-addition, code-completion — for the last one $$q$$ is itself code), stratified over 3 snippet lengths × 3 relevance levels.
3. The teacher answers $$q$$ **citing line ranges** (prompt in Appendix J); the cited lines become $$M_j=1$$, all others $$M_j=0$$. The prompt explicitly asks the teacher to also cite enclosing structure (`def`, `class`, `if/else` headers) so the kept lines parse.
4. A judge LLM (Qwen3-Next-80B-A3B-Thinking) rates query quality / deletion relevance / semantic preservation and keeps only "high" — ≈1/6 survive → **61,184** examples.

Token labels are inherited from lines: $$y_i = M_{j(i)}$$ where $$j(i)$$ is the line containing token $$i$$. (How $$S$$ is produced numerically is not stated; it is presumably derived from the teacher's relevance level.)

### 3.4 Training objective

Per-example CRF negative log-likelihood, normalised by length so long snippets don't push the model toward pruning everything, plus an MSE on document relevance:

$$
\mathcal L_{\text{prune}}=\frac1B\sum_{k=1}^{B}\frac{\log Z(C_k)-\mathrm{score}(C_k,y_k)}{n_k},\qquad
\mathcal L_{\text{rerank}}=\frac1B\sum_{k=1}^{B}(\hat S_k-S_k)^2
$$

$$
\mathcal L=(1-\lambda)\,\mathcal L_{\text{prune}}+\lambda\,\mathcal L_{\text{rerank}},\qquad \lambda=0.05
$$

AdamW, lr $$3\times10^{-5}$$, weight decay 0.01, dropout 0.4, batch 128 (16 × 8 GPUs), 3 epochs.

### 3.5 Inference (Algorithm 1, rewritten)

Given $$(q,C)$$:

1. Split $$C$$ into chunks of at most the model's window with 50-token overlap; run all chunks in parallel.
2. For each token compute a retain score $$s_i\in[0,1]$$ from the pruning head (the paper describes both Viterbi decoding — which gives $$s_i\in\{0,1\}$$ — and continuous token scores; the released code should be treated as ground truth for which is used).
3. Aggregate to lines by averaging:

   $$
   \bar s_j=\frac1{\vert T_j\vert }\sum_{i\in T_j}s_i
   $$

4. Keep line $$j$$ iff $$\bar s_j>\theta$$ with $$\theta=0.5$$ (chosen on a held-out validation set). Output kept lines in order with `[pruned]` markers for gaps.

The agent-side contract (Appendix J): hints must be full questions, must not contain file names/line numbers, and agents are told to read with `cat -n` / `nl -ba` so that if something important was pruned they can `sed` the exact line range back *without* a hint.

### 3.6 Evaluation metrics

- Compression ratio $$\rho=\vert C\vert /\vert \hat C\vert $$ (tokens). Baselines are *configured* to hit $$\rho\approx4$$ or $$8$$; SWE-Pruner's $$\rho$$ is whatever $$\theta=0.5$$ yields.
- Long Code Completion: Edit Similarity (ES) and Exact Match (EM) of the completed line.
- Long Code QA: accuracy.
- SWE-Bench Verified: resolve rate (patch passes all hidden tests).
- SWE-QA: LLM-as-judge score (0–10) over correctness/completeness/relevance/clarity/reasoning.
- Efficiency: total tokens, agent rounds, API cost ($), skimmer time-to-first-token (ms).
- Structure: fraction of pruned files that still parse with tree-sitter (AST correctness).

## 4. Results and analysis

**Organisation.** The paper has four experimental parts. **Part A** (the headline) plugs SWE-Pruner into real agents on multi-turn benchmarks and reports task success plus efficiency. **Part B** asks whether the *design choices* matter by swapping in alternative compressors inside the same agent on a 50-issue subset. **Part C** removes the agent and tests the skimmer as a stand-alone compressor on two single-turn long-context benchmarks against the same baselines at fixed compression budgets. **Part D** checks the two things that could kill practicality: the skimmer's own latency and whether the pruned code still parses; two case studies illustrate the mechanism. Common setup: skimmer = fine-tuned Qwen3-Reranker-0.6B, $$\theta=0.5$$, pruning applied only to outputs ≥500 chars; agent backbones called at temperature 0; agent step limit 250.

### Part A — Multi-turn agent tasks

Setup: **SWE-Bench Verified** (500 issues, 12 Python repos, Docker test grading) with **Mini-SWE-Agent** (tools: `cat`/`grep`, `sed`, shell), and **SWE-QA** (repo-level QA on streamlink, reflex, conan; LLM-judge scored) with **OpenHands**. Backbones: **Claude Sonnet 4.5** and **GLM-4.6**. Baseline = same agent without pruning (and without the hint field in the prompt).

- **Takeaway 1 — Large, model-agnostic token/cost savings with no loss (slight gain) in success.** Tokens drop 23% (Claude) and 38% (GLM); rounds drop 18% and 26%; solved issues go from 353→360 and 277→283. Cost falls 27–36%. Both prompt *and* completion tokens fall ~40% (Fig. 7), i.e. the agent also *writes* less because it needs fewer exploratory steps.

| SWE-Bench Verified | Rounds | Solved | Success % | Tokens (M) | Cost ($) |
|---|---|---|---|---|---|
| Mini-SWE-Agent (Claude Sonnet 4.5) | 51.0 | 353/500 | 70.6 | 0.911 | 0.504 |
| + SWE-Pruner | 41.7 | 360/500 | 72.0 (+1.4) | 0.701 (−23.1%) | 0.369 (−26.8%) |
| Mini-SWE-Agent (GLM-4.6) | 49.3 | 277/500 | 55.4 | 0.791 | 0.055 |
| + SWE-Pruner | 36.6 | 283/500 | 56.6 (+1.2) | 0.488 (−38.3%) | 0.035 (−36.4%) |

- **Takeaway 2 — Savings transfer to a different agent framework and task type, but round counts are model-dependent.** On SWE-QA, tokens fall 9–20% for Claude and 29–54% for GLM with judge scores within ±0.23. GLM, however, takes 29–41% *more* rounds after pruning (it explores more files when each read is smaller); the token total still falls because each round is much cheaper.

| SWE-QA (tokens K / score) | Streamlink | Reflex | Conan |
|---|---|---|---|
| Claude 4.5 → +Pruner | 611→557 (−9%), 8.36→8.59 | 1082→867 (−20%), 8.68→8.85 | 655→521 (−20%), 8.70→8.84 |
| GLM-4.6 → +Pruner | 318→145 (−54%), 8.56→8.56 | 142→101 (−29%), 8.37→8.23 | 176→117 (−34%), 8.58→8.45 |

- **Takeaway 3 — The mechanism is fewer redundant reads, not shorter files per se.** Case study `django-10554`: baseline exhausts 164 steps / 7.0M tokens (peak prompt 87.8K) and fails; pruned agent solves it in 56 steps / 1.17M tokens (−83%). Case `django-11740`: both succeed; pruned agent uses 6 more steps but 6% fewer tokens and 30% lower peak prompt, and skips the throw-away test scripts the baseline writes.

### Part B — Which design choices matter? (Table 3)

Setup: random 50-issue subset of SWE-Bench Verified, Mini-SWE-Agent + Claude Sonnet 4.5, each alternative compressor applied to the same file-read observations.

- **Takeaway — line-level + task-aware is the only variant that improves success; everything else trades accuracy for tokens.** Token-level pruning (LLMLingua-2) and chunk retrieval (RAG) save tokens but drop success to 54% / 50%; LLM summarisation gives 56% and adds a model call; LongCodeZip 54%. SWE-Pruner reaches 64% (vs 62% baseline) with the *lowest* token count.

| Method (50 issues) | Rounds | Success % | Tokens (M) |
|---|---|---|---|
| Mini-SWE-Agent | 52.3 | 62.0 | 0.972 |
| + LLMLingua-2 | 42.1 | 54.0 | 0.856 |
| + RAG | 40.2 | 50.0 | 0.771 |
| + LLM Summarize | 41.3 | 56.0 | 0.794 |
| + LongCodeZip | 44.3 | 54.0 | 0.889 |
| + SWE-Pruner | 41.1 | **64.0** | **0.670** |

### Part C — Single-turn long-context tasks (Table 4, Table 7)

Setup: **Long Code Completion** (500 Python examples, >5K-token contexts; ES/EM) and **Long Code QA** (up to 1M-token contexts; accuracy), answered by **Qwen2.5-Coder-7B-Instruct** (main) and **Seed-Coder-8B-Instruct** (Appendix G). The task question is the hint. Baselines: Full, No-Context, Selective-Context, LLMLingua-2, RAG (UniXCoder function chunks), LongCodeZip, each tuned to a 4× or 8× budget.

- **Takeaway 1 — At a nominal budget SWE-Pruner compresses far harder than the budget and still matches/beats every baseline.** Under the "8×" setting it reaches 10.9× on completion (57.6 ES / 31.0 EM vs Full 64.7 / 40.5) and 14.8× on QA with 58.7% accuracy — *above* the full-context 54.1%, presumably because removing distractors helps a 7B model.
- **Takeaway 2 — Token-level methods collapse as the budget tightens; line-level methods degrade gracefully.** Selective-Context falls to 48.7 ES and LLMLingua-2 to 44.7 ES at 8× (close to the 44.9 no-context floor); RAG and LongCodeZip hold ~56–58 ES but at lower compression.
- **Takeaway 3 — Same pattern with a second reader model** (Seed-Coder-8B): 8.1× / 56.7 ES on completion, 14.7× / 55.8% on QA, again best in class.

| Qwen2.5-Coder-7B | LCC 4×: ρ / ES / EM | LCC 8×: ρ / ES / EM | LCQA 4×: ρ / Acc | LCQA 8×: ρ / Acc |
|---|---|---|---|---|
| Full context | 1.0 / 64.65 / 40.5 | — | 1.0 / 54.05 | — |
| No context | ∞ / 44.90 / 13.5 | — | ∞ / 38.39 | — |
| Selective-Context | 3.27 / 52.48 / 22.0 | 7.49 / 48.67 / 17.0 | 3.69 / 55.36 | 7.32 / 51.79 |
| LLMLingua-2 | 3.32 / 49.47 / 15.5 | 7.89 / 44.74 / 13.0 | 3.57 / 55.36 | 7.68 / 51.33 |
| RAG | 3.29 / 58.97 / 30.5 | 6.60 / 55.82 / 29.0 | 3.06 / 58.04 | 5.87 / 55.86 |
| LongCodeZip | 2.77 / 57.77 / 28.0 | 7.85 / 56.08 / 27.5 | 3.98 / 52.25 | 7.39 / 54.95 |
| **SWE-Pruner** | **5.56** / 58.63 / **31.5** | **10.92** / **57.58** / **31.0** | **13.95** / **59.46** | **14.84** / **58.71** |

### Part D — Overhead and structure preservation

Setup: time-to-first-token of the skimmer vs. Qwen3 generative models (0.6B–32B) at 64–8192 input tokens (Table 6); AST validity via tree-sitter on Long Code Completion outputs (Table 8).

- **Latency is flat and small:** 42–49 ms up to 2K tokens, 102 ms at 8K, vs 1189 ms for Qwen3-32B; the paper argues this is <10% of a typical closed-model API round trip and is repaid by the 23–54% token cut.
- **Line-level pruning keeps code parseable:** AST-valid rate 87.3% for SWE-Pruner (applied after function RAG, whose own rate is 92.3%) vs 78.2% for random line dropping, 49.6% random token dropping, 12.4% Selective-Context, 0.29% LLMLingua-2. Applied on top of LongCodeZip it drops to 76.8%, showing that stacking compressors costs validity.

## 5. Three biggest limitations (AI assessment)

**1. The headline "improves success" claim is not statistically established, and the baseline is confounded.** The gains are +7/500 and +6/500 solved issues (+1.2–1.4 pp), well inside the run-to-run variance normally seen on SWE-Bench Verified (a few points), and no confidence intervals or per-seed numbers are reported ("averaged over three seeds where applicable" is never made concrete). The Part B comparison that supports the design argument uses only 50 issues (one issue = 2 pp). More subtly, the pruned agent runs with a *different system prompt* — it is told to "prefer reading files fully with `cat -n`" and to read widely before editing — so "baseline vs +SWE-Pruner" mixes the effect of the pruner with the effect of prompt engineering. A clean ablation (same prompt, hints ignored) is missing. The safe reading of the paper is "large token savings at no measurable cost", not "improves accuracy".

**2. "Self-adaptive" is really "a fixed 0.5 threshold on a learned score", and the resulting compression is neither controllable nor error-analysed.** The method cannot target a budget: at a nominal "4×" constraint it compresses LongCodeQA 13.95×, so the Table 4 comparisons are not equal-budget (they happen to favour SWE-Pruner, but the reverse could occur on other data — heavier pruning is not automatically fine). In the agent setting the agent never learns *what* was removed beyond a `[pruned]` marker, and the paper gives no precision/recall of the skimmer against ground-truth needed lines on real agent observations (grep hits, directory listings, tracebacks, multi-file `cat` output), which look nothing like the single-snippet training data. The only proxy is downstream success. The supervision itself is silver: a 30B teacher's line citations filtered by another LLM, with no human check, and the document-relevance label $$S$$ is never defined. The safety valve is the agent noticing a gap and re-reading with `sed` — which reintroduces rounds and tokens and is exactly what GLM-4.6 does on SWE-QA (+29–41% rounds).

**3. Narrow scope and a plausible train/test overlap the paper only half-addresses.** Everything is Python; the "polyglot" corpus is not evaluated on any other language. Training snippets come from GitHub repos with >2 stars collected in 2025, which very plausibly include django, sympy, matplotlib, etc. — the very repositories SWE-Bench Verified is built from — and the teacher-generated "debug/locate" hints resemble issue descriptions; leakage is discussed only for SWE-QA. The method also leaves the real long-horizon problem untouched: pruned observations still accumulate in context forever, so peak prompt length only drops ~30% and the claim that SWE-Pruner is "orthogonal and combinable" with history-compression methods is asserted, not tested. Finally, the skimmer's 8K-token window forces chunking with 50-token overlap, so cross-chunk dependencies (a class header in chunk 1, its method in chunk 3) can be scored inconsistently; no analysis of chunk-boundary effects is given.

**I think the pipeline has a lot of engineering things without any rationales. The self-attention part is unclear. The CRF is very indirect. Why not predict token labels directly in the language model to know whether or not to keep that token? The reranker training is unclear.**

## 6. How to reproduce

**Artifacts (verified on the GitHub README as of Sept 2026).** Code: `github.com/Ayanami1314/swe-pruner` (MIT). Trained skimmer: HuggingFace `ayanami-kitasan/code-pruner` (also vendored under `swe-pruner/model` via git-lfs); pip package `swe-pruner`. Training set (61K Python examples) and the SWE-Bench trajectories for both backbones are on Google Drive links in the README. Subfolders: `train/` (training guide), `downstream_eval/{multi_turn,single_turn}/`, `examples/` (OpenHands and Claude-Agent-SDK integrations), `utils/`.

**Step 1 — Serve the pruner.** Download the checkpoint, install per `swe-pruner/README.md` (uses `uv`), and start the HTTP server; the agent config expects `http://localhost:8000/prune` with `timeout 120`, `retries 3`, `min_chars 500`, `chunk_overlap_tokens 50`, `threshold 0.5`. A single GPU is enough for inference (0.6B model).

**Step 2 — Multi-turn agent evaluation.** Install Mini-SWE-Agent, replace its YAML with the Appendix-J template (adds the `<context_focus_question>` field, `step_limit 250`, `cost_limit 3`), point it at the pruner URL, and run SWE-Bench Verified in Docker with Claude Sonnet 4.5 or GLM-4.6 API keys at temperature 0; grade with the official harness. Expect ~$0.37–0.50 per instance for Claude (~\$200–250 per full run per configuration). For SWE-QA, use `downstream_eval/multi_turn/sweqa` with OpenHands and the `_BASH_DESCRIPTION_PRUNE` tool description. The README recommends Slurm with ≥4 GPUs for their scripts.

**Step 3 — Single-turn evaluation.** `downstream_eval/single_turn` covers Long Code Completion and LongCodeQA with Qwen2.5-Coder-7B-Instruct / Seed-Coder-8B-Instruct; baselines (LLMLingua-2, Selective-Context, UniXCoder RAG, LongCodeZip) must be installed separately and tuned to the 4×/8× budgets.

**Step 4 — Retrain from scratch (optional).** Sample snippets from HF `nick007x/github-code-2025`, generate hints and line citations with Qwen3-Coder-30B-A3B-Instruct (Appendix J "Silver Label" prompt, $$T{=}0.7$$, top-$$p$$ 0.9, 9 task types × 3 lengths × 3 relevance levels), filter with Qwen3-Next-80B-A3B-Thinking (Quality Evaluation prompt, keep "high"), then fine-tune Qwen3-Reranker-0.6B with the hyperparameters in §3.4 on 8 GPUs (`train/README.md`). Or skip data generation and use the released 61K set.

**Missing or under-specified.** (i) The held-out validation set used to pick $$\theta=0.5$$; (ii) the 50-instance subset IDs used in Table 3 and the seeds; (iii) how the document-relevance target $$S$$ is computed; (iv) the chunk size fed to the skimmer and how chunk scores are merged in the overlap; (v) whether inference uses Viterbi labels or continuous marginals for $$s_i$$; (vi) exact API model snapshots/dates for Claude Sonnet 4.5 and GLM-4.6 (results will drift as providers update models); (vii) the AST-validity experiment's exact protocol; (viii) baseline implementations for the agent-side comparison (LLMLingua-2/RAG/LongCodeZip wrapped as middleware) do not appear to be in the repo tree. Multilingual training data is described but only Python data is released.
