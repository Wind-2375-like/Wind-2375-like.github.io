---
layout: post
title: 'Reading Notes of DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference'
date: 2026-09-08 23:40:00
description: 'Last layer hidden state already has information related to uncertainty.'
tags: nlp efficient-inference
categories: ReadingNotes
thumbnail: assets/img/blogs/diffadapt.png
toc:
  sidebar: left
related_posts: false
---

Last layer hidden state already has information related to uncertainty.

<!--more-->

> **Reference**: https://arxiv.org/abs/2510.19669

## 1. Summary

Reasoning LLMs (DeepSeek-R1-style models) write a long "thinking" trace before every answer, which wastes tokens on easy questions and does not rescue them on questions beyond their ability. The paper first measures the model's per-token uncertainty (entropy) versus problem difficulty and finds a **U-shape**: easy problems are solved correctly yet with high entropy (the model "dithers"), medium problems have low entropy, and hard problems have high entropy because the model is genuinely lost. It then defines three fixed inference recipes — **Easy** (short prompt, low temperature, 40% of the token budget), **Normal** (full budget), **Hard** (a "fail-fast" recipe with 50% budget) — and shows via an **oracle** (pick the cheapest recipe that is correct per question) that adaptive selection could raise accuracy by +7.2% on Qwen3-4B while cutting tokens roughly in half. **DiffAdapt** approximates this oracle with a tiny 2-layer MLP **probe** trained on the frozen LLM's last-layer hidden state after reading the question; the probe outputs Easy/Normal/Hard, and the chosen recipe is used to generate. Evaluated on 5 models (Qwen3-4B, DS-R1-Qwen-7B, DS-R1-Llama-8B, Nemotron-1.5B, ThinkPrune-7B) and 8 benchmarks (GSM8K, MATH500, AIME24/25, OlympiadBench; OOD: Minerva, GPQA, MMLU-Pro), the three takeaways are: (i) DiffAdapt matches or beats the best fixed recipe at every token budget, in- and out-of-domain; (ii) it saves 10–22% tokens versus always-Normal (DEER, a training-free early-exit baseline, *increases* tokens by 27–53%), and cuts wall-clock time ~6×; (iii) it is cheap and robust — thresholds transfer across model families with ≈0.3% loss, 30% of training data suffices, and a blind LLM judge finds truncation-caused failures in only 2% of cases.

---

## 2. Motivation

**Why this problem.** Test-time scaling ("think longer") is expensive, and a uniform thinking budget is a bad allocation: easy problems overspend and hard problems often fail regardless of budget. **Existing work** falls into three camps: (a) *training-based* budget control (RL with length penalties — ThinkPrune, LC-R1, AdaCoT, Thinkless), which requires costly retraining of the LLM; (b) *training-free inference-time* control (DEER's confidence-based early exit, AlphaOne), which is flexible but relies on generic confidence signals that, the paper argues, generalize poorly and can even inflate token usage; (c) *auxiliary-model switchers* (a separate BERT or 7B model deciding when to think), which add non-trivial inference overhead. **Why DiffAdapt.** The paper's claim is that difficulty is already encoded in the LLM's own hidden state after reading the question, so a few-thousand-parameter probe can route each question to one of three pre-tuned recipes at negligible cost, with no LLM retraining and full compatibility with vLLM-style batching/caching. **Caveat on the motivation.** The "U-shaped entropy ⇒ overthinking" story is the paper's headline empirical finding, but the link is interpretive: high average token entropy on easy problems shows the model has many equally plausible continuations, not directly that it *spends unnecessary tokens*; the actual evidence that easy problems need fewer tokens comes from the oracle experiment (Section 4), not from the entropy curve. The entropy analysis therefore motivates the *three-region* framing more than it justifies the specific recipes.

---

## 3. Math

### 3.1 Setup and notation

| Symbol | Meaning |
|---|---|
| $$x$$ | a question; $$y$$ its ground-truth answer |
| $$\pi$$ | the frozen reasoning LLM |
| $$r \sim \pi(\cdot \mid x; \tau, L)$$ | a sampled trace with temperature $$\tau$$ and max length $$L$$ tokens |
| $$\mathrm{ans}(r)$$ | the final answer extracted from trace $$r$$ |
| $$L_{\max}$$ | per-benchmark maximum token budget (Appendix Table 13) |
| $$\mathcal{S}=\{\mathrm{E},\mathrm{N},\mathrm{H}\}$$ | the three inference strategies (Easy / Normal / Hard) |

### 3.2 Difficulty statistics of a question (Section 3 & Stage 1)

Sample $$n=10$$ traces $$r_1,\dots,r_n \sim \pi(\cdot\mid x;\ \tau{=}0.6,\ L{=}32\text{K})$$.

**Correctness rate**

$$
C(x) \;=\; \frac{1}{n}\sum_{j=1}^{n} \mathbb{1}\big[\mathrm{ans}(r_j)=y\big] \in [0,1].
$$

**Generation entropy.** At decoding step $$t$$ with next-token distribution $$p_t(\cdot)$$ over vocabulary $$V$$,

$$
H_t \;=\; -\sum_{v\in V} p_t(v)\log p_t(v), \qquad
\bar H(r) \;=\; \frac{1}{\vert r\vert }\sum_{t=1}^{\vert r\vert } H_t, \qquad
\bar H(x) \;=\; \frac{1}{n}\sum_{j=1}^{n}\bar H(r_j).
$$

$$\bar H(x)$$ is the model's average per-token uncertainty on $$x$$. Note it is the entropy of the *distribution* at each step, not a property of the sampled token.

**The U-shape finding.** Bucketing DeepMath-103K questions by GPT-4o difficulty $$d\in\{1,\dots,10\}$$, $$\mathbb{E}[\bar H \mid d]$$ is high for $$d\le 2$$, drops 22–25% to a minimum around $$d\approx 4$$, then rises again for $$d\ge 8$$; meanwhile $$\mathbb{E}[C\mid d]$$ decreases monotonically.

### 3.3 Inference strategies

Each strategy $$s\in\mathcal{S}$$ is a triple $$(\text{prompt}_s,\ \tau_s,\ \rho_s)$$: a fixed `<think>` prefix, a temperature, and a fraction of the budget. Generation under $$s$$ is $$r \sim \pi(\cdot \mid \text{prompt}_s \oplus x;\ \tau_s,\ \rho_s L_{\max})$$.

| $$s$$ | prompt (paraphrased) | $$\tau_s$$ | $$\rho_s$$ |
|---|---|---|---|
| E | "Straightforward — solve directly, double-check" | 0.5 | 0.4 |
| N | "Break into clear logical steps" (top-$$p$$ 0.95) | 0.8 | 1.0 |
| H | "Intricate — outline the main method, mind resources" (fail-fast) | 0.4 | 0.5 |

These were chosen by grid search over 125 combinations on MATH500: keep configs with accuracy $$\ge 95\%$$, pick the one with fewest average tokens.

### 3.4 Oracle (upper bound, Section 4)

For each $$x$$ generate one trace $$r_s(x)$$ under every $$s$$. Let $$\mathcal{S}^{+}(x)=\{s:\mathrm{ans}(r_s(x))=y\}$$.

$$
s^{\star}(x) \;=\;
\begin{cases}
\arg\min_{s\in\mathcal{S}^{+}(x)} \vert r_s(x)\vert  & \text{if } \mathcal{S}^{+}(x)\neq\varnothing\\[4pt]
\arg\min_{s\in\mathcal{S}} \vert r_s(x)\vert  & \text{otherwise.}
\end{cases}
$$

i.e. the cheapest correct strategy, else the cheapest one. On Qwen3-4B this assigns 82.3% E, 7.7% N, 10.0% H.

### 3.5 DiffAdapt pipeline

**Stage 1 — heuristic labels.** With per-model thresholds $$(\alpha,\beta,\gamma)$$ (e.g. Qwen3-4B: $$0.88, 0.32, 0.65$$; DeepSeek-R1 family: $$0.85, 0.35, 0.60$$):

$$
\ell(x)=
\begin{cases}
\mathrm{N} & C(x)\ge\alpha \ \wedge\ \bar H(x)\le\beta \quad (\text{confident \& correct: the "certainty region"})\\
\mathrm{H} & C(x)<\gamma \quad (\text{beyond capability})\\
\mathrm{E} & \text{otherwise} \quad (\text{correct but high-entropy: "overthinking"})
\end{cases}
$$

This yields a labeled set $$\mathcal{D}=\{(x_i,\ell(x_i))\}_{i=1}^{N}$$ with $$N=3000$$ (300 per difficulty level from DeepMath-103K). **Important:** these labels are *not* the oracle labels $$s^\star$$; they are a proxy derived from $$(C,\bar H)$$.

**Stage 2 — probe.** Let $$h(x)\in\mathbb{R}^{d}$$ be the last-layer hidden state at the final token after *prefilling* $$x$$ (no generation). The probe is a 2-layer MLP

$$
f_\phi(h)=\mathrm{softmax}\!\big(W_2\,\mathrm{ReLU}(W_1 h + b_1) + b_2\big)\in\Delta^{2},
\qquad
\mathcal{L}(\phi)=-\frac{1}{N}\sum_{i=1}^{N}\log f_\phi\big(h(x_i)\big)_{\ell(x_i)},
$$

trained with AdamW, lr $$10^{-3}$$, 100 epochs; $$\pi$$ stays frozen.

**Stage 3 — inference.**

$$
\hat s(x)=\arg\max_{s\in\mathcal{S}} f_\phi\big(h(x)\big)_s,
\qquad
r \sim \pi\big(\cdot \mid \text{prompt}_{\hat s}\oplus x;\ \tau_{\hat s},\ \rho_{\hat s}L_{\max}\big).
$$

The probe reads the prefill KV state that already exists, so it adds one MLP forward per request and does not touch decoding.

### 3.6 Evaluation metrics

- **Accuracy at budget $$b$$.** For $$b\in\{\tfrac13,\tfrac12,\tfrac23,\tfrac56,1\}$$, every method (fixed strategies, DEER, DiffAdapt) is run with $$L_{\max}\leftarrow b\,L_{\max}$$; report Avg@3 accuracy (mean over 3 runs) averaged over the in-domain or out-of-domain benchmark set. Curves of accuracy vs. $$b$$ are the main figures.
- **Token savings vs. always-Normal** over benchmark set $$\mathcal{B}$$:

  $$
  \mathrm{Savings}(\text{method})=\frac{1}{\vert \mathcal{B}\vert }\sum_{b\in\mathcal{B}}\frac{T_{b,\mathrm{N}}-T_{b,\text{method}}}{T_{b,\mathrm{N}}}\times 100\%,
  $$

where $$T_{b,\cdot}$$ is average tokens per problem on benchmark $$b$$. Negative means more tokens than Normal.
- **Latency:** end-to-end wall-clock minutes under vLLM.

---

## 4. Results and analysis

**Organization.** The experiments build a chain of four parts: **(A) Analysis** establishes the U-shaped entropy pattern that motivates three difficulty regions; **(B) Oracle** shows how much an ideal per-question strategy selector could gain, which sets the target; **(C) Main results** show that the learned probe (DiffAdapt) captures a large share of that gain against fixed strategies and DEER across models, budgets, and domains — including on models already RL-trained for brevity; **(D) Efficiency & robustness** quantify the token/latency savings and check that the design is not fragile (thresholds, probe architecture, data size, reasoning integrity).

**General setup.** Models: Qwen3-4B, DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Llama-8B (standard reasoning LLMs), plus Nemotron-Research-Reasoning-Qwen-1.5B and ThinkPrune-7B (length-control-RL models). Probe training data: 3,000 DeepMath-103K problems (300 per difficulty level), labeled by Stage 1 with the same model as proxy. Benchmarks: in-domain math (GSM8K, MATH500, AIME24, AIME25, OlympiadBench) and out-of-domain (Minerva, GPQA, MMLU-Pro). Per-benchmark $$L_{\max}$$ is set by observing the model's longest output under a 32K cap and rounding (e.g. Qwen3-4B: GSM8K 1,500; MATH 12,000; AIME 18,000; MMLU-Pro 3,000). Baselines: All-Easy, All-Normal, All-Hard, and DEER (think threshold 0.9, same $$L_{\max}$$). Evaluation follows LIMO/xVerify answer-checking protocols; each experiment is run 3 times.

### Part A — Overthinking analysis (Section 3, Appendix F)

*Setup.* DeepMath-103K, 300 questions per difficulty 1–10, 10 samples each at $$\tau=0.6$$, on DS-R1-Qwen-7B, DS-R1-Qwen-1.5B, Nemotron-1.5B. Plot $$\mathbb{E}[C\mid d]$$ and $$\mathbb{E}[\bar H\mid d]$$.

- **Takeaway A1 — entropy is U-shaped in difficulty while accuracy falls monotonically.** Entropy drops from difficulty 1–2 to a minimum at ~4, then rises past 8; correctness declines steadily. The paper names three regions: *overthinking* (high $$C$$, high $$\bar H$$), *certainty* (high $$C$$, low $$\bar H$$), *capability limit* (low $$C$$, high $$\bar H$$).

| Model | Entropy drop, easy → medium |
|---|---|
| DS-R1-Qwen-7B | ≈ 22–25% (Fig. 1a) |
| DS-R1-Qwen-1.5B | 23.3% |
| Nemotron-1.5B | 21.3% |

### Part B — Oracle upper bound (Section 4, Appendix D.3, G)

*Setup.* Qwen3-4B (plus DS-R1-Qwen-7B, Nemotron-1.5B, DS-R1-Llama-8B in appendix), all 8 benchmarks, $$L_{\max}=32$$K, the three strategies of Table 1, oracle = cheapest correct strategy.

- **Takeaway B1 — per-question strategy selection dominates every fixed strategy on every benchmark.** Oracle beats the best fixed strategy by +7.2% average accuracy on Qwen3-4B (+12.3%, +7.9%, +16.2% on the other three models) while using roughly half the tokens of Normal.
- **Takeaway B2 — most problems are best served by Easy.** 82.3% of questions get E, 7.7% N, 10.0% H, i.e. the token savings come overwhelmingly from not over-thinking easy questions, and the "Hard" bucket saves tokens by giving up early.

| Benchmark (Qwen3-4B) | Easy acc / tok | Normal acc / tok | Hard acc / tok | **Oracle acc / tok** |
|---|---|---|---|---|
| GSM8K | 89.9 / 170 | 93.2 / 562 | 93.5 / 512 | **96.2 / 198** |
| MATH500 | 82.8 / 718 | 96.2 / 2,662 | 93.4 / 2,425 | **98.0 / 1,279** |
| OlympiadBench | 51.0 / 1,629 | 73.6 / 6,722 | 68.6 / 5,896 | **76.9 / 2,756** |
| AIME24 | 16.7 / 3,505 | 60.0 / 10,672 | 46.7 / 10,167 | **66.7 / 4,429** |
| AIME25 | 16.7 / 2,443 | 53.3 / 13,733 | 40.0 / 10,635 | **56.7 / 4,675** |
| GPQA | 46.5 / 813 | 50.5 / 3,022 | 48.0 / 2,952 | **70.2 / 1,001** |
| MMLU-Pro | 60.4 / 526 | 65.4 / 2,076 | 62.5 / 1,873 | **74.6 / 690** |
| Minerva | 46.7 / 469 | 55.2 / 2,795 | 53.3 / 2,248 | **65.8 / 866** |

### Part C — DiffAdapt vs. fixed strategies and DEER (Section 6.2–6.3, Appendix D.1–D.2)

*Setup.* Accuracy-vs-budget curves at $$b\in\{33\%,50\%,67\%,83\%,100\%\}$$ of $$L_{\max}$$, averaged over the ID set and separately over the OOD set. Three reasoning LLMs (Fig. 3) and two LC-RL models (Fig. 4, DEER omitted). MMLU-Pro zero-shot transfer reported per budget (Table 9). A separate comparison plugs DiffAdapt into the ThinkLess 1.5B warm-up model and compares to the fully RL-trained ThinkLess (Table 10).

- **Takeaway C1 — DiffAdapt is on or above the upper envelope of the three fixed strategies at every budget, in-domain and OOD.** Fixed strategies have complementary shapes: Easy is strong at tiny budgets but plateaus; Normal/Hard keep improving with budget. DiffAdapt tracks the best of them and exceeds it at higher budgets. Gains are largest on models where the fixed strategies differ most (Qwen3-4B).
- **Takeaway C2 — DEER, the training-free early-exit baseline, is roughly competitive in-domain but degrades sharply OOD; DiffAdapt does not.** On OOD sets DEER sits 10–25 points below DiffAdapt for DS-R1-Llama-8B, indicating a learned difficulty signal generalizes better than a confidence-based exit rule.
- **Takeaway C3 — DiffAdapt is additive to length-control RL.** On Nemotron-1.5B and ThinkPrune-7B (already trained to be terse), All-Easy wins at low budgets, but DiffAdapt overtakes at 60–100% budgets. On ThinkLess, DiffAdapt on the *warm-up* model beats the RL-trained ThinkLess on GSM8K at every budget and matches it on MATH500 at 33–72% budgets with 35–50% fewer tokens (RL still wins at full budget: 73.5 vs 68.3).

| Qwen3-4B, in-domain avg. acc (%) | 33% | 50% | 67% | 83% | 100% |
|---|---|---|---|---|---|
| DiffAdapt (Table 5) | 59.3 | 67.2 | 72.4 | 75.8 | 76.8 |
| All-Normal (read from Fig. 3a) | ≈55 | ≈61 | ≈68 | ≈71 | ≈73 |
| All-Easy (Fig. 3a) | ≈51 | ≈51 | ≈51 | ≈51 | ≈51 |
| DEER (Fig. 3a) | ≈48 | ≈48 | ≈62 | ≈65 | ≈68 |

| MMLU-Pro (OOD, probe trained on math only) | 33% | 50% | 67% | 83% | 100% |
|---|---|---|---|---|---|
| DS-R1-Qwen-7B: DiffAdapt / Normal | 32.0 / 28.2 | 35.2 / 31.1 | 35.0 / 31.1 | 35.8 / 30.4 | 36.5 / 30.7 |
| DS-R1-Llama-8B: DiffAdapt / Normal | 22.1 / 17.5 | 31.3 / 27.9 | 33.7 / 31.8 | 35.5 / 32.1 | 39.9 / 32.5 |

### Part D — Efficiency and robustness (Sections 6.4, 7)

*Setup.* Token savings via the formula in §3.6 over all 8 benchmarks (Table 3). Latency on Qwen3-4B, first 40 OlympiadBench problems, batch 10, one A800, $$L_{\max}=32$$K (Table 4). Ablations on Qwen3-4B in-domain: transferred thresholds from the DeepSeek family, linear probe, 30% training data (Table 5). Reasoning-integrity check: 50 GSM8K questions, Qwen3-30B-A3B as blind pairwise judge, DiffAdapt vs. Normal (Tables 6–7).

- **Takeaway D1 — real token and latency savings; DEER actually spends more.** DiffAdapt saves 22.4% (Qwen3-4B), 9.7% (DS-R1-Qwen-7B), 10.1% (ThinkPrune-7B) vs. Normal; DEER uses 27.5% / 53.3% *more* tokens because its confidence rule frequently runs to the cap. Wall-clock: 64 min (vLLM) → 57 (DEER) → 10 (DiffAdapt), a ~6× speedup.
- **Takeaway D2 — the recipe is robust and data-cheap, but the nonlinearity matters.** Swapping in DeepSeek-family thresholds changes average accuracy by ~0.3 points; 30% of training data costs ~2.4 points on average (and is oddly *better* at the 33% budget); a linear head loses ~3.2 points, so the 2-layer MLP is needed.
- **Takeaway D3 — shorter traces are not broken traces.** The judge prefers DiffAdapt in 76% of pairs (citing conciseness); the baseline wins 12%, of which 5/6 are verbosity preference and only 1/50 (2%) is a genuine truncation-induced logic failure.

| Ablation (Qwen3-4B, ID, avg over budgets) | Avg acc (%) |
|---|---|
| DiffAdapt default | 70.9 |
| Transferred thresholds (DeepSeek-R1) | 71.2 |
| Linear probe | 67.7 |
| 30% training data | 68.5 |

---

## 5. Three biggest limitations (AI assessment)

**(1) The learned labels are not the oracle labels, and the headline "overthinking = high entropy" story is under-supported.** The oracle (§4) is defined operationally — cheapest strategy that gets the answer right — and is what actually justifies the method. But the probe is trained on a different, hand-designed rule over $$(C,\bar H)$$ with per-model thresholds: "Normal" is assigned to questions the model already answers confidently and correctly, which is exactly where one would expect the *shortest* recipe to suffice, while the residual "Easy" class absorbs everything else including mid-correctness questions. The paper never reports how well $$\ell(x)$$ agrees with $$s^\star(x)$$, how the probe's predicted class distribution compares to the oracle's 82/8/10 split, or the probe's own classification accuracy. Similarly, average per-token entropy being higher on easy questions is consistent with benign explanations (short answers dominated by free-form phrasing/verification tokens, or more valid solution paths), so the "U-shape ⇒ wasted compute" inference is interpretive; the causal evidence for savings comes entirely from the oracle table, not from the entropy analysis that the paper is framed around. This makes it hard to know *why* the probe works and whether the entropy component of the labels contributes at all (no ablation removes $$\beta$$).

**(2) "Hard" means "give up early," so the method is partly a triage system rather than a better reasoner, and the probe cannot see the reasoning.** The Hard recipe uses *half* the budget and a fail-fast prompt; its token savings come from abandoning the ~10% of questions the model is predicted to fail. That is a legitimate cost-control policy for serving, but it runs against the purpose of test-time scaling — the questions where extra compute might matter most are exactly the ones cut off — and it means accuracy gains on hard benchmarks (AIME) must come from the Easy/Normal routing of the remaining questions, which the paper does not decompose. Moreover, the probe reads only the prefill hidden state of the question; it has no access to how the solution is actually unfolding, and the appendix itself lists the resulting failure mode (misclassify-as-Easy → truncation) while offering only an untested fallback idea. A more defensible design would treat the probe's decision as a prior and let early generation signals revise it, which the authors defer to future work.

**(3) Evaluation rigor is uneven and several headline numbers do not line up.** Accuracy is reported as averages over benchmarks of wildly different size and difficulty (AIME has 30 questions; Avg@3 on it has huge variance) at *fractions* of a per-model, per-benchmark budget that the authors themselves set from observed maxima — a nonstandard protocol that makes cross-paper comparison impossible. Token savings (Table 3) are given for only 3 of 5 models and are never paired with the accuracy at that setting in a single table, so "comparable or improved accuracy while saving 22.4% tokens" cannot be checked directly from the paper. The abstract/intro alternate between "up to 22.4%" savings vs. Normal, "50%" (oracle), and "62%" (vs. DEER, which is inflated because DEER overspends), and between "over 10%" and "+7.2%" oracle accuracy gain. Ablations and the integrity study use one model (Qwen3-4B) and 50 GSM8K items, and the learned-switcher baselines the paper lists as most similar (ThinkSwitcher, Zhang et al. 2025b, budget-guidance BERT) are never compared against — only the training-free DEER and, in an appendix, ThinkLess.

---

## 6. How to reproduce

**Step 1 — Get the code and data (currently missing).** The reproducibility statement promises code, prompts, evaluation scripts, configs, seeds, and plotting scripts "upon publication," but the paper gives no repository URL and, as of this writing, no public DiffAdapt repository is easily findable; you would have to re-implement or contact the authors (xliu886@connect.hkust-gz.edu.cn). Everything else is public: DeepMath-103K (HuggingFace, with GPT-4o difficulty labels), the five model checkpoints (Qwen3-4B, DeepSeek-R1-Distill-Qwen-7B / Llama-8B, nvidia/Nemotron-Research-Reasoning-Qwen-1.5B, ThinkPrune-7B), the eight benchmarks, the LIMO evaluation harness and xVerify answer checker, and vLLM.

**Step 2 — Build the probe training set.** Sample 300 problems per difficulty level 1–10 from DeepMath-103K (3,000 total; the exact sampling seed and any filtering are not given). For each, generate $$n=10$$ traces with the Normal-style prompt at $$\tau=0.6$$, max 32K tokens, recording per-step next-token entropies; compute $$C(x)$$ and $$\bar H(x)$$. Label with the rule in §3.5 using Table 12 thresholds ($$\alpha,\beta,\gamma$$ = 0.88/0.32/0.65 for Qwen3-4B, 0.85/0.35/0.60 for the DeepSeek family; thresholds for Nemotron-1.5B and ThinkPrune-7B are **not reported**). The paper also does not state which prompt/`<think>` prefix was used during this labeling pass, nor the validation split used for the optional "sanity check."

**Step 3 — Train the probe.** Prefill each question, take the last-layer hidden state at the final token (which token position and whether any chat template is applied are unspecified), and train the 2-layer MLP (hidden width **not reported**) with cross-entropy, AdamW, lr $$10^{-3}$$, 100 epochs (batch size, weight decay, class balancing, and train/val split unspecified). For the ablations, repeat with a linear head and with a random 30% subset.

**Step 4 — Run adaptive inference.** For each benchmark and model, set $$L_{\max}$$ from Table 13, then for each budget fraction $$b$$ generate with the probe-selected strategy using the full prompts in Appendix C: Easy ($$\tau$$=0.5, $$0.4\,bL_{\max}$$), Normal ($$\tau$$=0.8, top-$$p$$=0.95, $$bL_{\max}$$), Hard ($$\tau$$=0.4, $$0.5\,bL_{\max}$$). The prompts are injected as a pre-filled `<think>` prefix; the exact concatenation with the model's chat template is not shown. Run three times and average; also run the three fixed strategies and DEER (threshold 0.9, same $$L_{\max}$$) as baselines.

**Step 5 — Evaluate and compare.** Score answers with the LIMO/xVerify protocol, compute Avg@3 accuracy per benchmark, average within the ID and OOD groups (it is unclear whether this average is per-benchmark unweighted or per-question), compute token savings with the §3.6 formula, and measure latency on the first 40 OlympiadBench problems (batch 10, single A800). For the oracle, generate one trace per strategy per question at 32K and apply the selection rule. For the integrity study, sample 50 GSM8K questions, pair DiffAdapt vs. Normal outputs, and prompt Qwen3-30B-A3B with the judge template in Appendix E.2 (the 50-item sample and the judge's decoding settings are not provided).
