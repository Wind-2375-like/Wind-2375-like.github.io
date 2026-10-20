---
layout: post
title: 'Reading Notes of Selective Context: Compressing Context to Enhance Inference Efficiency of Large Language Models'
date: 2026-09-12 23:30:00
description: 'TL;DR: using perplexity on a small LLM as the heuristic. The assumption is that context tokens with lower surprisal can be deleted. Not really optimize the policy model to minimize the cost.'
tags: nlp efficient-inference
categories: ReadingNotes
thumbnail: assets/img/blogs/selective-context.png
toc:
  sidebar: left
related_posts: false
---

TL;DR: using perplexity on a small LLM as the heuristic. The assumption is that context tokens with lower surprisal can be deleted. Not really optimize the policy model to minimize the cost.

<!--more-->

> **Reference**: https://arxiv.org/abs/2310.06201

## 1. Summary

LLMs get slow, memory-hungry, and eventually truncate input when the context is long. The paper proposes **Selective Context**: before sending text to an LLM, score every noun phrase by how *surprising* it is to a small language model (self-information, $$-\log p$$), and delete the least surprising fraction. Tested on arXiv papers, BBC news, and ShareGPT chats with GPT-3.5/4, LLaMA-7/13/30B, and Vicuna-7/13B on summarisation, QA, conversation, and context reconstruction. Takeaways: (1) removing 50% of phrases cuts GPU memory 36% and per-token latency 32% while BERTScore-F1 drops only 0.023; (2) it clearly beats random deletion; (3) phrase-level pruning works best, and QA/summarisation degrade least.

## 2. Motivation

Attention cost grows quadratically in context length, and models have fixed windows. Prior fixes either change the architecture (sparse/local attention: Sparse Transformer, Longformer) or train the model to compress prompts into soft tokens (Gist, AutoCompressor); both require modifying or retraining the LLM. The authors argue the *input itself* is redundant — natural language repeats itself, and some content is already memorised by the LLM (Fig. 1) — so pruning input is a training-free, model-agnostic, complementary route. Weak point: the paper never argues *why* surprisal under a small, query-agnostic LM should track what the target LLM needs to answer a specific query; that link is assumed, not motivated.

## 3. Math

**Setup.** A context is a token sequence $$C=(x_1,\dots,x_n)$$ and a task query $$q$$ (a question, "summarise", or the last user turn). Goal: produce a shorter $$C'\subset C$$ such that $$\text{LLM}(C',q)\approx\text{LLM}(C,q)$$, at lower cost.

**Step 1 — token self-information** with a *small* causal scorer $$M$$ (LLaMA-7B or GPT-3 `curie`, not the target LLM):

$$
I(x_t) = -\log_2 P_M\!\left(x_t \mid x_{<t}\right).
$$

Implementation detail: $$x_{<t}$$ is truncated to the *current sentence* (scoring is done sentence by sentence), because the authors observed that later tokens in a long document get systematically low surprisal.

**Step 2 — lexical units.** Segment $$C$$ into non-overlapping spans $$u_1,\dots,u_m$$ (spaCy noun chunks for phrase-level; NLTK sentences for sentence-level; single tokens for token-level). By additivity of log-probabilities,

$$
I(u_j)=\sum_{x_t\in u_j} I(x_t).
$$

(The paper's Eq. 4 writes $$I(x_0)I(x_1)$$; it should be a sum.)

**Step 3 — percentile filtering.** Given compression ratio $$r\in\{0.2,0.35,0.5,0.65,0.8\}$$ (fraction of units to drop), set the threshold to the $$r$$-quantile of unit scores and keep everything above it:

$$
\tau_r=\text{Quantile}_r\big(\{I(u_j)\}_{j=1}^m\big),\qquad
C'=\big(u_j : I(u_j)\ge\tau_r\big)\ \text{in original order.}
$$

Note this is per-document adaptive, and drops exactly $$r$$ of *units*, not of *tokens* (at $$r=0.5$$ phrase-level, 42.8% of tokens are removed in the example).

**Evaluation.** There is no gold answer. The reference is the LLM's own full-context output:

$$
y^*=\text{LLM}(C,q),\qquad y=\text{LLM}(C',q),\qquad \text{score}=\text{sim}(y,y^*)
$$

with $$\text{sim}\in\{\text{BLEU},\text{METEOR},\text{ROUGE-1/2/L},\text{BERTScore P/R/F1}\}$$. With greedy decoding one reference is used; with temperature 0.7, $$K=4$$ references $$\{y^*_k\}$$ are sampled and the metric is computed against the set. For *reconstruction*, $$q$$ = "recover the original" and $$y^*=C$$ itself.

**Faithfulness** (manual, GPT-3.5 only): extract OpenIE triples $$T(y)$$ from the pruned-context answer and measure

$$
\text{Unfaithful}=\frac{\vert \{t\in T(y): y^*\not\models t\}\vert }{\vert T(y)\vert },
$$

with "Sorry, I don't know" answers counted separately as refusals.

## 4. Results and analysis

**Organisation and general setup.** All experiments compare the LLM's output on pruned context against its own output on full context. Data: 408 arXiv intros, 294 BBC articles, 470 ShareGPT conversations, all created after March 2023 (post-training-cutoff), all $$\le 2048$$ tokens. Target models: GPT-3.5, GPT-4, LLaMA-7B/13B/30B, Vicuna-7B/13B. Scorer: LLaMA-7B for open models, `curie` for OpenAI. Phrase-level units unless stated; $$r\in\{0.2,\dots,0.8\}$$. The paper studies three things: **(A)** the efficiency–quality trade-off as $$r$$ grows, and whether surprisal beats random pruning; **(B)** whether pruning makes answers unfaithful; **(C)** where the method works best — across tasks, unit granularity, and model scale/instruct-tuning.

### Part A — Trade-off vs. compression ratio, and vs. random deletion

Setup: Tables 1–2 average over all target models and all four tasks; Table 1 uses temperature 0.7 with 4 references, Table 2 uses greedy decoding and adds a random-deletion baseline. Efficiency numbers come from a single case study (Vicuna-13B, one arXiv paragraph, Fig. 6).

- **Takeaway 1: mild pruning is nearly free; heavy pruning is not.** At $$r=0.2$$, BLEU drops 0.05 and BERTScore-F1 0.007; at the headline $$r=0.5$$, BERTScore-F1 drops 0.023 but BLEU nearly halves (0.347→0.179) and ROUGE-1 drops 0.12. Beyond 0.5 the authors themselves call it "less valuable".
- **Takeaway 2: surprisal beats random.** With greedy decoding, Selective Context at $$r=0.5$$ (BERTScore-F1 0.900, ROUGE-1 0.642) roughly matches random deletion at $$r=0.2$$ (0.899, 0.666) — i.e. informed pruning buys ~2.5× more compression at equal quality.
- **Takeaway 3: real cost savings.** At $$r=0.5$$ on Vicuna-13B: CUDA memory 77.7 GB → 61.9 GB (−36%), latency 110.8 → 76.3 ms/token (1.32× faster); building the pruned context took 46 ms once.

| Method (greedy) | $$r$$ | BLEU | ROUGE-1 | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|---|
| Random deletion | 0.2 | 0.437 | 0.666 | 0.566 | 0.899 |
| Random deletion | 0.5 | 0.283 | 0.576 | 0.432 | 0.873 |
| Selective Context | 0.2 | 0.527 | 0.714 | 0.631 | 0.931 |
| Selective Context | 0.5 | 0.350 | 0.642 | 0.501 | 0.900 |
| Selective Context | 0.8 | 0.160 | 0.464 | 0.319 | 0.866 |

| Temp 0.7, avg. all models | $$r$$ | BLEU (Δ) | ROUGE-1 (Δ) | BERTScore-F1 (Δ) |
|---|---|---|---|---|
| Full context | – | .347 | .571 | .909 |
| Selective | 0.2 | .295 (−.05) | .540 (−.03) | .902 (−.007) |
| Selective | 0.5 | .179 (−.17) | .449 (−.12) | .887 (−.023) |
| Selective | 0.8 | .070 (−.28) | .311 (−.26) | .863 (−.047) |

### Part B — Faithfulness

Setup: 1000 QA pairs (200 per ratio), GPT-3.5 only, manual entailment check of OpenIE triples against the full-context answer.

- **Takeaway 1: pruning rarely causes hallucinated facts.** Unfaithful triples stay at 2.7–5.1% for $$r\le0.65$$, 8.6% at $$r=0.8$$.
- **Takeaway 2: the model instead gets shorter or refuses.** Refusals go 0 → 19 (9.5%) at $$r=0.65$$ and 27 at $$r=0.8$$; answer length shrinks from 160 to 131 tokens (−18%) at 0.65 and 104 at 0.8. So the failure mode is *omission*, not *fabrication*.

| $$r$$ | #Sorry / 200 | Answer len. | Unfaithful |
|---|---|---|---|
| Full | 0 | 160.3 | – |
| 0.2 | 0 | 156.5 | .027 |
| 0.5 | 4 | 140.2 | .038 |
| 0.65 | 19 | 131.2 | .051 |
| 0.8 | 27 | 103.7 | .086 |

### Part C — Where it works: tasks, granularity, model scale

Setup: Fig. 3 splits Table 1 by task; Fig. 5 compares token/phrase/sentence units by BERTScore-F1; Fig. 4 is a human study (3 students, 1150 summaries, ~55 per model×ratio) of summary acceptance for LLaMA and Vicuna.

- **Takeaway 1: tasks that need the gist survive; tasks that need the exact text don't.** Summarisation and QA curves are highest and decline gradually; reconstruction drops steepest (it asks the model to recover the deleted words), conversation is lowest throughout (already low BLEU ≈ 0.15 at $$r=0.2$$, reflecting open-ended replies).
- **Takeaway 2: phrases > tokens > sentences.** Phrase-level is consistently best (BERTScore-F1 ≈ 0.93→0.87 over ratios); token-level is ~0.03 lower; sentence-level is unstable (~0.83, non-monotone) — deleting whole sentences loses too much at once.
- **Takeaway 3: scale doesn't help, instruct-tuning does.** Human acceptance shows no trend across 7B/13B/30B, but Vicuna (instruct-tuned) is accepted ~0.8–0.9 vs LLaMA ~0.3–0.45 at low ratios; base LLaMA "goes wild" on ungrammatical pruned input.

## 5. Three biggest limitations (AI assessment)

**The evaluation measures self-consistency, not correctness, and the only baseline is random deletion.** Every metric compares the pruned-context output to the *full-context* output of the same model, so a model that is equally wrong both times scores perfectly, and a model that gives a better, shorter answer on pruned context is penalised. n-gram metrics (BLEU/ROUGE) additionally punish legitimate paraphrase, which explains why BERTScore looks flat while BLEU halves. The faithfulness study partly addresses this but only for one model and one task. There is no comparison to obvious alternatives — extractive summarisation, TF-IDF/stopword removal, or the trained compressors (Gist, AutoCompressor) the intro positions against — so we cannot tell whether surprisal is a *good* importance signal or merely better than noise.

**Surprisal is query-agnostic and computed by a different, smaller model, sentence by sentence.** What the scorer finds surprising (rare names, numbers, typos, formulas) is not what a specific question needs; what it finds predictable (a topic sentence the target LLM would still need as an anchor) gets deleted — the case study shows exactly this, with the background paragraph pruned away. Because scoring is done per sentence, cross-sentence redundancy (the paper's own motivating example, "I did get the groceries") is largely *invisible* to the scorer: a repeated fact in a new sentence looks surprising again. And the output is ungrammatical word salad (Fig. 2), which is why base LLaMA fails; the method implicitly relies on instruct-tuned models being robust to broken input.

**The headline claims outrun the experiments.** The paper motivates itself with long documents and truncation, but caps all inputs at 2048 tokens, so the regime where the method matters most is never tested. The 36%/32% efficiency numbers are from one paragraph on one model, and the scorer cost is reported as a fixed 46 ms rather than as what it is — a full forward pass of a 7B model over the entire context, which scales with the very length the method is meant to tame. At the advertised $$r=0.5$$, BLEU falls from 0.35 to 0.18 and 2–10% of QA answers become refusals; "comparable performance" is a generous description, and the safe operating point the data actually supports is $$r\approx0.2$$–$$0.35$$, where savings are modest.

## 6. How to reproduce

**Code and data.** [github.com/liyucheng09/Selective_Context](https://github.com/liyucheng09/Selective_Context) (the authors state code and data are released there); a longer prior version is arXiv:2304.12102.

**Step 1 — build the datasets.** Collect arXiv intro+background sections, full BBC articles, and ShareGPT conversations created after March 2023, filtered to ≤2048 tokens (targets: 408 / 294 / 470 documents). *Missing:* the exact crawling/filtering scripts and how multilingual ShareGPT conversations were handled are not described; if the released data differs, counts will not match.

**Step 2 — score and prune.** Run the scorer (LLaMA-7B for open targets; OpenAI `curie` for GPT targets) sentence by sentence to get per-token $$-\log_2 p$$; merge into spaCy noun chunks (`merge_noun_chunks`) or NLTK sentences; sum; drop units below the $$r$$-quantile; rejoin in order. *Missing / blocking:* `curie` has been retired by OpenAI, so the GPT-side scoring cannot be replicated exactly; tokenizer alignment between scorer and spaCy spans is not specified.

**Step 3 — generate with target LLMs.** Prompt GPT-3.5/GPT-4 (versions released before 30 March 2023), LLaMA-7/13/30B, Vicuna-7/13B on four tasks with both $$C$$ and $$C'$$; greedy for Table 2, temperature 0.7 with 4 full-context samples for Table 1. *Missing:* the actual prompts for each task, how QA questions/answers were generated from the original context, the exact OpenAI snapshot names (which are no longer served), and hardware for the timing numbers.

**Step 4 — evaluate.** Compute BLEU, METEOR, ROUGE-1/2/L, BERTScore against the full-context references; for faithfulness, extract OpenIE triples from GPT-3.5 QA answers and manually check entailment against the reference; for the human study, have annotators accept/reject ~1150 summaries. *Missing:* metric implementations/versions, the OpenIE tool, annotator instructions and agreement, and "Table 5" for lexical units is referenced but the paper only contains Figure 5.
