---
layout: post
title: 'Reading Notes of Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine'
date: 2026-10-20 23:00:00
description: 'TL;DR: A "better" scorer for evaluating coding agents.'
tags: nlp llm-agents
categories: ReadingNotes
thumbnail: assets/img/blogs/hgm.png
toc:
  sidebar: left
related_posts: false
---

TL;DR: A "better" scorer for evaluating coding agents.

<!--more-->

> **Reference**: https://arxiv.org/pdf/2510.21614

## Key terms (read first)

| Term | Meaning |
|---|---|
| LLM (large language model) | A neural network that generates text and code, e.g., GPT-5, GPT-5-mini, Qwen3-Coder. |
| Coding agent | A program (a "scaffold") that wraps an LLM in a loop with tools such as a shell (bash) and a file editor, so it can read a code repository, edit files, run tests, and submit a fix. The LLM stays fixed; the agent's own code is what changes in this paper. |
| Benchmark, task, accuracy | A benchmark is a fixed set of coding problems (tasks). Hidden tests mark each attempt as pass (1) or fail (0). Accuracy is the fraction of tasks passed. |
| SWE-bench Verified / Lite | Real GitHub issues from 12 popular Python repositories; the agent must write a code patch that makes hidden tests pass. Verified has 500 human-checked tasks; Lite has 300 tasks, 93 of which also appear in Verified. SWE-Verified-60 is a 60-task subset of Verified. |
| Polyglot | A benchmark of programming exercises in C++, Java, Python, Go, Rust, and JavaScript, from the Aider coding-assistant project. |
| Self-modification | An agent edits a copy of its own source code. The edited copy is its *child*; the original is the *parent*. |
| Archive tree, clade | All agents created so far, linked parent → child, form a tree. The *clade* of an agent is that agent plus all of its descendants (its subtree). The word comes from biologist Julian Huxley: a lineage sharing one common ancestor. |
| Metaproductivity | How good an agent is at producing better future agents, as opposed to how good it is at solving tasks itself. |
| Gödel Machine | A theoretical self-rewriting program (Schmidhuber, 2003) that changes its own code only after proving that the change raises expected long-term utility. It is optimal in principle but needs formal proofs. |
| DGM, SICA | Darwin Gödel Machine and Self-Improving Coding Agent: the two prior self-improving coding-agent systems that HGM is compared against. |
| SWE-agent | A leading human-designed coding agent; it serves as the human reference. |
| Bayesian / Thompson sampling | Choosing among options by drawing one random plausible success rate per option from a belief distribution and taking the largest draw (made precise in the Math section). |
| Allocated CPU-hours | CPU cores reserved × wall-clock hours; the paper's measure of compute cost. |
| Checked leaderboard results | SWE-bench leaderboard entries whose results the benchmark maintainers have verified. |

## Summary

Self-improving coding agents (LLM programs that fix bugs) rewrite their own code, growing a family tree of versions; which to modify next? Prior methods pick the top benchmark scorer, but strong scorers can have weak descendants. The Huxley–Gödel Machine (HGM) rates each agent by pooled test success across its subtree ("clade") and chooses what to modify or test by Bayesian sampling. On SWE-bench Verified and Polyglot: (1) clade scores predict descendants' best far better; (2) HGM finds stronger agents with 2.4–6.9× fewer CPU-hours than DGM; (3) its GPT-5-mini-optimized agent, run with GPT-5, matches top human-designed agents on SWE-bench Lite.

## Motivation

Agents that improve their own improvement process could compound progress without human redesign. The Gödel Machine accepts only self-edits it can prove beneficial: optimal, but proofs are impractical. DGM and SICA instead edit the agents with the highest benchmark scores. This fails because a score measures current skill, not the quality of future descendants (the Metaproductivity–Performance Mismatch); scores from few tests are noisy; and testing every new child immediately wastes budget. HGM judges agents by their descendants' results, the quantity the Gödel Machine optimizes. The motivation is clear; its empirical evidence is correlational, from one run per method.

## Math

### (a) Prerequisites

**Notation used throughout** (simplified from the paper; the paper's $n_{\text{success}}(a)$ is $s(a)$ here, and its clade sum $n^{C}_{\text{success}}(a)$ is $S(a)$):

| Symbol | Meaning |
|---|---|
| $a$, $a_0$ | an agent; the initial (root) agent |
| $\mathcal{T}_t$ | the archive tree after step $t$; $\lvert\mathcal{T}_t\rvert$ is its number of agents |
| $\mathcal{C}(a)$ | the clade of $a$: $a$ plus all its descendants |
| $\mathcal{D}$ | the task set (benchmark) |
| $U(a)\in[0,1]$ | the true accuracy of agent $a$ on $\mathcal{D}$ (its *utility*) |
| $s(a)$, $f(a)$ | passes and fails observed for agent $a$ itself |
| $S(a)$, $F(a)$ | passes and fails summed over every agent in $\mathcal{C}(a)$ |
| $N_t$ | total number of single agent–task tests done so far |
| $B$, $b$ | total budget and remaining budget |
| $\pi$ | the search policy, i.e., the algorithm (SICA, DGM, or HGM) |

**P1 — Empirical accuracy.** *Intuition:* each test of agent $a$ is a coin flip whose unknown bias is $U(a)$; the share of heads estimates the bias.

$$\hat{p}(a)=\frac{s(a)}{s(a)+f(a)}$$

*Example:* 6 passes and 4 fails give $\hat{p}=6/10=0.6$.

**P2 — Beta belief (posterior).** *Intuition:* instead of a single number, keep a whole belief curve over the unknown success rate. A *posterior* is the belief after seeing data. It starts flat and narrows as tests accumulate. Starting from the flat $\mathrm{Beta}(1,1)$, after $s$ passes and $f$ fails the belief is $\mathrm{Beta}(1+s,\,1+f)$.

$$p\sim\mathrm{Beta}(x,y),\qquad \mathbb{E}[p]=\frac{x}{x+y},\qquad \mathrm{Var}[p]=\frac{xy}{(x+y)^2(x+y+1)}$$

*Example:* 3 passes, 7 fails → $\mathrm{Beta}(4,8)$: mean $4/12\approx0.333$, variance $32/(144\cdot13)\approx0.0171$, standard deviation $\approx0.131$.

**P3 — Thompson sampling.** *Intuition:* let every option "roll" a plausible success rate from its belief and pick the highest roll. A well-tested strong option wins most rounds; a barely tested option wins some rounds, which keeps exploring it.

$$\theta_i\sim\mathrm{Beta}(x_i,y_i)\ \text{for each option } i,\qquad \text{choose } \arg\max_i\theta_i$$

*Example:* option A has $\mathrm{Beta}(4,8)$ (mean 0.33, well tested), option B has $\mathrm{Beta}(2,2)$ (mean 0.5, barely tested). One round draws $\theta_A=0.29$, $\theta_B=0.71$ → B. Another round draws $0.41$ and $0.35$ → A.

**P4 — Bandits.** A *multi-armed bandit* is a row of slot machines ("arms") with unknown payout rates; each pull reveals one outcome. In an *infinite-armed bandit* you may also add brand-new machines. Here an arm is an agent, a pull is one test of that agent on one task, and adding an arm is creating a new agent by self-modification.

**P5 — Q-value and Bellman optimality.** *Intuition:* the Q-value of an action is the expected final payoff if you take that action now and then keep following policy $\pi$. *Bellman optimality* says that always taking the action with the larger Q-value (under the best continuation) is the optimal strategy.

$$Q^{\pi}(\text{state},x)=\mathbb{E}_{\pi}\big[\text{final reward}\mid\text{take action } x \text{ now}\big]$$

*Example:* "accept" leads to final reward 0.6 or 0.4 with equal chance, so $Q=0.5$; "reject" leads to 0.45 for sure, so $Q=0.45$. Accept.

**P6 — POMDP.** A *partially observable Markov decision process* is a sequential decision problem in which the decision-maker sees only part of the true state.

### (b) Problem setup

**Setting 1 — Self-improvement as tree search (all experiments).**

- *Input:* an initial agent $a_0$; a *self-modification LLM* that writes code edits to agents; a *task-solving LLM* that powers every agent while it attempts benchmark tasks; a benchmark $\mathcal{D}$; a budget $B$ counted in single agent–task tests (800 or 8000 in the experiments).
- *Each step*, the policy takes one action: **expand**$(a)$ — agent $a$ rewrites a copy of its own code, and the copy joins the tree as $a$'s child; or **evaluate**$(a)$ — run $a$ on one task it has not tried and record pass or fail.
- *Output:* when the budget is spent, the policy returns $a_{\text{final}}=\arg\max_{a\in\mathcal{T}_B}\mathrm{Score}(a)$ for its own scoring rule.
- *Objective.* *Intuition:* the expected true accuracy of the agent you hand in at the end.

$$J(\pi)=\mathbb{E}\big[U(a_{\text{final}})\big]$$

*Example:* if a method hands in an agent of true accuracy 0.55 in half of its possible runs and 0.60 in the other half, $J=0.575$.

- *How performance is measured:* the accuracy of $a_{\text{final}}$ on the benchmark, plus the compute spent (allocated CPU-hours).
- *Reference points:* the initial agent, DGM and SICA under the same test budget, and the human-designed SWE-agent.
- *Out of scope:* training or fine-tuning LLM weights (only agent code changes), rewards collected before the end, the general time-aware Gödel Machine, and the safety of executing self-written code.

**Setting 2 — The Gödel Machine game (theory only).** The paper's **Assumption 1**:

1. Only the final agent matters; there is no reward before the end.
2. *Repeatable trials:* an agent's expected result on a task does not depend on when it is tested or what happened before (each test runs in a freshly reset environment).
3. The Gödel Machine's prover knows the utility function, and proofs cost no budget.
4. Each self-modification costs exactly one budget unit, and the budget is finite and known.

*Game:* the state is (tree, current parent $p$, newest child $c$, remaining budget $b$); the decision-maker observes only $(p, c, b)$. Each step it chooses **accept** ($c$ becomes the parent) or **reject** ($p$ stays the parent); the new parent then produces a fresh child, and $b$ drops by one. When $b=0$ it outputs $p$ or $c$, and the reward is $U$ of that output.

**Setting 3 — Transfer test (final experiments).** No search: the finished agent is run on SWE-bench Lite, optionally with its task-solving LLM swapped from GPT-5-mini to GPT-5.

### (c) Evaluation metrics

**M1 — Accuracy** (Settings 1 and 3). *Intuition:* the share of benchmark tasks the agent solves; this is what the whole search tries to maximize.

$$\mathrm{Acc}(a)=\frac{1}{\lvert\mathcal{D}\rvert}\sum_{d\in\mathcal{D}}\mathbb{1}\big[a\text{ solves }d\big]$$

*Example:* 34 of 60 tasks solved → 56.7%.

**M2 — Allocated CPU-hours.** *Intuition:* how much compute the run reserved to finish its budget; lower means cheaper and faster. CPU-hours $=$ cores reserved $\times$ wall-clock hours. *Example:* 8 cores for 40 hours = 320 CPU-hours.

**M3 — Empirical CMP (the target for estimator quality).** *Intuition:* after a run ends, ask "how good did this agent's family actually get?" by taking the best observed accuracy among its descendants (excluding the agent itself).

$$\mathrm{CMP}^{\mathrm{emp}}(a)=\max_{a'\in\mathcal{C}(a)\setminus\{a\}}\hat{p}(a')$$

*Example:* $a$'s descendants scored 0.45, 0.58, and 0.52 → $\mathrm{CMP}^{\mathrm{emp}}(a)=0.58$.

**M4 — Guidance scores being tested.** SICA and DGM guide the search with an agent's own accuracy $\hat{p}(a)$. HGM's guide is its clade estimate (defined in part f), tested in a *leak-free* form. *Intuition:* pool the clade's results but remove (i) $a$'s own tests and (ii) the whole branch that contains the best descendant, so the predictor never sees the answer it is predicting. Let $b^{\star}$ be the child of $a$ whose clade contains $a$'s best-scoring descendant.

$$\widehat{\mathrm{CMP}}^{\,\mathrm{lf}}(a)=\frac{S(a)-s(a)-S(b^{\star})}{\big[S(a)+F(a)\big]-\big[s(a)+f(a)\big]-\big[S(b^{\star})+F(b^{\star})\big]}$$

*Example:* $a$ itself is 2/5; child $b_1$'s clade (holding the best descendant) totals 12 passes and 8 fails; child $b_2$'s clade totals 6 passes and 4 fails. Then $S(a)=20$, $F(a)=15$, and the estimate is $(20-2-12)/(35-5-20)=6/10=0.6$, i.e., only $b_2$'s branch is used.

**M5 — Pearson correlation (unweighted and weighted).** *Intuition:* does the guidance score rise when the target rises? $+1$ is perfect agreement, $0$ is none, $-1$ is opposite. The weighted version counts each agent by the number of tests $w_i$ behind its prediction, so well-supported predictions matter more; the unweighted version uses $w_i=1$.

$$r_w=\frac{\sum_i w_i(x_i-\bar{x}_w)(y_i-\bar{y}_w)}{\sqrt{\sum_i w_i(x_i-\bar{x}_w)^2}\,\sqrt{\sum_i w_i(y_i-\bar{y}_w)^2}},\qquad \bar{x}_w=\frac{\sum_i w_ix_i}{\sum_i w_i}$$

Here $x_i$ is the guidance score of agent $i$ and $y_i$ its empirical CMP. *Example (unweighted):* $x=(0.2,0.4,0.6)$, $y=(0.3,0.5,0.4)$. Both means are 0.4; deviations are $(-0.2,0,0.2)$ and $(-0.1,0.1,0)$. Then $r=0.02/\sqrt{0.08\times0.02}=0.02/0.04=0.5$.

**M6 — SWE-bench Lite, filtered vs. standard.** *Filtered* drops the 93 tasks shared with SWE-bench Verified, leaving 207 tasks the agent never saw during search; it measures pure transfer. *Standard* uses all 300 tasks, matching the public leaderboard. *Example:* 40.1% on filtered ≈ 83 of 207 tasks.

### (d) The standard approach and why it fails

The standard rule is "greedy by score": pick the current best scorer as the parent, then test its new child on a batch of tasks. *Intuition:* bet on whoever performs best today.

$$a_{\text{parent}}=\arg\max_{a\in\mathcal{T}_t}\hat{p}(a)$$

*Example:* agent A scores 0.55 and agent B scores 0.45, so the rule keeps expanding A.

- **Failure F1 — Metaproductivity–Performance Mismatch.** The right quantity for choosing a parent is how good its descendants become, not $\hat{p}(a)$. *Example:* A's children score 0.50, 0.52, 0.48 (best 0.52); B's children score 0.58, 0.60 (best 0.60). The greedy rule keeps feeding A.
- **Failure F2 — Noise.** With few tests, $\hat{p}$ is unreliable: a lucky 3/3 (1.00) outranks a solid 30/50 (0.60).
- **Failure F3 — Rigid budget use.** Every new child receives its batch of tests even after failing nine in a row, while older promising agents receive none.
- **Failure F4 — Sequential execution.** Modification and testing wait on each other, so hardware sits idle and runs take long.

### (e) Existing baselines

| Method | Which agent to modify | Which agent to test | What it fixes | What remains |
|---|---|---|---|---|
| SICA | The best scorer so far (greedy), with the full history of past attempts in its prompt | The newest agent, on the entire dataset | Gives a simple, working self-improvement loop; the history helps the LLM learn from past edits | F1–F4 all remain; the growing history overflows the LLM's context window (it stalled after 360 of 800 tests on SWE-Verified-60) |
| DGM | A random agent from the archive, with a weight that grows with its score and shrinks with how many children it already has | The newest agent, progressively: a small batch first, more tasks only if promising | Keeps every agent (an open-ended archive) and explores beyond the single best; partial early stopping (part of F3) | Still ranks by an agent's own score (F1), small noisy batches (F2), newest-child-first testing (F3), sequential (F4) |

### (f) The HGM pipeline

**One-sentence idea.** Rate each agent by how well its entire family of descendants performs, and use Bayesian sampling to decide, one test at a time, whether to create a new agent or test an existing one, and which.

**Analogy.** Progeny testing in animal breeding: a breeding bull is ranked by the milk yield of his daughters, not by his own appearance. HGM ranks an agent by its descendants' test results, not by its own score.

**Step 1 — The ideal target: clade-metaproductivity (CMP).** *Intuition:* "If I expand agent $a$ now and keep growing the tree until the budget runs out, how good will the best member of $a$'s family be?" Let $\mathcal{T}_B$ be the final tree produced by expanding $a$ now and then following $\pi$, and $\mathcal{C}_{\mathcal{T}_B}(a)$ be $a$'s clade in that final tree.

$$\mathrm{CMP}^{\pi}(\mathcal{T},a)=\mathbb{E}_{\mathcal{T}_B\sim\pi}\Big[\max_{a'\in\mathcal{C}_{\mathcal{T}_B}(a)}U(a')\Big]$$

*Example:* two equally likely futures in which the best family member has accuracy 0.60 or 0.50 → $\mathrm{CMP}=0.55$. (The paper also defines a *global* version, GMP, that takes the best agent in the whole final tree; GMP is the exact Q-value of "expand $a$" but is too broad to estimate. CMP narrows attention to $a$'s own subtree.)

**Step 2 — Why CMP is the right target (Theorem 1).** *Assumptions:* Assumption 1 (Setting 2). *Result:* access to the true CMP (an "oracle") suffices to implement the Gödel Machine. *Proof idea:* in the Gödel game, once an agent is chosen as parent, every later agent descends from it, and the final output (last parent or last child) also lies in its clade. Hence "best final agent in $a$'s clade" equals "final reward of choosing $a$ now," which is exactly the Q-value:

$$\mathrm{CMP}^{\pi}(\text{state},a)=Q^{\pi}(\text{state},a)$$

A Gödel Machine accepts a child only when it can prove that doing so yields higher expected utility, which is the same as comparing the CMP of child and parent (ties keep the parent). Choosing the larger Q-value is Bellman-optimal (P5). *Example:* $\mathrm{CMP}(\text{child})=0.58>\mathrm{CMP}(\text{parent})=0.55$ → accept; if both were 0.55 → keep the parent.

**Step 3 — Estimate CMP by pooling the clade.** *Intuition:* the true CMP is unknown, so pool every pass/fail recorded anywhere in $a$'s clade. This equals an average of the members' accuracies weighted by how often each was tested. Because HGM tests strong agents more (Step 5), strong descendants carry more weight, so the pooled rate behaves like a soft maximum over the family.

$$\widehat{\mathrm{CMP}}(a)=\frac{S(a)}{S(a)+F(a)}=\sum_{a'\in\mathcal{C}(a)}\frac{n(a')}{\sum_{a''\in\mathcal{C}(a)}n(a'')}\,\hat{p}(a'),\qquad n(a')=s(a')+f(a')$$

*Example:* $a$ is 3/10, its child is 6/10, and its grandchild is 16/20. Pooled: $(3+6+16)/(10+10+20)=25/40=0.625$, while $a$'s own score is only 0.30. The heavily tested grandchild supplies half the weight.

**Step 4 — Choose whom to expand: Thompson sampling on clade counts, sharpened over time.** *Intuition:* each agent draws a plausible "family quality" from its clade belief, and the highest draw gets to self-modify. A schedule $\tau$ grows as the budget runs out: early draws are spread out (explore), late draws hug the means (exploit). Multiplying both Beta parameters by $\tau$ keeps the mean but shrinks the spread.

$$\theta_a\sim\mathrm{Beta}\big(\tau(1+S(a)),\,\tau(1+F(a))\big),\qquad a^{\star}_{\text{expand}}=\arg\max_a\theta_a,\qquad \tau=\frac{B}{b}$$

*Example:* $S=3$, $F=7$. At the start ($\tau=1$): $\mathrm{Beta}(4,8)$, mean 0.333, standard deviation 0.131. Halfway ($b=B/2$, $\tau=2$): $\mathrm{Beta}(8,16)$, same mean, standard deviation 0.094.

**Step 5 — Choose whom to test: Thompson sampling on each agent's own counts.** *Intuition:* the same lottery, but on each agent's own record, among agents that still have untried tasks; the winner is tested on one new task. Strong agents get tested often, which sharpens their estimates and makes them dominate their ancestors' clade pools. Weak agents stop receiving tests after a few failures (early stopping).

$$\phi_a\sim\mathrm{Beta}\big(\tau(1+s(a)),\,\tau(1+f(a))\big),\qquad a^{\star}_{\text{eval}}=\arg\max_a\phi_a$$

After the test, add the pass or fail to $s$ or $f$ of $a^{\star}$, and to $S$ or $F$ of $a^{\star}$ and every one of its ancestors. *Example:* an agent at 0/9 has $\mathrm{Beta}(1,10)$ (mean 0.09); an agent at 5/10 has $\mathrm{Beta}(6,6)$ (mean 0.5). With $\tau=1$, the 0/9 agent wins the draw only about 0.9% of the time.

**Step 6 — Expand or test? (widening rule from UCB-AIR).** *Intuition:* UCB-AIR is an infinite-armed-bandit algorithm (Wang et al., 2008) that adds a new arm only when the number of pulls is large enough. HGM keeps the number of agents growing like a power of the number of tests, so the tree widens more slowly than evidence accumulates and each agent gets more tests over time.

$$\text{expand at step } t\iff N_t^{\alpha}\ge\lvert\mathcal{T}_t\rvert,\qquad \alpha=0.6$$

*Example:* after $N=100$ tests, $100^{0.6}\approx15.8$. A tree of 12 agents → expand; a tree of 16 → test. A run of 800 tests ends with about $800^{0.6}\approx55$ agents; a run of 8000 with about 220.

**Step 7 — Hand in the best-belief agent.** *Intuition:* return the agent whose *pessimistic* estimate is highest, so a lucky short streak cannot beat a long solid record. Let $q_{\epsilon}$ be the $\epsilon$-th percentile of a distribution (the value below which $\epsilon\%$ of it lies); the paper uses $\epsilon=1$, the 1st percentile.

$$a_{\text{final}}=\arg\max_{a}\;q_{\epsilon}\Big(\mathrm{Beta}\big(1+s(a),\,1+f(a)\big)\Big)$$

*Example:* agent X at 3/3 has $\mathrm{Beta}(4,1)$, whose 1st percentile is $0.01^{1/4}\approx0.32$. Agent Y at 30/50 has $\mathrm{Beta}(31,21)$ with mean 0.596 and standard deviation 0.067, so its 1st percentile is $\approx0.596-2.33\times0.067\approx0.44$. Y is handed in.

**Step 8 — Run asynchronously.** Several workers (20 in the released configuration) each run one iteration — an expansion or a test — and start the next iteration the moment they finish, using the latest counts. Jobs still running are counted when applying the Step 6 rule. The run opens with 5 expansions of the root agent executed in parallel, so early tests have several children to choose from. Easy tasks finish faster than hard ones, which makes early estimates slightly optimistic; the paper reports that this bias fades once an agent has about 50 tests.

**Pseudocode.**

```text
Input: root a0, test budget B, alpha = 0.6, epsilon = 1 (percentile)
T <- {a0};  s, f, S, F <- 0 for every agent;  N <- 0
while N < B:                                  # every worker runs this loop concurrently
    tau <- B / (B - N)                        # sharpening schedule, grows toward the end
    if N^alpha >= |T|  (counting running jobs):          # widen the tree
        for each expandable agent a in T:                # code: agents with >= 1 pass
            theta_a ~ Beta(tau*(1+S(a)), tau*(1+F(a)))   # clade belief
        a* <- argmax theta_a
        child <- a* rewrites a copy of its own code      # self-modification LLM
        add child to T under a*
    else:                                                # gather evidence
        for each agent a in T with untried tasks:
            phi_a ~ Beta(tau*(1+s(a)), tau*(1+f(a)))     # own belief
        a* <- argmax phi_a
        r <- run a* on one untried task                  # task-solving LLM, r in {0,1}
        update s(a*) or f(a*); update S or F for a* and all its ancestors
        N <- N + 1
return argmax over a of 1st-percentile( Beta(1+s(a), 1+f(a)) )
```

**How HGM fixes each failure.**

| Failure | HGM's fix |
|---|---|
| F1 Mismatch | Parents are ranked by the clade estimate of CMP (Steps 3–4), the target that reproduces the Gödel Machine under Assumption 1 (Step 2). |
| F2 Noise | Pooling over the clade gives each decision far more data; Beta beliefs and Thompson sampling account for uncertainty; the final pick is pessimistic (Step 7). |
| F3 Rigid budget | Testing is decoupled from creation and happens one agent–task pair at a time (Steps 5–6), so weak agents stop being tested early. |
| F4 Sequential | Workers run expansions and tests asynchronously (Step 8). |

## Results and Analysis

**Organization.** The experiments answer three questions in sequence. **Part A** checks the premise: does a guidance score predict how good an agent's descendants become? (Table 1; metrics M3–M5.) **Part B** checks the payoff: does better guidance yield better agents with less compute? (Table 2; metrics M1–M2.) **Part C** checks scale and generality: when the search is scaled to the full SWE-bench Verified, does the resulting agent rival human designs and transfer to new tasks and a stronger LLM? (Tables 3–4; metrics M1, M6.) **General setup:** every HGM run uses the asynchronous version with $\tau=B/b$, $\epsilon=1$, and $\alpha=0.6$; HGM, DGM, and SICA all start from the same initial agent, taken from DGM's code: a single LLM loop with a bash tool and a file-editing tool, with a one-hour limit per run of the agent. Budgets are counted in single agent–task tests. **How to read the tables:** in Table 1, columns are correlations in $[-1,1]$, and higher means the guidance score better tracks long-term self-improvement. In Table 2, the accuracy columns are better when higher (the "+" number is the gain over the initial agent), and the CPU-hours columns are better when lower. In Tables 3–4, columns are accuracy on SWE-bench Lite (filtered = 207 never-seen tasks; standard = all 300 tasks), higher is better. Task counts in parentheses are computed from the reported percentages.

### Part A — Does the guidance score predict long-term self-improvement?

**Question and setup.** After each method finishes its search on SWE-Verified-60 and on Polyglot, the authors take every agent in its tree and compute the target, empirical CMP (M3: the best accuracy among its descendants). They correlate it with the method's own guidance score (M4): an agent's own accuracy for SICA and DGM, and the leak-free clade estimate for HGM. Correlations are reported unweighted and weighted by the number of tests behind each prediction (M5). **How to read:** around 0.3 means the score barely tells you which agents have productive descendants; around 0.8 means it tells you well.

| Guidance score | SWE-Verified-60, weighted | SWE-Verified-60, unweighted | Polyglot, weighted | Polyglot, unweighted |
|---|---|---|---|---|
| SICA (own accuracy) | 0.444 | 0.444 | 0.274 | 0.274 |
| DGM (own accuracy) | 0.285 | 0.406 | 0.383 | 0.357 |
| **HGM (clade estimate)** | **0.778** | **0.512** | **0.626** | **0.873** |

**Takeaway 1 — An agent's own score is a weak guide; the clade estimate is a strong one.**

- Own-accuracy scores correlate only 0.27–0.44 with descendants' best result, which is the empirical face of the Metaproductivity–Performance Mismatch (failure F1).
- HGM's clade estimate reaches 0.78 (SWE-Verified-60) and 0.63 (Polyglot) weighted, and 0.51 and 0.87 unweighted — the highest in every column.
- The weighted gain is largest on SWE-Verified-60 (0.778 vs. at most 0.444), showing that well-supported clade estimates are the most trustworthy ones.

### Part B — Does better guidance give better agents with less compute?

**Question and setup.** Each method runs a budget of 800 tests. On SWE-Verified-60, GPT-5 writes the self-modifications and GPT-5-mini solves the tasks; on Polyglot, a quantized Qwen3-Coder-480B-A35B-Instruct writes the self-modifications and Qwen3-Coder-30B-A3B-Instruct solves the tasks. Agents see no hidden tests or test results on Polyglot. Every reported Polyglot percentage equals a whole number of tasks out of 59 (e.g., 30.5% = 18/59, 20.3% = 12/59), which matches the released code's Polyglot subset rather than the full benchmark; this task count is inferred, not stated in the paper. The initial agent scores 40.0% (24/60) on SWE-Verified-60 and 20.3% on Polyglot. HGM hands in its best-belief agent; DGM and SICA hand in their agent with the highest observed accuracy. The total API cost for all three methods was about 5,000 USD. **How to read:** look for the method that is simultaneously highest in accuracy (M1) and lowest in CPU-hours (M2).

| Method | SWE-Verified-60 accuracy (%) | CPU-hours | Polyglot accuracy (%) | CPU-hours |
|---|---|---|---|---|
| Initial agent | 40.0 | — | 20.3 | — |
| SICA | 50.0 (+10.0), 30/60 | did not finish (stalled after 360 tests) | 25.4 (+5.1), ≈15/59 | 572 |
| DGM | 53.3 (+13.3), 32/60 | 1231 | 27.1 (+6.8), ≈16/59 | 2385 |
| **HGM** | **56.7 (+16.7), 34/60** | **517** | **30.5 (+10.2), ≈18/59** | **347** |

**Takeaway 2 — HGM finds the strongest agent and uses the least compute.**

- Accuracy: HGM's gain over the initial agent is +16.7 points on SWE-Verified-60 (DGM +13.3, SICA +10.0) and +10.2 on Polyglot (DGM +6.8, SICA +5.1).
- Compute: HGM needs 2.38× fewer CPU-hours than DGM on SWE-Verified-60, and 6.86× fewer than DGM and 1.65× fewer than SICA on Polyglot. SICA's run on SWE-Verified-60 stopped improving after 360 tests (45% of its budget) because its prompts exceeded the LLM's context window.
- Figure 1 (right) plots the best accuracy found against CPU-hours on SWE-Verified-60: HGM's curve reaches 56.67% at ≈517 CPU-hours, while the DGM curve plateaus and runs to ≈1231 CPU-hours. The figure labels the two baseline values the other way around from Table 2 (DGM 50%, SICA 53.33%); with either labeling, HGM is highest.

### Part C — Does the approach scale, rival human designs, and transfer?

**Question and setup.** The search is scaled to the full SWE-bench Verified (500 tasks) with 8000 tests, and GPT-5-mini solves the tasks. The initial agent is made stronger and less biased: it keeps only the bash tool (no file editor) and gets five-hour limits for both self-modification and task runs, which yields 53.2% to start. The best-belief agent is then tested on SWE-bench Lite (Setting 3) in two forms: with GPT-5-mini (Table 3) and with GPT-5 swapped in (Table 4). Comparisons are the initial ancestor; SWE-agent, the top human-designed agent with checked results, re-run by the authors with GPT-5-mini; and SWE-agent's leaderboard-leading submission with Claude 4 Sonnet. **How to read:** the filtered column (M6) is the clean test of generalization; the standard column is what the public leaderboard shows.

Result on the optimization benchmark itself: SWE-bench Verified accuracy rises from 53.2% to **61.4%**, the top GPT-5-mini system on the leaderboard and within the top 10 of all checked submissions, including systems on more expensive LLMs.

Table 3 — same LLM (GPT-5-mini), new tasks:

| Agent | SWE-Lite filtered, 207 tasks (%) | SWE-Lite standard, 300 tasks (%) |
|---|---|---|
| HGM initial ancestor | 34.8 (≈72) | 44.0 (132) |
| SWE-agent + GPT-5-mini | 39.6 (≈82) | 47.6 (≈143) |
| **HGM best-belief agent** | **40.1 (≈83)** | **49.0 (147)** |

Table 4 — stronger LLM (GPT-5) vs. the leaderboard leader:

| Agent | SWE-Lite filtered, 207 tasks (%) | SWE-Lite standard, 300 tasks (%) |
|---|---|---|
| SWE-agent, leaderboard #1 (Claude 4 Sonnet) | 48.3 (≈100) | 56.7 (170) |
| **HGM best-belief agent + GPT-5** | 47.8 (≈99) | **57.0 (171)** |

**Takeaway 3 — The evolved design generalizes across tasks and LLMs and reaches the level of the best human-designed agent.**

- New tasks: on the 207 never-seen tasks, the HGM agent beats its own ancestor by +5.3 points (40.1 vs. 34.8), so the search improved general coding ability rather than memorizing SWE-bench Verified. With the same GPT-5-mini, it also edges out SWE-agent (40.1 vs. 39.6; 49.0 vs. 47.6), which isolates the gain to agent design rather than the LLM. Its 49.0% standard score ranks second among checked SWE-bench Lite submissions.
- New LLM: with GPT-5 swapped in, the agent scores 57.0% on standard SWE-bench Lite, one task above the leaderboard leader (56.7%) and the best among checked results; on the filtered set it is one task behind (47.8 vs. 48.3). This is the paper's "human-level" claim.
- Other observations: inspecting the evolved code shows self-written additions such as automatic error diagnosis after failed tests and multi-attempt test-and-revise loops, and some agents performed several self-edits within a single expansion step.

## Limitations

The statistical evidence is thin relative to the strength of the claims. Every comparison is a single run per method with no repeated seeds or confidence intervals, even though LLM-driven self-modification is highly random. On SWE-Verified-60 one task is 1.7 points, so HGM's lead over DGM is 34 vs. 32 tasks (and on the roughly 59-task Polyglot set, 18 vs. 16), and the SWE-bench Lite comparisons that support "human-level" are decided by one to four tasks, with the GPT-5 comparison winning by one task on the standard set and losing by one on the filtered set. The correlation study scores each estimator on its own method's tree, so differences in tree shape and in how tests were allocated are mixed into the comparison of estimators. Figure 1 and Table 2 also assign the two baseline accuracies differently.

The theory justifies the target but not the estimator or the algorithm. Theorem 1 holds only under strong assumptions (repeatable trials, free proofs, unit-cost modifications, a game restricted to accepting or rejecting one child), and it is close to definitional: CMP is constructed to equal the Q-value in that game, so an oracle for it gives optimal decisions almost by construction. The practical algorithm departs from that game in several ways: it tests agents (an action the theorem removes), expands any agent in the tree, and runs asynchronously. Its estimator is a pooled mean rather than an expected maximum under the future policy, so its value depends on how tests happened to be allocated. No bound links the estimate to true CMP, so "approximates the Gödel Machine" is a design motivation rather than a guarantee.

The efficiency and generality claims mix algorithm with engineering and stay within one domain. Much of the CPU-hour saving comes from asynchronous parallel execution, which could be added to DGM and SICA as well, and the dominant real cost — LLM tokens and API spend per method — is not compared. The 61.4% result is measured on the same benchmark that guided the search and the final selection, so only the filtered SWE-bench Lite numbers are clean, and those tasks come from the same 12 Python repositories as SWE-bench Verified. All experiments concern coding agents improving their scaffolds on two benchmarks with two model families; the safety of agents that rewrite and execute their own code is left unstudied.

## How to Reproduce

**Step 1 — Get the code.** The official implementation is at [github.com/metauto-ai/HGM](https://github.com/metauto-ai/HGM) under the Apache-2.0 license and is built on the Darwin Gödel Machine codebase. The main entry point is `hgm.py` (with `tree.py` and `hgm_utils.py` for the tree search), agent code lives in `coding_agent.py` and `coding_agent_polyglot.py`, settings live in `config.yaml`, and a `best_agent/` folder holds the released discovered agent, which lets you reproduce the transfer results without rerunning the search.

**Step 2 — Build the environment.** Create a Python 3.10 conda environment and install `requirements.txt`; install Docker and confirm your user can run containers, because every task runs inside a container; export `OPENAI_API_KEY`. The provided `run.sh` activates an environment named `agent` while the README creates one named `hgm`, so align the two names. The repository executes untrusted, model-generated code, so run it on an isolated machine.

**Step 3 — Prepare the benchmarks.** Clone SWE-bench into `swe_bench/` at the commit pinned in the README and install it in editable mode; build the Polyglot data with `python -m polyglot.prepare_polyglot_dataset` (git must have a user name and email configured). By default `hgm.py` searches on `swe_bench/subsets/small.json` + `medium.json`, which is SWE-Verified-60 (DGM's first two evaluation stages), and on `polyglot/subsets/small.json` + `medium.json` for Polyglot; with `--full_eval` it loads all 500 tasks of `princeton-nlp/SWE-bench_Verified` from Hugging Face. The initial agent is first tested on every search task, and those tests do not count toward the budget. For the filtered SWE-bench Lite evaluation, remove the 93 instance IDs that also appear in SWE-bench Verified to obtain the 207-task set.

**Step 4 — Match the paper's settings.** The released `config.yaml` sets `alpha: 0.6`, `max_task_evals: 800`, `max_workers: 20`, one-hour timeouts, and GPT-5-mini for self-modification, task solving, and failure diagnosis; every option also has a command-line flag. The paper's sharpening schedule $\tau=B/b$ is the `cool_down` option with `beta: 1.0` (the code multiplies both Beta parameters by $B^{\beta}/(B-N)^{\beta}$); it ships switched off, so pass `--cool_down` to match the paper. For the SWE-Verified-60 comparison, pass `--self_improve_llm` with GPT-5. For Polyglot, pass `--polyglot` and serve Qwen3-Coder-480B-A35B-Instruct (int4/int8 mixed quantization made with Intel's AutoRound) and Qwen3-Coder-30B-A3B-Instruct behind an endpoint that `llm.py` can call. For the full-Verified run, pass `--full_eval --max_task_evals 8000`, set both timeouts to 18000 seconds, and point `--initial_agent_name` at a bash-only initial agent built as described in Appendix C.1.

**Step 5 — Run the search.** Launch `./run.sh`, which cleans up Docker and calls `python hgm.py` (add the flags from Step 4 to that call); results go to `output_hgm/<run_id>/`, where `hgm_metadata.jsonl` records every agent's parent and its pass/fail history. The search loop itself does not pick the final agent, so apply the best-belief rule (Step 7 of the pipeline: highest 1st percentile of $\mathrm{Beta}(1+s,1+f)$) to those records. Expect hundreds of allocated CPU-hours per 800-test run (517 for HGM on SWE-Verified-60) and API spending on the order of the paper's roughly 5,000 USD across all three methods.

**Step 6 — Evaluate and compare.** Score the handed-in agent with `evaluate_agent.py` on SWE-bench Verified and on SWE-bench Lite (filtered and standard), then swap the task-solving LLM to GPT-5 for the transfer test. For baselines, run DGM from [github.com/jennyzzt/dgm](https://github.com/jennyzzt/dgm), SICA from the code released with Robeyns et al. (2025), and SWE-agent using its Claude 4 Sonnet leaderboard configuration with GPT-5-mini substituted as the backbone.

**Parts the paper and repository leave unspecified.** Random seeds and run-to-run variance; how allocated CPU-hours were counted for each method and how many cores the baselines used; the exact task-ID lists for SWE-Verified-60 and filtered SWE-bench Lite; which LLM writes self-modifications in the full-Verified run and which LLM performs failure diagnosis in each experiment; the meaning of the option `n_pseudo_descendant_evals` (set to 10000 and passed into the clade-count computation), which the paper does not describe; a ready-made script for the best-belief selection; scripts for the correlation analysis of Part A (Appendix D gives the formulas); the search trees and logs of the reported runs; the Qwen serving setup; and the exact SWE-agent + GPT-5-mini configuration.