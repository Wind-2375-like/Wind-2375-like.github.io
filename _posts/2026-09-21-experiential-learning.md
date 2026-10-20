---
layout: post
title: 'Reading Notes of Welcome to the Era of Experience'
date: 2026-09-21 23:00:00
description: 'TL;DR: Introducing experiential learning'
tags: nlp llm-agents
categories: ReadingNotes
thumbnail: assets/img/blogs/era-of-experience.png
toc:
  sidebar: left
related_posts: false
---

TL;DR: Introducing experiential learning

<!--more-->

> **Reference**: https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf

## Why Stream, not Episodic

What is a trajectory / episode?

$$
\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)
$$

What is a policy?

$$
a_{t} \sim \pi_{\theta}(\cdot\mid s_{t})
$$

What is an objective?

$$
\max_{\pi_{\theta}} \mathbb{E}[\sum_{t=1}^{T} r_t]
$$

|           | Episodic | Stream |
| --------- | -------- | ------ |
| Data      | Many independent trajectories, each reset to $$s_0 \sim \rho_0$$ at the start | One trajectory, $$t = 0, 1, 2, \dots$$, never reset |
| Learning  | Only during training; frozen after deployment | Online; can update at every step (weights or memory) |
| Objective | $$\mathbb{E}[\sum_{t=1}^{T} r_t]$$; the books close at $$T$$ | $$\lim_{T\to\infty}\frac{1}{T}\sum_{t=1}^{T} r_t$$; an action now answers for rewards months away |

## Autonomous vs. Human-Centric

|                      | Human-Centric | Autonomous |
| -------------------- | ------------- | ---------- |
| Action Space         | $$\mathcal{A} = \{\text{text sent to the user}\}$$ | $$\mathcal{A} = \{\text{keyboard/mouse events}, \text{API calls}, \text{code}, \text{motor commands}, \dots\}$$. Natural language loses its privilege; actions break down into more atomic instructions (keyboard/mouse events, etc.). Talking to humans is just one subset, and the policy picks human-friendly or machine-friendly actions instrumentally |
| Transition           | Entirely the human brain. Both throughput (waiting on slow human output) and the information ceiling (cannot exceed humans) are low | Depends on the action/tool used; each one selects a different transition |
| Resulting Trajectory | One human observation, one machine action, in turn | Fewer and fewer states involve a human; the agent grows more and more autonomous |

On autonomy: executing an environment step does not require a human action as input. Equivalently: the human is demoted from "the environment itself" to "a small part of the environment" (or merely the source of goals).

## Grounded Reward

What is a reward in the human-data era? A human's verdict on the action (or a reward model trained to mimic it):

$$
r_{t+1} = R_{\text{human}}(s_t, a_t)
$$

What is a grounded reward? A measurement of the consequences:

$$
r_{t+1} = R_{\phi}(g, s_{t+1})
$$

where $$g$$ is the user's goal and $$s_{t+1}$$ carries signals from the world itself: heart rate, exam results, CO$$_2$$ levels, cost, error rate, profit, ...

|          | Human Prejudgement | Grounded |
| -------- | ------------------ | -------- |
| Looks at | $$(s_t, a_t)$$: the action, before anything happens | $$s_{t+1}, s_{t+2}, \dots$$: what actually happened |
| Ceiling  | The rater. A strategy the rater underappreciates gets no credit, so it is never found | The world. Whatever moves the signal gets credit, including strategies no human would pick |
| Supply   | One label costs one human judgment | Free and abundant: a new measurement at every step of the stream |

"Grounded" still requires human involvement. The signal is steered toward what each user wants, e.g., "improve my fitness" → some function of heart rate, sleep duration, and steps; "help me learn Spanish" → exam results. Then spend human feedback sparingly, on correcting.

A bi-level optimization:

Outer loop, sparse, from the user ($$U$$ is feedback such as satisfaction):

$$
\max_{\phi}\; U(\pi^{\star}_{\phi})
$$

Inner loop, dense, from the world:

$$
\pi^{\star}_{\phi} = \arg\max_{\pi} \lim_{T\to\infty}\frac{1}{T}\sum_{t=1}^{T} R_{\phi}(g, s_t)
$$

$$\phi$$ defines the rubric to compute reward. For example, to choose heart rate and sleep duration as fitness measurement, with hyperparameters about how to combine the two metrics to form one reward. Then we optimize the model toward that reward. **Then we need to avoid Goodhart's law about reward hacking**, so human will inspect whether they really satisfy with the policy. If not, that means the rubric needs to be changed.

**But does it really exist a rubric that will be universally satisfiable by everyone?** Even for each person, **it is not very feasible to find a rubric -> train -> find another rubric ...** Maybe human does not know exactly what will make them "happy".

## Planning and Reasoning

What is reasoning? Computation spent before committing to an action. Thought tokens $$z$$ make the LLM a universal computer, free to run any algorithm before it answers:

$$
z_{1:k} \sim \pi_{\theta}(\cdot\mid s_t), \qquad a_t \sim \pi_{\theta}(\cdot\mid s_t, z_{1:k})
$$

What is planning? Looking ahead through a world model, which predicts what the world does next, reward included:

$$
\hat{p}_{\psi}(s_{t+1}, r_{t+1}\mid s_t, a_t)
$$

|                 | Human-Centric | Experiential |
| --------------- | ------------- | ------------ |
| Medium          | $$z$$ is human language, trained to imitate how humans think | $$z$$ is whatever computes best: symbolic, distributed, continuous, differentiable. Shaped by reward, not by resemblance to human thought |
| Checked against | Human data: does it read like an expert's reasoning, does it match the expert's answer | The world: hypothesize, experiment, observe, update |
| Look-ahead      | A model of text: what would a human say next? | A model of the world: what would happen next? Plan over the agent's own actions and their causal effects; every step of the stream corrects $$\psi$$ |

The paper's thought experiment: imitate the best human thinking of 5,000 years ago and you get animism; of 1,000 years ago, theism; of 300 years ago, Newton; of 50 years ago, quantum mechanics. Each jump needed experiments, not just more thinking inside the old frame.

Planning and reasoning are not exclusive: an LLM can sit inside the planner, proposing actions or simulating and evaluating their consequences.

On grounding thought: universality only covers internal computation. The agent can run any algorithm, but nothing checks the premises. Without feedback from the world, an agent, however sophisticated, only recycles what humans already know.

## Reinforcement Learning

The simulation era (Atari, AlphaZero, Dota) was superhuman but narrow: closed problems, each with one precisely defined reward. The human-data era is broad but capped at what humans know. The era of experience wants both. Its ingredients are the classic RL concepts, which the human-data era bypassed by putting a human where each one used to be:

|                      | Human stand-in | Why streams need it back |
| -------------------- | -------------- | ------------------------ |
| Reward               | The rater's prejudgement | Grounded $$R_{\phi}$$ (above) |
| Value function       | The rater is the value function: scores the action on the spot | No $$T$$: the books never close, so there is no final return to wait for. Bootstrap from the unfinished stream: $$\delta_t = r_{t+1} - \bar{r} + v(s_{t+1}) - v(s_t)$$, with $$\bar{r}$$ the running average reward |
| Exploration          | The human prior already lands near good behavior | Beating the prior means trying what no human would. Needs to be principled, yet practical enough for the real world |
| World model          | Reasoning in human language | Planning over real consequences needs $$\hat{p}_{\psi}$$ (above) |
| Temporal abstraction | Human language again: words are ready-made sub-goals, and episodes are short | Atomic actions × horizons of months. Assigning credit keystroke by keystroke does not scale; needs options that span longer and longer stretches of time |

On RL: the human-data era never refuted these concepts. It substituted a human for each one. Demote the human, and they all come due again.
