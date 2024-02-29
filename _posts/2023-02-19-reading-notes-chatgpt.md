---
layout: post
title: Reading Notes of How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources
date: 2023-02-19 12:50:37
description: Reading notes of Yao's notes of "How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources".
tags: NLP LLM
categories: ReadingNotes
thumbnail: https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/imgimage-20230218225348976.png
images:
  compare: true
  slider: true
---

Reading notes of Yao's notes of [How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources (notion.site)](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1).

**Keywords:** code and instruction tuning; trade of between in-context learning and instruction tuning; unlock or inject the abilities; instruction-tuning, supervised instruction-tuning, and RLHF instruction-tuning; knowledge and reasoning.

## Reference

> Fu, Yao; Peng, Hao and Khot, Tushar. (Dec 2022). How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources. Yao Fu’s Notion. https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1

## Road Map

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/imgimage-20230218225348976.png" alt="image-20230218225348976"  />

Note: The base model for code-davinci-002 is highly likely not be the initial GPT-3 davinci model.

## Summary

The authors' conclusion:

- The **language generation ability** + **basic world knowledge** + **in-context learning** are from pretraining (`davinci`)
- The ability to **store a large amount of ==knowledge==** is from the 175B scale.
- The ability to **follow instructions** and **generalizing to new tasks** are from scaling instruction tuning (`davinci-instruct-beta`)
- The ability to perform **complex ==reasoning==** is likely to be from training on code (`code-davinci-002`)
- The ability to **generate neutral, objective, safe, and informative answers** are from alignment with human. Specifically:
    - If supervised tuning, the resulting model is `text-davinci-002`
    - If RLHF, the resulting model is `text-davinci-003`
    - Either supervised or RLHF, the models cannot outperform `code-davinci-002` on many tasks, which is called the *alignment tax*. (RLHF on 003 just recovers the in-context learning ability.)
- The **dialog ability** is also from RLHF (`ChatGPT`), specifically it tradeoffs in-context learning for:
    - Modeling dialog history
    - Increased informativeness
    - Rejecting questions outside the model’s knowledge scope

| Ability                                                      | OpenAI Model                                                 | Training Method                                    | OpenAI API            | OpenAI Paper                                                 | Open Source Approximate                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------- | --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| GPT-3 Series                                                 |                                                              |                                                    |                       |                                                              |                                                              |
| Generation  <br>+ World Knowledge  <br>+ In-context Learning | GPT-3 Initial<br><br>**Many abilities already within this model, although superficially weak** | Language Modeling                                  | Davinci               | [GPT 3 Paper](https://arxiv.org/abs/2005.14165)              | [Meta OPT](https://arxiv.org/abs/2205.01068)<br>[BLOOM](https://arxiv.org/abs/2211.05100)<br>[Galactica](https://arxiv.org/abs/2211.09085) |
| + Follow Human Instruction<br>+ Generalize to unseen task (free lunch of scaling instructions) | Instruct-GPT initial                                         | Instruction Tuning                                 | Davinci-Instruct-Beta | [Instruct-GPT paper](https://arxiv.org/abs/2203.02155)       | [T0 paper](https://arxiv.org/abs/2110.08207)<br>[Google FLAN paper](https://arxiv.org/abs/2109.01652)<br>[Google FLAN paper](https://arxiv.org/abs/2210.11416) |
| + Code Understanding<br>+ Code Generation                    | Codex initial                                                | Training on Code                                   | Code-Cushman-001      | [Codex Paper](https://arxiv.org/abs/2107.03374)              | [Salesforce CodeGen](https://arxiv.org/abs/2203.13474)       |
| GPT-3.5 Series                                               |                                                              |                                                    |                       |                                                              |                                                              |
| ++ Code Understanding<br>++ Code Generation<br>++ Complex Reasoning / Chain of Thought (*why*?)<br>+ Long-term dependency  (*probably*) | Current Codex<br><br>**Strongest model in GPT3.5 Series**    | Training on text + code <br>Tuning on instructions | Code-Davinci-002      | [Codex Paper](https://arxiv.org/abs/2107.03374)              | ??                                                           |
| ++ Follow Human Instruction<br>- In-context learning<br>- Reasoning<br>++ Zero-shot generation | Instruct-GPT supervised<br><br>**Trade in-context learning for zero-shot generation** | Supervised instruction tuning                      | Text-Davinci-002      | [Instruct-GPT paper](https://arxiv.org/abs/2203.02155), supervised part | [T0 paper](https://arxiv.org/abs/2110.08207)<br/>[Google FLAN paper](https://arxiv.org/abs/2109.01652)<br/>[Google FLAN paper](https://arxiv.org/abs/2210.11416) |
| + Follow human value<br>+ More detailed generation<br>+ In-context learning<br>+ Zero-shot generation | Instruct-GPT RLHF<br><br>**More aligned than 002, less performance loss** | Instruction tuning w. RLHF                         | Text-Davinci-003      | Instruct-GPT paper, RLHF part [Summarization .w human feedback](https://arxiv.org/abs/2009.01325) | [DeepMind Sparrow paper](https://arxiv.org/abs/2209.14375)<br>[AI2 RL4LMs](https://arxiv.org/abs/2210.01241) |
| ++ Follow human value<br>++ More detailed generation<br>++ Reject questions beyond its knowledge (why?)<br>++ Model dialog context<br>-- In-context learning | ChatGPT<br><br>**Trade in-context learning for dialog history modeling** | Tuning on dialog w. RLHF                           | -                     | -                                                            | [DeepMind Sparrow paper](https://arxiv.org/abs/2209.14375)<br/>[AI2 RL4LMs](https://arxiv.org/abs/2210.01241) |

## Limitations

1. **On-the-fly overwriting the model’s belief**: when the model expresses its belief in something, it might be hard to correct it when the belief is wrong. There seems to be a hierarchy of how strong the belief is, which means that there exists a list of very strong core belief. ==**It is extremely important to ensure such core belief should be absolutely 100% aligned with human.**==
2. **Formal reasoning**: the GPT-3.5 series cannot do reasoning within formal, strict systems like math or first-order logic.
    1. The word “reasoning” is less well-defined. Yet if we view there is a spectrum of ambiguity like (a) very ambiguous, no reasoning; (b) mixture of logic and ambiguous statements; (c). no ambiguity has to be very rigorous, then,
    2. The model can do very well on type (b) reasoning with ambiguity.
    3. The model cannot do type (c) reasoning, **yet whether such reasoning should be done by a language model or a symbolic system is up for discussion.**
    4. Update: [Wolfram Alpha as the Way to Bring Computational Knowledge Superpowers to ChatGPT—Stephen Wolfram Writings](https://writings.stephenwolfram.com/2023/01/wolframalpha-as-the-way-to-bring-computational-knowledge-superpowers-to-chatgpt/)
3. **Retrieval from the Internet**: the GPT-3.5 series cannot directly search the internet (for now)
    1. Likely tested within OpenAI: [WebGPT: Improving the Factual Accuracy of Language Models through Web Browsing (openai.com)](https://openai.com/blog/webgpt/)
    2. **Separate the knowledge and reasoning**: it would be ideal if we could offload the knowledge part to the outside retrieval system and let the language model only focus on reasoning.
    3. **Combining LLMs (reasoning) and search (knowledge) is a good direction**: LLMs are good for reasoning, not for knowledge. The knowledge within LLMs are unreliable and cannot be verified. On the other hand, the knowledge from search engine is orders or magnitude larger than LLM’s internal knowledge, can one can easily verify credibility of search results by checking the source.
        1. Retrieval-augmented models?
        2. Reduce the model size? (175B storing unreliable knowledge -> search engines)

## Questions to be answered

1. How does the GPT-3.5 acquire the reasoning ability (CoT)? (Likely because of code training according to the authors).
2. Which kinds of abilities are unlocked/injected by what means[^1]? The authors' hypothesis (according to scale):
    1. Abilities from code training: injection
    2. Abilities from instruction tuning: unlock

## What I need to further know about

- instruction-tuning? supervised instruction-tuning? RLHF instruction-tuning?
- code training?

[^1]: "by what means" is hypothesized in the [summary](##summary) section but some of them are not proved.

