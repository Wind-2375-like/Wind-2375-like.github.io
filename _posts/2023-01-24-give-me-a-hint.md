---
layout: post
title: Could you give me a hint? Generating inference graphs for defeasible reasoning
date: 2023-01-24 00:09:26
description: A reading note about a paper related to defeasible reasoning.
tags: NLP CausalReasoning
categories: ReadingNotes
thumbnail: https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost@main/img/ef38ea8e65d046de980563edf515c91a7ab0a7e07b5ce76705bf9fae61410310.png
images:
  compare: true
  slider: true
---

**Keywords:** Inference Graphs, Transfer Learning, Defeasible Reasoning, Human Aiding.

<img alt="Fig 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost@main/img/5e400103c58e8b20417827784022bd36321554b8932953bc79f3d4cd060d2dee.png" width="100%" data-zoomable/>

Link: [2105.05418.pdf (arxiv.org)](https://arxiv.org/pdf/2105.05418.pdf)

## Contribution

- Inference/Influence Graph for representing defeasible reasoning
- Transfer learning for generating influence graph
- Improvement of human performance with the aid of influence graph

## Method

### Inference Graph and Influence Graph

<img alt="Fig 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost@main/img/aaac32bcb2ab9b58ed0f595dfab10c0f6efec1727fae8cbcdff2ed41aaaf87fe.png" width="100%" data-zoomable/>

A typical causal inference is `P -> H`, where `P` is the premise and `H` is the hypothesis, i.e. the outcome of the premise. The arrow shows the causal relationship. Given new evidence, the hypothesis may be strengthened `P|U -+-> H` or weakened `P|U- -x-> H`. When it is weakened, this kind of reasoning is called defeasible reasoning. We can treat `U` as "assumptions" and `U-` as "defeaters".

Then the paper adopted an inference graph (left). Added contextualizers (something related to premises) and mediators (deduction step from `U` to `H`). I think it is the dataset WIQA's contribution (the two works share the same author) instead of the paper. But anyway, the paper switched its context from inference graph to influence graph (right), which is directly from WIQA.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost@main/img/f36f9ba147a9e2566ea2f9705f2e7b4f06bcabc54772063a67ad538feb065a9b.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost@main/img/24ebec2264b5bf120a465ebce58e4fb71b9800c8f8961b36c594459648f31619.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Green arrow: positive effect. Red arrow: negative effect.
</div>

### Transfer Learning

Transfer learning refers to the task when you want to let your model perform a task on one dataset, but it doesn't have any label. So you find another similar dataset with labels and adjust it to the same format as your intended task/dataset. Then you train your model on this similar dataset and then do the prediction on your target dataset.

Since the domain in the test set has changed, a good performance in the test set will indicate not only good accuracy but also excellent generalizability.

The paper aims to generate an influence graph given `P`, `H`, and `U`. It uses seq2seq architecture. The input of the encoder is easy, which is:

<img alt="Fig 5" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost@main/img/0b16ab260bf0ce6c2a55d66a46a9991ccd11eee92253b84a7d73dcb479432bf8.png" width="100%" data-zoomable/>

The output of the decoder is the sequential representation of the influential graph. See the appendix for details.

<img alt="Fig 6" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost@main/img/8a03da0d2851efa544057266c1382199a4b9d3c0e046a6447098927ce7843a07.png" width="50%" data-zoomable/>

The authors train the models (GPT2, T5) on WIQA and generate these graphs on SNLI, SOCIAL, and ATOMIC.

They did a train-test split to WIQA and evaluated their model.

<img alt="Fig 7" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost@main/img/90d230c07462707fc530967d3b99a94104cea66b26a27d02879ad817c01eaa5c.png" width="50%" data-zoomable/>

### Human Improvement

Then they observed if the graph helps people. They sampled a challenging subset.

<img alt="Fig 8" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost@main/img/91c77c1c460f40ad46021f486a00c1e3b526bd43551bceda704ae511489b4a67.png" width="50%" data-zoomable"/>

There is some improvement.
