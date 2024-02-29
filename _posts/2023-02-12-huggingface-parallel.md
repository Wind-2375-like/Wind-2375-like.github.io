---
layout: post
title: Huggingface parallel training for solving the CUDA out of memory issue
date: 2023-02-12 23:37:32
description: Document a workable solution for the annoying CUDA Out Of Memory (OOM).
tags: NLP CUDA
categories: TechEssays
thumbnail: assets/img/blogs/oom.jpg
images:
  compare: true
  slider: true
---

## Background

I am running conditional text generation experiments these days on various seq2seq models. The script went well on other models except for the T5-large model. I continuously got the CUDA OOM error.

The T5-large model is not so big and I'm running code on a server with 4 NVIDIA Corporation GP102s. Each one has 12G of video memory. It seems that I could run my code. However, even I set the batch size to 1 for each device, I got the CUDA OOM error.

## Environment

My huggingface transformer version is `4.20.1` and my code looks like [this](https://github.com/huggingface/notebooks/blob/main/examples/summarization.ipynb) (`preprocess_function`, `dataset.map`, `trainer`).

## Some tries

I followed the [advice](https://github.com/huggingface/transformers/issues/9311) and added `--fp16` and `--sharded_ddp` but neither of them work.

## Solution

[Parallelize](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Model.parallelize)

`( device_map = None )`

Parameters

- **device_map** (`Dict[int, list]`, optional, defaults to None) — A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always automatically mapped to the first device (for esoteric reasons). That means that the first device should have fewer attention modules mapped to it than other devices. For reference, the t5 models have the following number of attention modules:
  - t5-small: 6
  - t5-base: 12
  - t5-large: 24
  - t5-3b: 24
  - t5-11b: 24

Uses a device map to distribute attention modules of the model across several devices. If no device map is given, it will evenly distribute blocks across all devices.

Example:

```python
# Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
model = T5ForConditionalGeneration.from_pretrained("t5-3b")
device_map = {
    0: [0, 1, 2],
    1: [3, 4, 5, 6, 7, 8, 9],
    2: [10, 11, 12, 13, 14, 15, 16],
    3: [17, 18, 19, 20, 21, 22, 23],
}
model.parallelize(device_map)
```

I just copy the code and replace "t5-3b" with "t5-large" and it works!
