# Prompting, Reasoning and Agents

## The big picture

The earlier chapters explained how a model is built, trained and run. This chapter is about **using** a trained model well: how the text you give it steers what it does, and how models can be made to "think" before they answer. None of the techniques in the first sections change a single weight; they work entirely through the prompt and the decoding settings of [Chapter 3](03-inference.md). The last section shows how newer **reasoning models** are trained to think on their own.

1. **In-context learning:** teaching a task inside the prompt, with instructions or a few examples.
2. **Chain-of-thought:** asking for the steps before the answer.
3. **Self-consistency:** asking several times and taking the majority answer.
4. **Reasoning models:** models trained with reinforcement learning to write long reasoning before answering.

!!! note "Work in progress"
    Agents (models that use tools and act over many steps) and retrieval will be added to this chapter as the matching lectures of the CME295 course are reviewed.

## In-context learning

A pre-trained model has seen so much text that it can often do a new task just from how the prompt is written, without any fine-tuning. This is called **in-context learning**: the model picks up the task from the **context** (the prompt) at inference time, and its weights stay exactly as they were. It matters because it is the cheapest way to adapt a model: changing a prompt takes seconds, while fine-tuning ([Chapter 2](02-training.md#stage-2-supervised-fine-tuning-sft)) takes data, compute and a new copy of the weights.

### Zero-shot and few-shot prompts

- **Zero-shot:** the prompt only describes the task. No solved examples are given.
- **Few-shot:** the prompt also contains a few solved examples (often 1 to 10), so the model can copy their format and logic. GPT-3 (2020) showed that large models do this surprisingly well.

**Worked example: the fraud prompt** from [Chapter 5](05-evaluation.md#the-big-picture).

**Zero-shot:**

```text
Label the transaction as FRAUD or OK.
Transaction: €2,400 at an electronics shop in another country, 03:12, card usually used for groceries.
Label:
```

**Few-shot** (two solved examples first, made up for illustration):

```text
Label each transaction as FRAUD or OK.
Transaction: €38 at the usual supermarket, 18:05.
Label: OK
Transaction: €1,950 in gift cards, online, new device, 02:40.
Label: FRAUD
Transaction: €2,400 at an electronics shop in another country, 03:12, card usually used for groceries.
Label:
```

Both prompts end with `Label:`, so the most likely next token is the label itself. The examples in the few-shot prompt show the model the exact answer format and what kind of evidence points to fraud.

### Why it works, and what it costs

The model is still doing nothing but next-token prediction ([Chapter 1](01-transformers.md#from-the-last-vector-back-to-a-word)). During pre-training it saw countless texts where a pattern is set up and then continued, such as lists of questions and answers, so continuing the pattern in the prompt is exactly what it learned to do. Attention lets the last position look back at the examples while it predicts the label.

Examples are not free: every example adds tokens that must be processed at every request.

| | Zero-shot | Few-shot with 3 examples of about 60 tokens |
| --- | --- | --- |
| Extra tokens per request | 0 | 3 × 60 = 180 |
| Extra tokens per month at 100,000 transactions | 0 | 18,000,000 |
| Effect | Shortest prompt, fastest prefill | Usually more accurate and better-formatted answers; longer prefill ([Chapter 3](03-inference.md#prefill-vs-decode-why-the-first-token-is-slower)) |

Whether the extra cost is worth it is decided as in Chapter 5: measure both prompts on the same labelled sample, and compare the metrics and the costs.

**Practical tips:**

- Make the examples look like the real inputs, and include both labels, so the model does not learn that the answer is always the same.
- Keep the format identical in every example; the model copies the format very faithfully.
- For an exact output format, combine the prompt with [constrained decoding](03-inference.md#constrained-decoding).
