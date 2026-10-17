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

## Chain-of-thought

Some questions need several steps of reasoning, and a model that must give the answer as its very next token has no room to work them out. **Chain-of-thought (CoT)** prompting asks the model to write the intermediate steps first and the answer last. It matters because, for multi-step problems in maths, logic and planning, this simple change makes answers much more often correct.

### Why writing the steps helps

A model produces one token per forward pass, and each pass does a fixed amount of computation ([Chapter 3](03-inference.md#generation-step-by-step)). When the answer must come immediately, all the reasoning has to happen inside a single pass. When the model first writes the steps, every written step becomes part of the input for the next token: the model can read its own intermediate results, exactly as a person uses scratch paper. More output tokens means more computation spent on the problem.

### Worked example

The question (made-up numbers): *A customer made three purchases of €45, €120 and €80 within ten minutes. Their usual daily spend is €50. How many times their usual daily spend is this?*

| Prompt style | What the model writes | Result |
| --- | --- | --- |
| Direct answer | `About 3 times.` | Wrong: a guess from the look of the numbers |
| Chain-of-thought ("Think step by step, then give the answer.") | `Total: 45 + 120 + 80 = 245 euros. Ratio: 245 ÷ 50 = 4.9. Answer: 4.9 times.` | Right: 4.9 |

Two ways to get a chain of thought:

- **Few-shot CoT:** include solved examples that show their working, not just the answers (the method of Wei et al., 2022).
- **Zero-shot CoT:** simply add an instruction such as "Let's think step by step." (Kojima et al., 2022).

### Limits

- **Longer answers cost more.** Every reasoning token is a decode step: 200 extra tokens at 7 ms each add 1.4 seconds ([Chapter 3](03-inference.md#worked-example-a-7-billion-weight-model)).
- **Little gain on simple tasks.** Looking up a fact or labelling an obvious case needs no steps.
- **The written steps are not guaranteed to be the real reason.** A model can write plausible steps and still reach a wrong answer, or a right answer for the wrong reason, so the steps are useful but must not be trusted blindly.

## Self-consistency

A single chain of thought can take a wrong turn. **Self-consistency** asks the same question several times, with sampling switched on so the chains differ, and returns the answer that comes up most often. It trades extra computation for accuracy, and it needs no change to the model or the prompt.

### How it works

1. Sample several chains of thought for the same question, with a temperature above 0 (for example 0.7) so that they differ ([Chapter 3](03-inference.md#sampling-and-temperature)).
2. Extract the final answer from each chain.
3. Return the **majority answer**: the one that appears most often.

**Worked example:** the purchase question from the previous section, asked 5 times (made-up outputs):

| Chain | Reasoning (shortened) | Final answer |
| --- | --- | --- |
| 1 | 45 + 120 + 80 = 245; 245 ÷ 50 | 4.9 |
| 2 | Total 245; ratio 4.9 | 4.9 |
| 3 | 45 + 120 + 95 = 260; 260 ÷ 50 (copied a number wrongly) | 5.2 |
| 4 | 245 ÷ 50 | 4.9 |
| 5 | 245 ÷ 100 (used the wrong daily spend) | 2.45 |

4.9 appears 3 times out of 5, so self-consistency answers 4.9. Correct chains tend to agree with each other, while mistakes scatter in different directions.

### Why voting helps, in numbers

Suppose each chain is right with probability 0.6, independently of the others. The vote is certainly right when more than half of the chains are right:

| Chains sampled | Right chains needed for a majority | Probability the vote is right |
| --- | --- | --- |
| 1 | 1 | 0.600 |
| 5 | 3 | 0.683 |
| 11 | 6 | 0.753 |

(These are lower bounds: when the wrong answers all differ, a smaller group of matching right answers can still win the vote.)

The cost grows with every chain: 5 chains cost about 5 times the decode time and tokens of one. Self-consistency therefore suits questions with a short, checkable final answer, such as a number or a label, where the extra accuracy is worth the price. It does not help with open-ended writing, where no two answers are identical and there is nothing to vote on.
