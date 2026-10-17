# How LLMs Work

A plain-English book on how large language models work, from the first token to the business case. Every idea comes with a small worked example you can check by hand.

## Who this book is for

Readers with a technical background (for example a computing degree or a career in enterprise software) who are not machine-learning specialists. No prior knowledge of neural networks is assumed: the Introduction covers the foundations you need.

## How to read it

Read the chapters in order the first time; each one builds on the last. Use the navigation bar at the top to jump between chapters, and the **Previous / Next** buttons at the bottom of each page to move through the book.

| Chapter | What you will learn |
| --- | --- |
| [0 · Introduction](00-introduction.md) | The maths and machine-learning basics you need, and how text modelling evolved from one-hot vectors and RNNs to Transformers |
| [1 · Transformers](01-transformers.md) | BPE tokenization, embeddings, positional encoding, where Q, K and V come from, attention, and the full Transformer block |
| [2 · Training](02-training.md) | How the model learns: batches, padding, the shifted-right decoder input, the loss and backpropagation |
| [3 · Inference](03-inference.md) | How the model generates text: the KV cache, greedy and beam search, temperature, top-k and top-p |
| [4 · Quantization](04-quantization.md) | Memory, number formats, quantization, and what changed in Transformers since 2017 |
| [5 · Evaluation](05-evaluation.md) | Perplexity, MSE, accuracy, precision, recall, F1, AUC-ROC, and business outcome metrics |
| [6 · Prompting](06-prompting.md) | In-context learning, chain-of-thought, self-consistency, and how reasoning models are trained |

## Running examples

The same examples appear throughout the book so you never meet unexplained numbers:

- The sentence **`YOUR CAT IS A LOVELY CAT`**, with token IDs YOUR = 105, CAT = 6587, IS = 5475, A = 3578, LOVELY = 65.
- The original Transformer's settings: 512 numbers per token, 8 attention heads, 6 layers.
- The translation **"I love you very much" → "Ti amo molto"**.
- The next-token **logits** CAT 6.2, DOG 4.9, IS 2.1, LOVELY 1.0, which softmax turns into probabilities 0.772, 0.211, 0.013 and 0.004.
- A **fraud check**: an LLM labelling 100 transactions as fraud or not fraud, used to explain evaluation metrics.
