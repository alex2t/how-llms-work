# How LLMs Work

Plain-English notes on how large language models work, from the first token to the business case. Every idea comes with a small worked example you can check by hand.

## Who these notes are for

Readers with a technical background (for example a computing degree or a career in enterprise software) who are not machine-learning specialists. No prior knowledge of neural networks is assumed: the Introduction covers the foundations you need.

## How to read it

Read the chapters in order the first time; each one builds on the last. Use the navigation bar at the top to jump between chapters, and the **Previous / Next** buttons at the bottom of each page to move through the notes.

| Chapter | What you will learn |
| --- | --- |
| [0 · Introduction](00-introduction.md) | The maths and machine-learning basics you need, and how text modelling evolved from one-hot vectors and RNNs to Transformers |
| [1 · Transformers](01-transformers.md) | BPE tokenization, embeddings, positional encoding, where Q, K and V come from, attention, and the full Transformer block |
| [2 · Training](02-training.md) | How the model learns: pre-training data, batches and padding, the loss and backpropagation, fine-tuning, RLHF and DPO, LoRA and distillation |
| [3 · Inference](03-inference.md) | How the model generates text: the KV cache, greedy and beam search, temperature, top-k and top-p, and context limits |
| [4 · Quantization](04-quantization.md) | Memory, number formats, quantization, Mixture of Experts, and what changed in Transformers since 2017 |
| [5 · Evaluation](05-evaluation.md) | Perplexity, MSE, accuracy, precision, recall, F1, AUC-ROC, and business outcome metrics |
| [6 · Prompting](06-prompting.md) | Writing clear prompts, in-context learning, chain-of-thought, self-consistency, self-critique, prompt chaining, and how reasoning models are trained |
| [7 · RAG](07-rag.md) | Retrieval-augmented generation: chunking, embeddings, vector and keyword search, re-ranking, citations, evaluation, and when to use long context or fine-tuning instead |
| [8 · RAG security](08-rag-security.md) | Attacks on a RAG system and their defences (data poisoning, data leaks and access control, prompt injection, unauthorised tool calls), and how to build and run a RAG pipeline in production |
| [9 · Advanced RAG](09-advanced-rag.md) | Techniques beyond basic RAG: better chunks and index entries, late interaction (ColBERT), page images (ColPali), question decomposition, RAPTOR and GraphRAG, CRAG and Self-RAG, and how to choose and measure them |
| [10 · The future of RAG](10-future-of-rag.md) | The "RAG is dead" debate against the evidence: long context windows, agentic search, agentic RAG, and context engineering |
| [11 · Agents](11-agents.md) | Models that work in a loop with tools: tool calling, MCP, memory, guardrails, evaluating agents, and multi-agent systems |
| [12 · Deploying agents](12-deploying-agents.md) | Taking an agent from demo to production in a bank: choosing the use case, the agent charter, clean data, testing, staged rollout, tracing and drift, adoption, the business case and the month-three review |

## Running examples

The same examples appear throughout the notes so you never meet unexplained numbers:

- The sentence **`YOUR CAT IS A LOVELY CAT`**, with token IDs YOUR = 105, CAT = 6587, IS = 5475, A = 3578, LOVELY = 65.
- The original Transformer's settings: 512 numbers per token, 8 attention heads, 6 layers.
- The translation **"I love you very much" → "Ti amo molto"**.
- The next-token **logits** CAT 6.2, DOG 4.9, IS 2.1, LOVELY 1.0, which softmax turns into probabilities 0.772, 0.211, 0.013 and 0.004.
- A **fraud check**: an LLM labelling 100 transactions as fraud or not fraud, used to explain evaluation metrics. In Chapters 7 to 11 the same bank builds a fraud-policy assistant, secures it for production, and builds an agent that handles disputed card payments.
