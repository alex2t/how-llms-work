# How LLMs Work

Plain-English notes on how large language models work, from the way a model reads text to the way it is evaluated in production. Every idea is explained in plain words and backed by a small worked example with real numbers that you can check by hand.

**Read it online: https://alex2t.github.io/how-llms-work/**

## What the notes cover

The notes follow one model's life from start to finish:

- **Foundations:** the vectors, dot products and matrices that everything is built on, and how text modelling evolved from one-hot vectors and recurrent networks to the Transformer.
- **The Transformer:** tokenization, embeddings, positional encoding, where queries, keys and values come from, and how attention works.
- **Training:** pre-training data, the loss, backpropagation, fine-tuning, preference training (RLHF and DPO), LoRA and distillation.
- **Inference:** how text is generated token by token, the KV cache, and the decoding rules (greedy, beam search, temperature, top-k, top-p).
- **Efficiency:** number formats, quantization, Mixture of Experts, and what has changed in Transformers since 2017.
- **Evaluation:** perplexity, classification metrics, benchmarks, LLM-as-judge, and how model metrics connect to business outcomes.
- **Prompting and reasoning:** in-context learning, chain-of-thought, and how reasoning models are trained.

A few running examples, such as the sentence `YOUR CAT IS A LOVELY CAT` and a fraud-detection use case, are reused across chapters, so the reader never meets unexplained numbers. Every worked calculation has been checked with a script.

## Who it is for

Readers with a technical background, for example a computing degree or a career in enterprise software, who want to understand what happens inside ChatGPT, Claude or Gemini. No machine-learning or advanced maths background is needed: the Introduction covers the foundations, and every term is defined the first time it appears.

## Chapters

| # | Chapter | What you will learn |
| --- | --- | --- |
| 0 | Introduction: foundations and history | The maths you need, and the path from one-hot vectors and RNNs to Transformers |
| 1 | How Transformers Work | BPE tokenization, embeddings, positional encoding, Q/K/V and attention |
| 2 | Training: how the model learns | Data, loss, backpropagation, fine-tuning, RLHF, DPO, LoRA and distillation |
| 3 | Inference and decoding | The generation loop, KV cache, decoding rules and context limits |
| 4 | Quantization, efficiency and the modern Transformer | Number formats, quantization, Mixture of Experts and post-2017 changes |
| 5 | Evaluation: how we measure a model | Perplexity, precision, recall, F1, AUC, benchmarks and business metrics |
| 6 | Prompting, reasoning and agents | In-context learning, chain-of-thought, self-consistency and reasoning models |

The notes are a work in progress: a section on agents and retrieval will be added to Chapter 6.

## How it was made

The notes grew out of several AI and LLM engineering courses. I took excerpts from the courses and wrote the text with AI assistance, each tool used for a different job:

- **Claude** explained the excerpts I gave it, or went further into a topic when I asked.
- **NotebookLM (Google Gemini)** summarised lecture videos, used to check which topics were missing.
- **Claude Code** wrote and edited the pages. I read every chapter and pointed out the passages that were unclear or badly phrased, and Claude Code rewrote them until they were easy to follow. It also checked the arithmetic and built the site.
- **Antigravity CLI (Gemini 4)** then rewrote each chapter to improve the wording and added diagrams and short refreshers on the maths.
- **Claude Code** reviewed each Gemini rewrite against the original: it rechecked every number, tested the diagrams and links, fixed the errors it found, and replaced the chapter with the improved version.

## Credits

- Stanford University, [CME295: Transformers & Large Language Models](https://cme295.stanford.edu/syllabus/), by Afshine Amidi and Shervine Amidi (lectures on YouTube, slides on the course site). Used to check which topics the notes should cover; explanations are paraphrased, and no slide content is copied.
- Google NotebookLM (Gemini) summaries of the CME295 lectures, used as a topic checklist. They are kept locally and not published.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017).
- Umar Jamil, [Transformer from scratch](https://github.com/hkproj/transformer-from-scratch-notes) slides (CC BY-NC 4.0), used as a reference for explanations.
- Written with the help of [Claude](https://claude.ai) and [Claude Code](https://claude.com/claude-code) (Anthropic), and Antigravity CLI (Google Gemini).
- Built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

Each chapter ends with its own list of sources.

## Build locally

```bash
pip install -r requirements.txt
mkdocs serve          # preview at http://127.0.0.1:8000
mkdocs build --strict # check before committing
```

Every push to `main` publishes the site to GitHub Pages through GitHub Actions.
