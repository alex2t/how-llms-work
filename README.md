# How LLMs Work

A plain-English book on how large language models work, with worked numeric examples in every chapter. Built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), published on GitHub Pages:

**📖 Read it: https://alex2t.github.io/how-llms-work/**

## Chapters

| # | Chapter | Status |
| --- | --- | --- |
| 0 | Introduction: foundations and history | Written (draft) |
| 1 | How Transformers Work: BPE, Attention and Q/K/V | Written |
| 2 | Training: how the model learns | Written (draft) |
| 3 | Inference and decoding | Written (draft) |
| 4 | Quantization, efficiency and the modern Transformer | Written (draft) |
| 5 | Evaluation: how we measure a model | Written (draft) |
| 6 | Prompting, reasoning and agents | Planned |

The detailed task list is in [`plan.md`](plan.md). Proposed changes to written chapters, including topics found missing when comparing the book with Stanford's CME295 course (one lecture at a time), are in [`fix.md`](fix.md).

## Build locally

```bash
pip install -r requirements.txt
mkdocs serve          # preview at http://127.0.0.1:8000
mkdocs build --strict # check before committing
```

Every push to `main` publishes the site through GitHub Actions (`.github/workflows/deploy.yml`). One-time setup: repository Settings → Pages → Source = **GitHub Actions**.

## Progress

One line per implemented section, newest at the bottom.

- 2026-10-17 — Setup — Repository scaffold: MkDocs Material site with a chapter navigation bar, GitHub Pages workflow, home page and chapter placeholders.
- 2026-10-17 — Setup — GitHub Pages: site published; workflow actions updated to Node 24 versions.
- 2026-10-17 — Chapter 1 — Import cleanup: formulas now render as math, date line removed, encoder–decoder diagram added.
- 2026-10-17 — Chapter 2 — Formulas: 8 formulas now render as math.
- 2026-10-17 — Chapter 2 — Status and number check: chapter recorded as written, numbers checked, proposed fixes listed in fix.md.
- 2026-10-17 — Chapter 2 — Fixes: three numbers corrected, chapter references turned into links, home page lists all running examples.
- 2026-10-17 — Chapter 3 — Inference vs training: what is switched off at inference, with a memory example.
- 2026-10-17 — Chapter 3 — Generation step by step: the translation generated one token per pass, for encoder–decoder and chat models.
- 2026-10-17 — Chapter 3 — Prefill vs decode: why the first token is slower, with a 7B-model timing example.
- 2026-10-17 — Chapter 3 — The KV cache: what it stores, why it is safe, and how much memory it needs.
- 2026-10-17 — Chapter 3 — Greedy decoding and beam search: a beam of 2 worked through on the CAT/DOG example.
- 2026-10-17 — Chapter 3 — Sampling and temperature: the CAT/DOG logits at T = 0.5, 1 and 2.
- 2026-10-17 — Chapter 3 — Top-k sampling: keep the k most likely tokens and renormalise, with its main weakness.
- 2026-10-17 — Chapter 3 — Top-p sampling: the nucleus grows and shrinks with the model's confidence.
- 2026-10-17 — Chapter 3 — How the rules combine: temperature, top-k and top-p applied in sequence, plus typical settings.
- 2026-10-17 — Chapter 3 — Stopping: end token, max tokens, stop sequences and the context window.
- 2026-10-17 — Chapter 3 — Cheat sheet and sources: chapter complete.
- 2026-10-17 — Chapter 1 — Training and generation: shortened to a summary linking to Chapters 2 and 3.
- 2026-10-17 — Chapter 4 — Why size matters: memory and decode speed of a 7B model in each number format.
- 2026-10-17 — Chapter 4 — Number formats: range and precision of FP32, FP16, BF16, INT8 and INT4.
- 2026-10-17 — Chapter 4 — How quantization works: scale, zero point, rounding error, outliers and group-wise scales.
- 2026-10-17 — Chapter 4 — Kinds of quantization: when, what and how, including GPTQ, AWQ and GGUF.
- 2026-10-17 — Chapter 4 — Quality vs speed: the trade-off by bit count and how to check a quantized model.
- 2026-10-17 — Chapter 4 — What changed since 2017: the modern Transformer compared with the original, with a worked example for each change.
- 2026-10-17 — Chapter 4 — Cheat sheet and sources: chapter complete.
- 2026-10-17 — Chapter 3 — Links to Chapter 4 now open the exact section.
- 2026-10-17 — Chapter 5 — Three levels of evaluation: training loss, model metrics and business outcomes, on the fraud example.
- 2026-10-17 — Chapter 5 — Loss vs metric: why training uses cross-entropy but people judge with metrics.
- 2026-10-17 — Chapter 5 — Perplexity: what it means and why it needs the same tokenizer.
- 2026-10-17 — Chapter 5 — MSE: a regression loss, and why it does not fit next-token prediction.
- 2026-10-17 — Chapter 5 — Classification metrics: confusion matrix, accuracy, precision, recall, F1, ROC and AUC on the fraud sample.
- 2026-10-17 — Chapter 5 — Evaluating generative LLMs: benchmarks, human evaluation and LLM-as-judge.
- 2026-10-17 — Chapter 5 — Business outcomes: KPIs, cost and ROI, customer satisfaction, baselines and the threshold trade-off.
- 2026-10-17 — Chapter 5 — Choosing metrics for a project: an 8-step checklist, including drift monitoring.
- 2026-10-17 — Chapter 5 — Cheat sheet and sources: chapter complete.
- 2026-10-17 — Chapters 2–4 — Links to Chapter 5 now open the exact section.
- 2026-10-17 — Introduction — What you need to know first: vectors, dot product, matrix multiplication, transpose, neurons, loss and embeddings.
- 2026-10-17 — Introduction — Historical evolution: from one model per task to one model for all tasks.
- 2026-10-17 — Introduction — Vector embeddings: one-hot vectors, Word2Vec and their limits.
- 2026-10-17 — Introduction — RNNs and LSTMs: how they read text in order, and the three problems attention removed.
- 2026-10-17 — Introduction — Cheat sheet and sources: chapter complete.
- 2026-10-17 — CME295 review — Lecture 1: compared with the book; 5 missing topics proposed in fix.md.
- 2026-10-17 — CME295 review — Lecture 2: 7 missing topics proposed in fix.md, including Mixture of Experts and a new prompting and reasoning chapter.
- 2026-10-17 — CME295 review — Lecture 3: 9 missing topics proposed in fix.md, mostly for Chapter 2, plus reasoning models for Chapter 6.
- 2026-10-17 — Chapter 2 — Where pre-training data comes from, data mixtures and model collapse (CME295 L3-1, L3-2).
- 2026-10-17 — Chapter 2 — Packing: the alternative to padding, with the document mask (CME295 L3-4).
- 2026-10-17 — Chapter 2 — Masked language modelling: how encoder-only models such as BERT are trained (CME295 L2-4).
- 2026-10-17 — Chapter 2 — Preference training in detail: reward model, KL penalty, DPO loss and their costs (CME295 L3-6).
- 2026-10-17 — Chapter 2 — Cheap fine-tuning with LoRA and QLoRA; instruction tuning named (CME295 L3-5).
- 2026-10-17 — Chapter 2 — Distillation: a small model learns from a large one; linked from Chapter 4 (CME295 L3-8).
- 2026-10-17 — Chapter 4 — Stretching the context of a trained model with RoPE scaling (CME295 L2-6).
- 2026-10-17 — Chapter 4 — Mixture of Experts: more weights, same compute (CME295 L2-1).
- 2026-10-17 — Chapter 3 — What a chat looks like as tokens: the Llama 3 chat template (CME295 L1-3).
- 2026-10-17 — Chapter 3 — Constrained decoding: forcing a valid format, on the fraud label (CME295 L2-3).
- 2026-10-17 — Chapter 3 — How big a context window is, and why a bigger window is not a perfect memory (CME295 L2-7, L2-5).
- 2026-10-17 — Chapter 5 — Bits per byte: comparing models with different tokenizers (CME295 L3-3).

## Credits

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017).
- Umar Jamil, [Transformer from scratch](https://github.com/hkproj/transformer-from-scratch-notes) slides (CC BY-NC 4.0), used as a reference for explanations.
- Stanford University, [CME295: Transformers & Large Language Models](https://cme295.stanford.edu/syllabus/), by Afshine Amidi and Shervine Amidi (lectures on YouTube, slides on the course site). Used to check which topics the book should cover; explanations are paraphrased, and no slide content is copied.
- Google Gemini notebook summaries of the CME295 lectures, used as a topic checklist for that review. They are kept locally and not published.
