# How LLMs Work

A plain-English book on how large language models work, with worked numeric examples in every chapter. Built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), published on GitHub Pages:

**📖 Read it: https://alex2t.github.io/how-llms-work/**

## Chapters

| # | Chapter | Status |
| --- | --- | --- |
| 0 | Introduction: foundations and history | Planned |
| 1 | How Transformers Work: BPE, Attention and Q/K/V | Written |
| 2 | Training: how the model learns | Written (draft) |
| 3 | Inference and decoding | Written (draft) |
| 4 | Quantization, efficiency and the modern Transformer | Written (draft) |
| 5 | Evaluation: how we measure a model | In progress |

The detailed task list is in [`plan.md`](plan.md).

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

## Credits

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017).
- Umar Jamil, [Transformer from scratch](https://github.com/hkproj/transformer-from-scratch-notes) slides (CC BY-NC 4.0), used as a reference for explanations.
