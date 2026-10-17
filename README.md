# How LLMs Work

A plain-English book on how large language models work, with worked numeric examples in every chapter. Built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), published on GitHub Pages:

**📖 Read it: https://alex2t.github.io/how-llms-work/**

## Chapters

| # | Chapter | Status |
| --- | --- | --- |
| 0 | Introduction: foundations and history | Planned |
| 1 | How Transformers Work: BPE, Attention and Q/K/V | Written, being imported |
| 2 | Training: how the model learns | Planned |
| 3 | Inference and decoding | Planned |
| 4 | Quantization, efficiency and the modern Transformer | Planned |
| 5 | Evaluation: how we measure a model | Planned |

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

## Credits

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017).
- Umar Jamil, [Transformer from scratch](https://github.com/hkproj/transformer-from-scratch-notes) slides (CC BY-NC 4.0), used as a reference for explanations.
