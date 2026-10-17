# CLAUDE.md — How LLMs Work (MkDocs book)

This file tells Claude Code how to work on this repository. Read it, `plan.md` and `README.md` at the start of every session.

## What this project is

A plain-English book that explains how large language models work, written as five chapters with worked numeric examples. It is built with **MkDocs + Material for MkDocs** and published to **GitHub Pages** (`https://alex2t.github.io/how-llms-work/`) by a GitHub Actions workflow on every push to `main`.

| # | Chapter | File | Status |
| --- | --- | --- | --- |
| — | Home page (about the book, how to read it) | `docs/index.md` | Written (draft) |
| 0 | Introduction: foundations and history | `docs/00-introduction.md` | Placeholder |
| 1 | How Transformers Work: BPE, Attention and Q/K/V | `docs/01-transformers.md` | Written in Claude Docs; must be imported (see `plan.md`, Phase 0) |
| 2 | Training: how the model learns | `docs/02-training.md` | Placeholder |
| 3 | Inference and decoding | `docs/03-inference.md` | Placeholder |
| 4 | Quantization, efficiency and the modern Transformer | `docs/04-quantization.md` | Placeholder |
| 5 | Evaluation: how we measure a model | `docs/05-evaluation.md` | Placeholder |

`plan.md` is the single source of truth for what is done and what is left.

## Repository layout

```
.
├── CLAUDE.md                  # this file: rules for Claude Code + progress log
├── plan.md                    # task list: what is done, what is left
├── README.md                  # public project page + progress summary
├── mkdocs.yml                 # site config, theme and navigation bar
├── requirements.txt           # Python packages for the build
├── .github/workflows/deploy.yml   # builds and publishes to GitHub Pages
└── docs/
    ├── index.md               # Home page
    ├── 00-introduction.md     # Foundations and history
    ├── 01-transformers.md … 05-evaluation.md
    ├── images/                # diagrams (SVG/PNG) used by chapters
    └── javascripts/mathjax.js # LaTeX rendering
```

## Commands

```bash
pip install -r requirements.txt   # once
mkdocs serve                      # live preview at http://127.0.0.1:8000
mkdocs build --strict             # must pass with no warnings before every commit
```

Publishing is automatic: push to `main` and the workflow deploys to GitHub Pages. One-time setup: in the GitHub repo, Settings → Pages → Source = **GitHub Actions**.

## Working rules (follow every time)

1. **Propose before changing existing content.** Before editing or reorganising any section that is already written, show the user what you plan to change (a short before/after or a bullet list) and wait for approval. New sections can be written directly.
2. **Never delete without asking.** If a change would remove a paragraph, table, example or section, say exactly what would be removed and why, and wait for an explicit yes. Moving text is also a change that needs approval.
3. **One section at a time.** Write a section, run `mkdocs build --strict`, then update the logs (rule 4), then move to the next section.
4. **After each section is implemented, update three files:**
   - `plan.md`: tick the task `[x]` and add any follow-up task you discovered.
   - `README.md`: add one line to "Progress" (date, chapter, section, one-sentence summary) and update the chapter status table.
   - `CLAUDE.md`: add one line to the "Progress log" below and update the chapter status table above.
5. **Check every number.** Any worked example must be computed, not guessed. Use a quick Python snippet to verify arithmetic (softmax, log, means, memory sizes) before writing it.
6. **Keep the navigation in sync.** Every new page must be added to `nav:` in `mkdocs.yml`; never leave a page out of the navigation bar.
7. **Commit messages:** `chN: <section> — <what changed>`, e.g. `ch2: padding mask — add 2-sentence batch example`.

## Writing style (match Chapter 1)

- **Audience:** a technical reader (computing degree, enterprise software background) who is not an ML specialist. English may not be the reader's first language: short, direct sentences; no unfinished sentences.
- Open every section with one or two sentences saying what it explains and why it matters.
- Define every term in plain words the first time it appears (e.g. "logit", "batch", "gradient", "vocabulary", "d_model"). Explain every letter in a formula.
- Every important idea gets a **small worked example with real arithmetic**. If numbers are made up, say so in one short line.
- **Reuse the running examples** so the reader never meets new mystery numbers:
  - Sentence `YOUR CAT IS A LOVELY CAT`, token IDs YOUR=105, CAT=6587, IS=5475, A=3578, LOVELY=65 (DOG=4012 used for comparison).
  - Base model: d_model = 512, 8 heads, d_k = 64, 6 layers, d_ff = 2,048.
  - Translation: "I love you very much" → "Ti amo molto" (with `<SOS>`, `<EOS>`).
  - Logits: CAT 6.2, DOG 4.9, IS 2.1, LOVELY 1.0 → probabilities 0.772, 0.211, 0.013, 0.004.
  - Fraud example (Chapter 5): an LLM labelling 100 transactions as fraud / not fraud.
- Tables for step-by-step calculations and comparisons; numbered lists for procedures; bullet lists for properties.
- When the user comments that a part "is not clear": rewrite it with a definition plus a concrete example. Do not just shorten it.
- End each chapter with a **Cheat sheet** table and a **Sources** list.
- Link back to earlier chapters by page and heading instead of repeating them, e.g. `[causal mask](01-transformers.md#what-the-mask-looks-like)`.

## Markdown conventions for MkDocs

- Math: inline `$...$`, display blocks on their own lines:
  ```
  $$
  \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  $$
  ```
  Do **not** use fenced ```` ```latex ```` blocks: they render as code, not math.
- One `#` heading per page (the chapter title); sections are `##`, subsections `###`.
- Callouts with Material admonitions: `!!! note`, `!!! tip "Key idea"`, `!!! warning`.
- Diagrams: prefer Mermaid (```` ```mermaid ````) for flows; otherwise put an SVG in `docs/images/` and reference it with a relative path.
- Sources from the attached slides by Umar Jamil are CC BY-NC 4.0: paraphrase and credit them; do not copy slide images.

## Progress log

Add one line per implemented section, newest at the bottom: `YYYY-MM-DD — Chapter N — Section — one-sentence summary`.

- 2026-10-17 — Setup — Repository scaffold — MkDocs Material site, navigation tabs, GitHub Pages workflow, home page and placeholders for the Introduction and Chapters 1–5 created.
- 2026-10-17 — Setup — GitHub Pages — site published at https://alex2t.github.io/how-llms-work/; workflow actions updated to Node 24 versions.
