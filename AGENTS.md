# AGENTS.md — Antigravity (AGY) Instructions for *How LLMs Work*

This file defines Antigravity's role, rules, and workflow for this repository. Antigravity loads `AGENTS.md` automatically at the start of every session.

---

## 1. Project Overview

**How LLMs Work** is a set of plain-English notes explaining how large language models work, built with **MkDocs + Material for MkDocs** and published at `https://alex2t.github.io/how-llms-work/`.

### Chapter Map

| `<CHAPTER>` | Original File (`<FILE>`) | AGY Rewrite Output | Claude Review File | Rewrite Status |
| --- | --- | --- | --- | --- |
| `introduction` | `docs/00-introduction.md` | `gemini/introduction_agy.md` | `gemini/introduction_rw_claude.md` | Completed |
| `transformers` | `docs/01-transformers.md` | `gemini/transformers_agy.md` | `gemini/transformers_rw_claude.md` | Completed |
| `training` | `docs/02-training.md` | `gemini/training_agy.md` | `gemini/training_rw_claude.md` | Pending |
| `inference` | `docs/03-inference.md` | `gemini/inference_agy.md` | `gemini/inference_rw_claude.md` | Completed |
| `quantization` | `docs/04-quantization.md` | `gemini/quantization_agy.md` | `gemini/quantization_rw_claude.md` | Completed |
| `evaluation` | `docs/05-evaluation.md` | `gemini/evaluation_agy.md` | `gemini/evaluation_rw_claude.md` | Pending |
| `prompting` | `docs/06-prompting.md` | `gemini/prompting_agy.md` | `gemini/prompting_rw_claude.md` | Pending |

---

## 2. Primary Mission: Chapter Rewrites

> [!CAUTION]
> **NEVER touch `docs/`:** Antigravity (Gemini) must **NEVER** edit, create, or delete any file inside the `docs/` folder. Always write rewrites exclusively to `gemini/<CHAPTER>_agy.md`.

When asked to rewrite a chapter `<CHAPTER>`:

1. **Read for consistency first:** Read `docs/index.md`, the target chapter `<FILE>`, and adjacent chapters in `docs/` so terminology, cross-links, and running examples stay 100% consistent.
2. **Write only to `gemini/<CHAPTER>_agy.md`:** Write the **complete** rewritten chapter to `gemini/<CHAPTER>_agy.md`. Do not touch `<FILE>` or any other file in `docs/`.
3. **Update the status table:** When a rewrite is completed, update the `Rewrite Status` column in this `AGENTS.md` file.

---

## 3. What Makes a Rewrite Better

The goal of each rewrite is to make the chapter **substantially clearer, better explained, and more visual** while keeping all original content intact:

1. **Accessible to every degree-holder (even if their degree was a long time ago):**
   - The target reader has a technical/computing degree or enterprise software background, but is **not** a machine-learning specialist, and may have studied university maths many years ago.
   - English may not be their first language.
   - Whenever a concept relies on foundational maths or computer science (e.g., dot products, matrix shapes, exponents, logarithms, $e^x$, derivatives/chain rule, probability distributions, variance/standard deviation, floating-point bits), include a **short, friendly refresher in plain words** right where it is used so the reader never has to stop and look it up elsewhere.
   - Every reader should understand every paragraph on the first reading.

2. **More visual Explanations:**
   - Add **Mermaid diagrams** (`flowchart LR`, `flowchart TD`, `sequenceDiagram`, `stateDiagram-v2`) to illustrate architectures, pipelines, step-by-step data flows, and training/inference loops. Do not repeat in a diagram the exact numbers already shown in an adjacent table; use diagrams to show structure and flow.
   - Use **structured tables** to break down multi-step calculations, side-by-side comparisons, and tensor shape transformations so the reader can trace how inputs become outputs.
   - Use **callouts** (`!!! note`, `!!! tip "Key idea"`, `!!! warning`) to highlight intuition, quick refreshers, and common pitfalls.

3. **Zero content loss:**
   - Every topic, paragraph, worked example, table, diagram, cheat-sheet row, and source from the original `<FILE>` must remain present in the rewrite unless it is factually or mathematically wrong.
   - Never drop existing sentences or replace a paragraph with a table without keeping the paragraph's text.
   - If anything is removed, moved, or renamed, document it explicitly in the top `<!-- CHANGES ... -->` block.

---

## 4. Audience, Style & Pedantic Rules

- **Call it "notes", never "book"**: Never refer to the project as a "book".
- **Sentence style**: Use short, direct, complete sentences. Avoid joining long clauses with em dashes (`—`). No unfinished fragments, no unexplained jargon, and no emoji.
- **Formatting consistency with other chapters**:
  - Do **not** wrap plain numbers or dimensions in LaTeX (`512`, `0.81`, `6 × 512`, not `$512$`, `$0.81$`, `$6 \times 512$`).
  - Do **not** add horizontal rules (`---`) between sections.
  - Match the original chapter's use of code font vs plain text for words and tokens.
- **Section openings**: Open every `##` section with one or two sentences stating what the section explains and why it matters.
- **Term & formula definitions**: Define every technical term in plain words the first time it appears. Explain **every single letter and symbol** immediately after every formula.
- **Worked examples & verified arithmetic**:
  - Give every important idea a small worked example with real arithmetic. If the numbers are made up for illustration, state that in one short line.
  - **Compute every number with a Python command/script — never guess.** Verify that every result in every table and formula follows from its inputs.

---

## 5. Running Examples (Reuse — Do Not Invent New Mystery Numbers)

Always reuse the established running examples across chapters (and check the exact vectors in `docs/01-transformers.md` before using them):

1. **Sentence:** `YOUR CAT IS A LOVELY CAT`
   - Token IDs: `YOUR = 105`, `CAT = 6587`, `IS = 5475`, `A = 3578`, `LOVELY = 65` (and `DOG = 4012` for comparison).
   - 3-number toy embeddings: `LOVELY = [0.45, 0.62, −0.11]`, `YOUR = [0.12, −0.40, 0.33]`, `A = [−0.05, 0.08, 0.91]`, `DOG = [0.77, 0.15, −0.48]`, `IS = [−0.30, 0.66, 0.05]`, `CAT = [0.81, 0.10, −0.52]`.
2. **Base model (2017 Transformer):**
   - $d_{\text{model}} = 512$, $8$ attention heads, $d_k = d_v = 64$, $6$ layers, $d_{\text{ff}} = 2{,}048$.
3. **Translation example:**
   - `"I love you very much"` $\rightarrow$ `"Ti amo molto"`, with `<SOS>` and `<EOS>`.
4. **Next-token logits & probabilities:**
   - Logits: `CAT = 6.2`, `DOG = 4.9`, `IS = 2.1`, `LOVELY = 1.0` $\rightarrow$ Softmax probabilities: `0.772`, `0.211`, `0.013`, `0.004`.
5. **Fraud detection example:**
   - An LLM labels transactions as `FRAUD` or `OK`.
   - Evaluation sample: 100 transactions with 10 real frauds ($\text{TP} = 8$, $\text{FN} = 2$, $\text{FP} = 6$, $\text{TN} = 84$ at threshold $0.5$).
   - Real production traffic: $100{,}000$ transactions/month with $2\%$ fraud ($2{,}000$ frauds, $98{,}000$ legitimate).

---

## 6. Markdown & MkDocs Conventions

- **Headings & Anchors:**
  - Exactly one `#` heading per file (the chapter title); sections use `##`, subsections use `###`.
  - **Keep every existing heading text verbatim** unless it is genuinely unclear, because other chapters link to headings via anchor (e.g. `01-transformers.md#what-the-mask-looks-like`). If you change any heading, mark it with `RENAMED` in the top `CHANGES` comment.
- **Cross-chapter links:**
  - Preserve all links to other chapters in the format `[text](02-training.md#heading-anchor)`.
- **Math formatting:**
  - Inline math: `$...$`
  - Display math: always on its own lines between `$$` lines (never `$$ ... $$` on a single line, and never unindented inside a numbered list, which breaks MkDocs HTML lists):
    ```markdown
    $$
    \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$
    ```
  - **Never** use ```` ```latex ```` code blocks.
- **Callouts:**
  - Use Material for MkDocs admonitions: `!!! note`, `!!! tip "Key idea"`, `!!! warning` (with 4-space indented body).
- **Diagrams:**
  - Use Mermaid fenced code blocks (```` ```mermaid ````). Quote node labels that contain parentheses, brackets, or special characters (e.g., `A[" Token (ID 105)"]`).
- **End of chapter:**
  - Always end with `## Cheat sheet` (summary table) and `## Sources` (bulleted list).

---

## 7. Required Output Format for `gemini/<CHAPTER>_agy.md`

At the very top of `gemini/<CHAPTER>_agy.md`, before the `#` title, include an HTML comment listing all changes made in the rewrite (what changed, in which section, and why, including added refreshers and visual diagrams). Mark anything removed, moved, or renamed with **REMOVED**, **MOVED**, or **RENAMED**:

```html
<!--
CHANGES
- Section "...": Added plain-English refresher on ... and a Mermaid diagram showing ...
- Section "...": Clarified step-by-step arithmetic table for ...
- RENAMED: Heading "..." to "..." (reason)
- REMOVED: ... (reason)
-->
```
