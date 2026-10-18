# CLAUDE.md — How LLMs Work (MkDocs notes)

This file tells Claude Code how to work on this repository. Read it, `plan.md` and `README.md` at the start of every session.

## What this project is

Plain-English notes that explain how large language models work, written as an introduction and six chapters with worked numeric examples. It is built with **MkDocs + Material for MkDocs** and published to **GitHub Pages** (`https://alex2t.github.io/how-llms-work/`) by a GitHub Actions workflow on every push to `main`.

| # | Chapter | File | Status |
| --- | --- | --- | --- |
| — | Home page (about the notes, how to read it) | `docs/index.md` | Written |
| 0 | Introduction: foundations and history | `docs/00-introduction.md` | Written |
| 1 | How Transformers Work: BPE, Attention and Q/K/V | `docs/01-transformers.md` | Written |
| 2 | Training: how the model learns | `docs/02-training.md` | Written |
| 3 | Inference and decoding | `docs/03-inference.md` | Written |
| 4 | Quantization, efficiency and the modern Transformer | `docs/04-quantization.md` | Written |
| 5 | Evaluation: how we measure a model | `docs/05-evaluation.md` | Written |
| 6 | Prompting, reasoning and agents | `docs/06-prompting.md` | Written; agents to come with later lectures |

`plan.md` is the single source of truth for what is done and what is left.

## Repository layout

```
.
├── CLAUDE.md                  # this file: rules for Claude Code
├── plan.md                    # task list: what is done, what is left
├── fix.md                     # proposed changes to existing content, waiting for approval
├── AGENTS.md                  # instructions for Antigravity (Gemini), written by Antigravity; committed
├── README.md                  # public project page, written like a book's back cover
├── mkdocs.yml                 # site config, theme and navigation bar
├── requirements.txt           # Python packages for the build
├── .github/workflows/deploy.yml   # builds and publishes to GitHub Pages
├── sources/                   # CME295 lecture summaries (Gemini notebook); git-ignored, never commit
├── gemini/                    # chapter rewrites by Gemini (*_agy.md) and Claude's reviews (*_rw_claude.md); git-ignored, never commit
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

**Environment notes (this machine).** Norton Antivirus intercepts HTTPS, so tools that use their own certificate list fail with "unable to get local issuer certificate". Git is fixed for this repository only (`git config --local http.sslBackend schannel`); for `uv` downloads add `--system-certs`.

## CME295 lecture review

The notes are being checked against Stanford's [CME295: Transformers & Large Language Models](https://cme295.stanford.edu/syllabus/) (Afshine and Shervine Amidi). The course has 9 lectures; slides appear one a week. Summaries made with a Gemini notebook are in `sources/` (one file per lecture).

- **One lecture at a time.** For each lecture, compare every topic with the notes and write only the gaps (not the topics already covered) into `fix.md`, section "CME295 gap analysis": what is missing, whether it needs a new chapter or belongs in an existing one (give the chapter and section), and a priority. The user decides each item before anything is written.
- **Summaries are pointers, not facts.** Verify every claim and number against the original paper or official documentation before it goes into the notes.
- **Paraphrase and credit.** Do not copy slide text or images; credit CME295 in the chapter's Sources when its lecture shaped a section.
- **`sources/` is git-ignored** and must never be committed or published.
- **The glossary (Phase 6) waits until Lecture 9 has been reviewed.**
- **After the review,** the user reads each chapter and flags unclear parts; rewrite those with a definition plus a concrete example (see Writing style).

## Working rules (follow every time)

1. **Propose before changing existing content.** Before editing or reorganising any section that is already written, show the user what you plan to change (a short before/after or a bullet list) and wait for approval. New sections can be written directly.
2. **Never delete without asking.** If a change would remove a paragraph, table, example or section, say exactly what would be removed and why, and wait for an explicit yes. Moving text is also a change that needs approval.
3. **One section at a time.** Write a section, run `mkdocs build --strict`, then update `plan.md` and the status table (rule 4), then move to the next section.
4. **After each section is implemented, update:**
   - `plan.md`: tick the task `[x]` and add any follow-up task you discovered.
   - `CLAUDE.md`: the chapter status table above, if a chapter's status changed.

   `README.md` has no progress log. Update it only when a chapter is added or its scope changes (see "README" below).
5. **Check every number.** Any worked example must be computed, not guessed. Use a quick Python snippet to verify arithmetic (softmax, log, means, memory sizes) before writing it.
6. **Keep the navigation in sync.** Every new page must be added to `nav:` in `mkdocs.yml`; never leave a page out of the navigation bar.
7. **Commit messages:** `chN: <section> — <what changed>`, e.g. `ch2: padding mask — add 2-sentence batch example`.
8. **Never edit files in `gemini/`.** Chapter edits always go to `docs/`, even when the user has a `gemini/` file open or selects lines in it. When the selection is in a `gemini/` file, find the same passage in `docs/` and edit it there. The only file Claude may write in `gemini/` is its own review, `<chapter>_rw_claude.md`, when asked for one.
9. **Change only what was asked.** Edit only the passage the user pointed to. Any change anywhere else (another paragraph, bullet, table, cheat sheet, Sources, another file) needs the user's permission first.
10. **Check the notes before adding information.** When the user asks for more information, first search all chapters for it. If it is already explained somewhere, do not write anything: say where (as a clickable `file:line` link) and ask what to do (link to it, move it, or expand it). Only a passage in the text that explains the topic counts as an explanation; a name in Sources, a heading, a cheat-sheet row or a passing mention does not.
11. **Report every addition as a clickable link.** After adding or changing text, give the exact location as `docs/<file>.md:<line>` so the user can click it.
12. **Chapter review by the user.** The user reads one chapter at a time and asks for clarifications. Apply each requested clarification **directly, without proposing it first** (this overrides rule 1 during a review; rule 2 still applies: ask before removing a whole paragraph, table, example or section). Run `mkdocs build --strict` after each change. The user's questions only show where the text is unclear: never copy a question into the notes; instead rewrite the passage so that no reader would need to ask it, and answer the question in the chat. **Do not record review changes** in `fix.md`, `plan.md`, `README.md` or anywhere else (this overrides rule 4 during a review), and **do not commit during the review**. Commit, push and deploy once, at the end of that chapter's review. Open decisions stay at the top of `fix.md`, under "Waiting for approval". When applying a requested change:
   - **Add useful context, inside the passage.** Within the passage the user pointed to, also add any extra information the reader would need to understand it. Changes outside that passage follow rule 9 (ask first).
   - **Existing coverage first.** If the topic is already explained elsewhere in the notes, follow rule 10 (say where and ask) before writing.
   - **Do not apply an incorrect suggestion.** If the user's proposed change is factually wrong, do not make it; explain in the chat why it is incorrect, with evidence, and leave the text as it is.

## Writing style (match Chapter 1)

- **Call it "notes", never "book"** (in the site and the README): "book" sounds presumptuous.
- **Audience:** a technical reader (computing degree, enterprise software background) who is not an ML specialist. English may not be the reader's first language: short, direct sentences; no unfinished sentences.
- Open every section with one or two sentences saying what it explains and why it matters.
- **Every numbered section (`##`, for example 3.3) starts with a clear introduction** before its first subsection: what the section is about, the key idea in plain words, and a short list of its subsections (3.3.1, 3.3.2, …) linked by anchor, saying what each one covers. Model: the introduction of 3.3 "Generation step by step" in `docs/03-inference.md`. Subsections (`###`, for example 3.3.1) do not need this introduction.
- Define every term in plain words the first time it appears (e.g. "logit", "batch", "gradient", "vocabulary", "d_model"). Explain every letter in a formula.
- Every important idea gets a **small worked example with real arithmetic**. If numbers are made up, say so in one short line.
- **Reuse the running examples** so the reader never meets new mystery numbers:
  - Sentence `YOUR CAT IS A LOVELY CAT`, token IDs YOUR=105, CAT=6587, IS=5475, A=3578, LOVELY=65 (DOG=4012 used for comparison).
  - Base model: d_model = 512, 8 heads, d_k = 64, 6 layers, d_ff = 2,048.
  - Translation: "I love you very much" → "Ti amo molto" (with `<SOS>`, `<EOS>`).
  - Logits: CAT 6.2, DOG 4.9, IS 2.1, LOVELY 1.0 → probabilities 0.772, 0.211, 0.013, 0.004.
  - Fraud example (Chapter 5): an LLM labelling transactions as fraud / not fraud. Evaluation sample of 100 transactions with 10 fraud (TP 8, FN 2, FP 6, TN 84 at threshold 0.5); real traffic 100,000 a month, 2% fraud.
- Tables for step-by-step calculations and comparisons; numbered lists for procedures; bullet lists for properties.
- When the user comments that a part "is not clear": rewrite it with a definition plus a concrete example. Do not just shorten it.
- End each chapter with a **Cheat sheet** table and a **Sources** list.
- Link back to earlier chapters by page and heading instead of repeating them, e.g. `[causal mask](01-transformers.md#what-the-mask-looks-like)`.
- **Every cross-reference is an anchor link** to the exact heading that explains the topic (`page.md#heading`), never a bare link to the top of a page (`page.md`). Check that the anchor matches an existing heading.
- **A link never replaces an explanation.** The reader must understand the current passage without clicking any link. When a passage uses a term or idea explained in another chapter, explain it again here, briefly but completely (a definition, plus a small example when useful); the link is only there for a reader who wants to see where else the topic is discussed. Never write "see Chapter N" or "Chapter N explains it in full" instead of explaining.

## Markdown conventions for MkDocs

- Math: inline `$...$`, display blocks on their own lines:
  ```
  $$
  \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  $$
  ```
  Do **not** use fenced ```` ```latex ```` blocks: they render as code, not math.
- One `#` heading per page (the chapter title); sections are `##`, subsections `###`.
- **Sections and subsections are numbered** (for example 3.2 and 3.2.1 in Chapter 3) so the reader always sees where a passage sits in the chapter's structure. The numbers are added automatically on the site by `docs/javascripts/numbering.js` (chapter number taken from the file name); never type numbers into the headings, because that would change every anchor.
- Callouts with Material admonitions: `!!! note`, `!!! tip "Key idea"`, `!!! warning`.
- Diagrams: prefer Mermaid (```` ```mermaid ````) for flows; otherwise put an SVG in `docs/images/` and reference it with a relative path.
- Sources from the attached slides by Umar Jamil are CC BY-NC 4.0: paraphrase and credit them; do not copy slide images.

## Gemini rewrite and Claude review

After the user has reviewed a chapter (rule 12), Gemini (Antigravity) is asked to rewrite it, following its own instructions in `AGENTS.md` (committed, not git-ignored). Claude then compares the rewrite with the original and gives an opinion. The user decides what happens next.

**File names** (all in `gemini/`, which is git-ignored and must never be committed or published):

| Chapter | Original | Gemini's rewrite | Claude's review |
| --- | --- | --- | --- |
| Introduction | `docs/00-introduction.md` | `gemini/introduction_agy.md` | `gemini/introduction_rw_claude.md` |
| 1 | `docs/01-transformers.md` | `gemini/transformers_agy.md` | `gemini/transformers_rw_claude.md` |
| 2 | `docs/02-training.md` | `gemini/training_agy.md` | `gemini/training_rw_claude.md` |
| 3 | `docs/03-inference.md` | `gemini/inference_agy.md` | `gemini/inference_rw_claude.md` |
| 4 | `docs/04-quantization.md` | `gemini/quantization_agy.md` | `gemini/quantization_rw_claude.md` |
| 5 | `docs/05-evaluation.md` | `gemini/evaluation_agy.md` | `gemini/evaluation_rw_claude.md` |
| 6 | `docs/06-prompting.md` | `gemini/prompting_agy.md` | `gemini/prompting_rw_claude.md` |

**When asked to review `gemini/<chapter>_agy.md`,** read it and the original in full, then write `gemini/<chapter>_rw_claude.md` with:

1. **Verdict:** one of *replace the chapter*, *keep the original and merge some parts*, or *keep the original*, with the main reasons in two or three sentences.
2. **Section by section:** a table with each section, which version is better (original, Gemini or equal) and why.
3. **What Gemini did better:** the specific passages worth taking, with the target section in the original.
4. **Problems in the rewrite:** factual errors, wrong or unchecked numbers (recompute every number with a script, rule 5), content dropped from the original, broken links or anchors, and departures from the Writing style and Markdown conventions.
5. **Recommended next step:** the exact changes to make if the user agrees.

Do not change anything in `docs/` while reviewing. The user then decides: replace the chapter with Gemini's version (after fixing the problems found), merge the parts listed, or keep the original. Applying that decision follows rule 12 (one commit, push and deploy at the end).

## README

`README.md` is the public face of the repository. Write it like the back cover of a book, in a professional tone (not a social-media post), and keep it concise. Sections, in this order:

1. Title, a short description and the link to the live site.
2. **What the notes cover.**
3. **Who it is for.**
4. **Chapters** (number, title, what you will learn).
5. **How it was made** (course notes, Claude, NotebookLM, Claude Code).
6. **Credits.**
7. **Build locally.**

No progress log, no status columns, no internal task lists and no one-time setup steps.
