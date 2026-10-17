# CLAUDE.md — How LLMs Work (MkDocs book)

This file tells Claude Code how to work on this repository. Read it, `plan.md` and `README.md` at the start of every session.

## What this project is

A plain-English book that explains how large language models work, written as five chapters with worked numeric examples. It is built with **MkDocs + Material for MkDocs** and published to **GitHub Pages** (`https://alex2t.github.io/how-llms-work/`) by a GitHub Actions workflow on every push to `main`.

| # | Chapter | File | Status |
| --- | --- | --- | --- |
| — | Home page (about the book, how to read it) | `docs/index.md` | Written (draft) |
| 0 | Introduction: foundations and history | `docs/00-introduction.md` | Written (draft) |
| 1 | How Transformers Work: BPE, Attention and Q/K/V | `docs/01-transformers.md` | Written, imported and live |
| 2 | Training: how the model learns | `docs/02-training.md` | Written (draft); fixes listed in `fix.md` |
| 3 | Inference and decoding | `docs/03-inference.md` | Written (draft) |
| 4 | Quantization, efficiency and the modern Transformer | `docs/04-quantization.md` | Written (draft) |
| 5 | Evaluation: how we measure a model | `docs/05-evaluation.md` | Written (draft) |
| 6 | Prompting, reasoning and agents | `docs/06-prompting.md` | Planned (approved in `fix.md`, L2-2) |

`plan.md` is the single source of truth for what is done and what is left.

## Repository layout

```
.
├── CLAUDE.md                  # this file: rules for Claude Code + progress log
├── plan.md                    # task list: what is done, what is left
├── fix.md                     # proposed changes to existing content, waiting for approval
├── README.md                  # public project page + progress summary
├── mkdocs.yml                 # site config, theme and navigation bar
├── requirements.txt           # Python packages for the build
├── .github/workflows/deploy.yml   # builds and publishes to GitHub Pages
├── sources/                   # CME295 lecture summaries (Gemini notebook); git-ignored, never commit
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

The book is being checked against Stanford's [CME295: Transformers & Large Language Models](https://cme295.stanford.edu/syllabus/) (Afshine and Shervine Amidi). The course has 9 lectures; slides appear one a week. Summaries made with a Gemini notebook are in `sources/` (one file per lecture).

- **One lecture at a time.** For each lecture, compare every topic with the book and write only the gaps (not the topics already covered) into `fix.md`, section "CME295 gap analysis": what is missing, whether it needs a new chapter or belongs in an existing one (give the chapter and section), and a priority. The user decides each item before anything is written.
- **Summaries are pointers, not facts.** Verify every claim and number against the original paper or official documentation before it goes into the book.
- **Paraphrase and credit.** Do not copy slide text or images; credit CME295 in the chapter's Sources when its lecture shaped a section.
- **`sources/` is git-ignored** and must never be committed or published.
- **The glossary (Phase 6) waits until Lecture 9 has been reviewed.**
- **After the review,** the user reads each chapter and flags unclear parts; rewrite those with a definition plus a concrete example (see Writing style).

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
  - Fraud example (Chapter 5): an LLM labelling transactions as fraud / not fraud. Evaluation sample of 100 transactions with 10 fraud (TP 8, FN 2, FP 6, TN 84 at threshold 0.5); real traffic 100,000 a month, 2% fraud.
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
- 2026-10-17 — Chapter 1 — Import cleanup — 11 latex blocks converted to math, date line removed, encoder–decoder Mermaid diagram added.
- 2026-10-17 — Chapter 2 — Formulas — 8 latex blocks converted to math blocks.
- 2026-10-17 — Chapter 2 — Status and number check — chapter recorded as written; worked numbers checked (33 of 36 correct); proposed fixes listed in fix.md.
- 2026-10-17 — Chapter 2 — Fixes F2–F7 — three number corrections, perplexity link to Chapter 5, 23 chapter links, two running examples added to the home page.
- 2026-10-17 — Chapter 3 — Big picture + Inference vs training — the generation loop; what is switched off at inference; memory example (260 MB vs 1,040 MB); why the forward pass is deterministic.
- 2026-10-17 — Chapter 3 — Generation step by step — Ti amo molto generated in 4 decoder passes with the encoder run once; decoder-only version with a prompt; comparison table.
- 2026-10-17 — Chapter 3 — Prefill vs decode — prefill is compute-bound, decode memory-bound; base-model FLOPs-per-byte table; 7B example (28 ms first token, 7 ms per token, 143 tokens/s).
- 2026-10-17 — Chapter 3 — The KV cache — why earlier K and V never change; cache growth table (4 vs 10 positions); memory formula; base model 24.6 KB/token, 7B model 0.52 MB/token and 68.7 GB for 32 users.
- 2026-10-17 — Chapter 3 — Choosing the next token: greedy and beam search — shared CAT/DOG table; beam of 2 finds DOG <EOS> (0.200) where greedy ends at 0.154; log scores and length penalty.
- 2026-10-17 — Chapter 3 — Sampling and temperature — sampling defined; T = 2 worked step by step; table at T = 0.5, 1, 2; order never changes; T = 0 means greedy.
- 2026-10-17 — Chapter 3 — Top-k sampling — k = 2 worked (0.785 / 0.215); table for k = 1, 2, 3; why a fixed k fails on confident vs unsure prompts.
- 2026-10-17 — Chapter 3 — Top-p (nucleus) sampling — p = 0.9 worked with running totals; table for p = 0.7, 0.9, 0.99; nucleus of 1 vs 6 tokens on the confident vs unsure prompts.
- 2026-10-17 — Chapter 3 — How the rules combine + typical settings — pipeline diagram; T = 2, k = 3, p = 0.9 worked (CAT 0.657, DOG 0.343) vs T = 1 (0.785 / 0.215); settings table by task.
- 2026-10-17 — Chapter 3 — Stopping — end token, max tokens and stop sequences with examples; stop_reason; context window (8,192 − 7,000 = 1,192); decoding IDs back to text and streaming.
- 2026-10-17 — Chapter 3 — Cheat sheet + Sources — 19-row cheat sheet and 8 sources; chapter complete (draft).
- 2026-10-17 — Chapter 1 — Training and generation (F1) — two 4-step lists replaced by a summary that links to Chapters 2 and 3.
- 2026-10-17 — Chapter 4 — Big picture + Why size matters — memory = weights × bytes; 7B model at 28 / 14 / 7 / 3.5 GB; decode speed limit 71 / 143 / 286 / 571 tokens/s.
- 2026-10-17 — Chapter 4 — Number formats — sign/exponent/fraction; FP32, FP16, BF16 bit layouts; stored-value table computed with NumPy (FP16 overflow at 70,000, underflow at 1e-8); INT8/INT4 ranges.
- 2026-10-17 — Chapter 4 — How quantization works — scale formula; five weights to INT8 (max error 0.0018) and INT4 (0.0576); zero point; outlier problem; group-wise scales (4.5 bits per weight, 3.9 GB for 7B).
- 2026-10-17 — Chapter 4 — Kinds of quantization — PTQ vs QAT table; weights-only (W4A16), weights + activations (W8A8), KV cache (68.7 → 34.4 GB); GPTQ, AWQ, GGUF (Q8_0 = 8.5 bits, Q4_0 = 4.5 bits).
- 2026-10-17 — Chapter 4 — Quality vs speed — 7B trade-off table (16 to 2.5 bits; 14.0 to 2.2 GB; 143 to 914 tokens/s); when quality drops; larger model at 4 bits vs smaller at 16; how to check a quantized model.
- 2026-10-17 — Chapter 4 — What changed since 2017 — overview table linking each change to its Chapter 1 section; 2017 base model vs Llama 3 8B; pre-norm, RMSNorm ([2,4,6,8] example), RoPE (distance-only score), GQA (KV cache 0.52 → 0.13 MB/token), SwiGLU, context and vocabulary.
- 2026-10-17 — Chapter 4 — Cheat sheet + Sources — 22-row cheat sheet and 13 sources; chapter complete (draft).
- 2026-10-17 — Chapter 3 — Links to Chapter 4 (F8) — four page links pointed to exact sections; grouped-query attention linked.
- 2026-10-17 — Chapter 5 — Big picture + Three levels of evaluation — fraud example defined (100-transaction sample with 10 fraud; real traffic 100,000/month at 2%); loss → model metrics → business outcomes table.
- 2026-10-17 — Chapter 5 — Loss vs metric — four fraud cases: loss sees every probability change, accuracy only right/wrong; side-by-side table.
- 2026-10-17 — Chapter 5 — Perplexity — formula; CAT 1.30, translation 1.47, uniform-4 = 4.00, perfect = 1; same-tokenizer caveat (6 vs 10 tokens: 1.65 vs 1.35); limits.
- 2026-10-17 — Chapter 5 — MSE — amount-at-risk example (MSE 1,031.25, RMSE 32.11, MAE 23.75); why token IDs have no distance; cross-entropy vs MSE table.
- 2026-10-17 — Chapter 5 — Classification metrics — confusion matrix (TP 8, FN 2, FP 6, TN 84); accuracy trap ("always OK" = 98% on real traffic); precision 0.571, recall 0.80, F1 0.667; ROC threshold table and AUC ≈ 0.94.
- 2026-10-17 — Chapter 5 — Evaluating generative LLMs — benchmarks (MMLU, HumanEval) and contamination; human evaluation; LLM-as-judge checked against an analyst (88% agreement); judge biases table; method comparison.
- 2026-10-17 — Chapter 5 — Business outcomes — KPI chain table; precision falls from 0.571 (sample) to 0.197 (real traffic); rules engine vs LLM (€370,000 vs €200,867 a month, 311 hours saved, ROI ≈ 5.8, payback 1.8 months); CSAT/NPS/complaints; baselines and A/B tests; threshold trade-off table (0.5 meets all KPI targets).
- 2026-10-17 — Chapter 5 — Choosing metrics for a project — 8-step checklist (objective to drift monitoring) filled in for the fraud example; data vs concept drift; re-evaluate after quantization or decoding changes.
- 2026-10-17 — Chapter 5 — Cheat sheet + Sources — 22-row cheat sheet and 9 sources; chapter complete (draft).
- 2026-10-17 — Chapters 2–4 — Links to Chapter 5 (F9) — three page links pointed to exact sections.
- 2026-10-17 — Introduction — Big picture + What you need to know first — vectors, dot product (CAT·DOG 0.888, CAT·A −0.506), shapes, 2×3 times 3×2 by hand, transpose; neuron with ReLU; loss, training, embeddings.
- 2026-10-17 — Introduction — Historical evolution of text modelling — timeline 1997–2026; task-specific RNN era; 2017 turning point; scaling table (65M → 1.5B → 175B, about 2,700×); agents.
- 2026-10-17 — Introduction — Vector embeddings — one-hot on a 5-word vocabulary (all dot products 0, all distances 1.414); Word2Vec CBOW and skip-gram; Paris − France + Germany = (4, 3), nearest Berlin (0.14); polysemy and word-order limits.
- 2026-10-17 — Introduction — RNNs and LSTMs — RNN formula and diagram; 3-step worked example (h = 0.462, 0.551, 0.189); LSTM gates; vanishing/exploding table (0.8ⁿ vs 1.2ⁿ); long-range and sequential limits; bridge table to Chapter 1.
- 2026-10-17 — Introduction — Cheat sheet + Sources — 19-row cheat sheet and 13 sources; chapter complete (draft).
- 2026-10-17 — CME295 review — Lecture 1 — compared with the Introduction and Chapter 1; no new chapter needed; 5 gaps proposed in fix.md (context bottleneck, tokenizer granularity, special tokens, WordPiece/Unigram, Word2Vec training) plus 2 notes.
- 2026-10-17 — CME295 review — Lecture 2 — 7 gaps proposed in fix.md: Mixture of Experts (Ch 4), prompting and reasoning (new Chapter 6 recommended), constrained decoding (Ch 3), masked language modelling (Ch 2), long-context quality (Ch 3, 5), RoPE context extension (Ch 4), tokens-to-pages table (Ch 3).
- 2026-10-17 — CME295 review — Lecture 3 — 9 gaps proposed in fix.md: pre-training data, model collapse, packing, LoRA in depth, preference-training maths, distillation (Ch 2); bits per byte, maths/code benchmarks with Elo and pass@k (Ch 5); reasoning models and GRPO (new Ch 6).
- 2026-10-17 — Chapter 2 — Pre-training data + model collapse (L3-1, L3-2) — pipeline diagram; filtering example (1,000,000 pages → 113,400, 11.3%); data mixtures and proxy models; training stages; collapse example (0.999¹⁰⁰⁰ ≈ 0.368).
- 2026-10-17 — Chapter 2 — Packing (L3-4) — padding waste (2 of 12; 76% at 1,000 of 4,096); packed row with zero padding; document mask table.
- 2026-10-17 — Chapter 2 — Masked language modelling (L2-4) — BERT 15% / 80-10-10 rule; worked YOUR CAT IS A [MASK] CAT (loss 0.511); next sentence prediction, [CLS], [SEP]; comparison table; link added in Chapter 1.
- 2026-10-17 — Chapter 2 — Preference training in detail (L3-6) — Bradley–Terry reward model (σ(1.5) = 0.818, loss 0.201 vs 1.701); KL penalty and reward hacking (KL 0.002 vs 0.125); DPO loss (0.693 → 0.474); RLHF vs DPO table (56 GB vs 28 GB).
- 2026-10-17 — Chapter 2 — LoRA in depth + instruction tuning (L3-5) — LoRA paragraph kept and expanded into a subsection: W = W₀ + B·A, 2×2 rank-1 example, 0.4% at 4,096², about 20 million adapters for a 7B model, full vs LoRA vs QLoRA memory, adapter swapping, PEFT family; "instruction tuning" named in Stage 2.
- 2026-10-17 — Chapter 2 — Distillation (L3-8) — soft-target loss; students A and B (one-hot 0.357 vs 0.051, soft 0.628 vs 0.855; minimum 0.607); three distillation methods incl. on-policy; link from Chapter 4.
- 2026-10-17 — Chapter 2 — Cheat sheet — 11 rows added for the CME295 topics (data pipeline to distillation).
- 2026-10-17 — Chapter 4 — RoPE context extension (L2-6) — position interpolation with the 0.1-radian example (speed 0.05: position 30 → 1.5; cos 0.989 vs 0.955); YaRN.
- 2026-10-17 — Chapter 4 — Mixture of Experts (L2-1) — router and top-2 example (weights 0.599 / 0.401, output [1.802, −0.997]); total vs active (Mixtral 46.7B / 12.9B, DeepSeek-V3 671B / 37B); cost table; load-balancing loss (1.0 vs 3.6); fine-grained and shared experts; overview row, opening list and cheat sheet updated.
