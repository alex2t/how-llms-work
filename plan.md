# plan.md — what is done and what is left

Tick a task `[x]` as soon as it is finished and passes `mkdocs build --strict`. Add new tasks when you discover them. Rules for editing existing content are in `CLAUDE.md` (propose first, never delete without asking).

Legend: `[ ]` to do · `[x]` done · `[~]` in progress

---

## Phase 0 — Setup and import

- [x] Scaffold: `mkdocs.yml` (Material theme, navigation tabs, prev/next footer, search, MathJax), `requirements.txt`, GitHub Pages workflow, home page, chapter placeholders.
- [x] Create the GitHub repo `how-llms-work`, push, and set Settings → Pages → Source = **GitHub Actions**. Confirm the site loads.
- [x] Import Chapter 1: the user exports the Claude Doc "How Transformers Work: BPE, Attention and Q/K/V" as Markdown and saves it as `docs/01-transformers.md` (replacing the placeholder). Then:
  - [x] Convert every ```` ```latex ```` block to a `$$ … $$` math block.
  - [x] Remove the date/byline line under the title; keep the `#` title.
  - [x] Replace the placeholder line `[embedded content: original Transformer · encoder and decoder stacks]` with a Mermaid diagram of the encoder–decoder (source tokens → self-attention → feed-forward → K,V → decoder cross-attention; decoder: output so far → masked self-attention → cross-attention → feed-forward → linear + softmax → next-token probabilities).
  - [x] Unescape `\[` / `\]` and `\_` where they render wrongly; check every table renders. (Checked: all render correctly, no change needed.)
  - [x] `mkdocs build --strict` passes; check all formulas display as math.
- [x] Shorten Chapter 1's "Training and generation" section into a summary that links to Chapters 2 and 3. **Propose the new text first** (it replaces existing content). Done: `fix.md`, F1.

## Phase 1 — Introduction: foundations and history (`docs/00-introduction.md`)

Goal: give the reader the background needed before Chapter 1, and the story of why Transformers replaced earlier approaches.

- [x] **What you need to know first** (prerequisites), each with a tiny worked example:
  - [x] Linear algebra: vectors, matrices, shapes, matrix multiplication (2×3 times 3×2 by hand), dot product as similarity, transpose.
  - [x] Basic machine learning: what a neural network is (weights, layers, activation), what a loss function is, training as "adjust weights to reduce the loss", what an embedding is (a learned vector per item). Point forward to Chapters 1 and 2 for detail.
- [x] **Historical evolution of text modeling** (timeline table + short text):
  - [x] Task-specific era (2010s): a separate specialised RNN model per task (sentiment analysis, translation, named-entity recognition); no sharing between tasks.
  - [x] The Transformer turning point (2017): "Attention Is All You Need"; one architecture that scales with compute, data and parameters. Link to Chapter 1.
  - [x] Scaling to modern LLMs (2022–2026): growing parameter counts and datasets lead to chat assistants and autonomous coding agents.
- [x] **Vector embeddings: static representations**
  - [x] One-hot encoding: each token is an orthogonal vector; worked example with a 5-word vocabulary showing every dot product is 0, so "cat" is as far from "dog" as from "bank" (no semantic relationship).
  - [x] Word2Vec: proxy tasks CBOW (predict a word from its context) and Skip-gram (predict the context from a word); learned semantic space; vector arithmetic Paris − France + Germany ≈ Berlin, with a small 2-D illustrative example.
  - [x] Word2Vec limitations: one vector per word, so no polysemy (river "bank" vs money "bank"); ignores word order. Link to Chapter 1 "One fixed vector per token, different meaning in context".
- [x] **Sequence architectures: RNNs and LSTMs**
  - [x] RNN: hidden state carried from step to step; small diagram (x1 → RNN → y1, state passed to the next step); worked 3-step example.
  - [x] LSTM: adds a cell state and gates to keep information longer (high level only).
  - [x] Limitations: vanishing/exploding gradients, long-range dependencies are hard, sequential processing blocks parallel training. Close with a bridge to Chapter 1: attention removes all three.
- [x] Cheat sheet + Sources.

## Phase 2 — Chapter 2: Training (`docs/02-training.md`)

- [x] Convert every ```` ```latex ```` block to a `$$ … $$` math block (8 blocks).
- [x] What training is: predict the next token, measure the error, adjust every weight. Link to Chapter 1 "What is learned, and what changes after training?".
- [x] Training data and batches: what a batch is; why sequences in a batch must have the same length.
- [x] Padding and the padding mask: add `<pad>` tokens so a batch is rectangular, then mask them so attention and the loss ignore them. 2-sentence batch + its padding mask; contrast with the causal mask in Chapter 1.
- [x] Decoder input "shifted right": `<SOS>` / `<EOS>`. Encoder input "`<SOS>` I love you very much `<EOS>`", decoder input "`<SOS>` Ti amo molto", target "Ti amo molto `<EOS>`". Input/target alignment table.
- [x] Teacher forcing, and why training happens in one time step: the whole target sentence is fed at once and the causal mask hides each position's answer, so all positions train in parallel. Contrast with word-by-word generation (Chapter 3).
- [x] The loss: cross-entropy, worked example with the CAT/DOG/IS/LOVELY probabilities. Two-line teaser for perplexity → Chapter 5.
- [x] Backpropagation and gradient descent in plain words; Adam, learning rate, warm-up schedule from the paper; dropout and label smoothing.
- [x] Stages of an LLM's life: pre-training, fine-tuning, RLHF/preference training; what changes in each.
- [x] Cheat sheet + Sources.
- [x] Check every worked number with a script (33 of 36 correct).
- [x] Apply the approved Chapter 2 fixes from `fix.md` (F2–F6: three number corrections, perplexity link, chapter links).

## Phase 3 — Chapter 3: Inference and decoding (`docs/03-inference.md`)

- [x] Inference vs training: frozen weights, no loss, no backpropagation.
- [x] Generation step by step: the translation produced one word per time step (Ti → amo → molto → `<EOS>`), reusing the encoder output; then the same idea for decoder-only chat models.
- [x] Prefill (prompt processing) vs decode (token generation); why the first token is slower.
- [x] The KV cache: why earlier keys and values are stored; memory estimate.
- [x] Choosing the next token, on one shared probability example:
  - [x] Greedy decoding and beam search (beam of 2 worked through).
  - [x] Temperature: divide logits by T before softmax; T = 0.5, 1, 2 on the same logits.
  - [x] Top-k: keep the k most likely tokens, renormalise.
  - [x] Top-p (nucleus): keep the smallest set summing to p, renormalise.
  - [x] How they combine; typical settings for factual vs creative output.
- [x] Stopping: `<EOS>`, max tokens, stop sequences.
- [x] Cheat sheet + Sources.
- [x] Point the Chapter 4 links to exact sections once Chapter 4 exists (`fix.md`, F8).

## Phase 4 — Chapter 4: Quantization, efficiency and the modern Transformer (`docs/04-quantization.md`)

- [x] Why size matters: parameters × bytes per parameter = memory; worked example for a 7B model.
- [x] Number formats: FP32, FP16, BF16, INT8, INT4 — size and precision table.
- [x] How quantization works: scale and zero-point; worked example rounding a few weights to INT8 and back; the rounding error.
- [x] Kinds of quantization: post-training vs quantization-aware; weights-only vs weights + activations; KV-cache quantization; GPTQ, AWQ, GGUF at a high level.
- [x] Quality vs speed trade-offs; when quality drops noticeably.
- [x] "What changed since 2017" box (table: what it is, why adopted, which Chapter 1 section it modifies):
  - [x] decoder-only instead of encoder–decoder
  - [x] pre-norm instead of post-norm; RMSNorm instead of LayerNorm
  - [x] RoPE instead of sinusoidal positional encoding
  - [x] grouped-query attention (GQA) to shrink the KV cache
  - [x] SwiGLU instead of ReLU in the feed-forward network
  - [x] larger context windows and vocabularies
- [x] Cheat sheet + Sources.

## Phase 5 — Chapter 5: Evaluation (`docs/05-evaluation.md`)

- [x] Three levels of evaluation (opening table): training loss → model metrics → business outcomes.
- [x] Loss vs metric; link to cross-entropy in Chapter 2.
- [x] Perplexity: e^cross-entropy, worked example from CAT/DOG/IS/LOVELY; what a good value means; only comparable with the same tokenizer.
- [x] MSE loss: worked example on predicted vs true numbers; why it suits regression, not next-token prediction.
- [x] Classification metrics on the fraud example (100 transactions):
  - [x] Confusion matrix (TP, FP, TN, FN).
  - [x] Accuracy and why it misleads on imbalanced data (2% fraud).
  - [x] Precision, recall, F1 from the same matrix; when to favour each.
  - [x] ROC curve and AUC: moving the threshold, TPR vs FPR, AUC = 0.5 vs 1.0; small threshold table.
- [x] Evaluating generative LLMs: benchmarks (limits, contamination), human evaluation, LLM-as-judge.
- [x] Business-centric (outcome) metrics, same fraud example:
  - [x] KPIs tied to a business objective: objective → KPI → model metric chain table.
  - [x] Time, cost and resource improvement: analyst hours saved, cost per case vs model running cost (link Chapter 4); worked ROI example.
  - [x] Customer satisfaction: false positives as friction, CSAT/NPS, complaint rate; why precision matters to customers.
  - [x] Benchmark comparison against a baseline (manual process, rules engine, previous model); A/B tests, before/after.
  - [x] Trade-off table: threshold vs recall, precision, analyst workload, customer friction.
- [x] Choosing metrics for a project: checklist (objective → KPI → model metric → baseline → target → post-go-live measurement and drift monitoring).
- [x] Cheat sheet + Sources.
- [x] Point earlier links to Chapter 5 at exact sections (`fix.md`, F9).

## Phase 6 — Polish

- [ ] Glossary page (`docs/glossary.md`) collecting every defined term with a link to where it is explained; add to the nav. **Wait until CME295 Lecture 9 has been reviewed (Phase 7).**
- [ ] Cross-link check: every "see Chapter N" is a working link.
- [ ] Read-through for consistency of running examples and notation across all chapters.

## Phase 7 — CME295 lecture review

Compare the book with each lecture of Stanford's CME295 (summaries in `sources/`, git-ignored). For each lecture, write the missing topics into `fix.md` ("CME295 gap analysis") with a recommendation: new chapter, or which existing chapter and section. The user decides; approved items become tasks here. Slides appear one a week.

- [x] Set-up: `sources/` git-ignored; credits in `README.md`; workflow in `CLAUDE.md`.
- [x] Lecture 1 — Transformer foundations, tokenization, embeddings, attention. Gap analysis in `fix.md`; L1-1 to L1-5 approved.
  - [ ] L1-1: Introduction → "Why RNNs hit a wall": the context bottleneck as a fourth problem, plus a row in the bridge table.
  - [ ] L1-2: Chapter 1 → BPE tokenizer: subsection "Word, character, subword or byte?" with a table and worked example.
  - [ ] L1-3: Chapter 1 special-tokens table; Chapter 3 worked chat example with role tokens.
  - [ ] L1-4: Chapter 1 → Practical consequences: short note on WordPiece and Unigram.
  - [ ] L1-5: Introduction → Word2Vec: the network and negative sampling.
- [x] Lecture 2 — Model families, Mixture of Experts, normalisation, context, decoding, prompting. Gap analysis in `fix.md`; L2-1 to L2-7 approved.
  - [ ] L2-1: Chapter 4 → new section "Mixture of Experts: more weights, same compute" + row in the overview table.
  - [ ] L2-2: new Chapter 6 "Prompting, reasoning and agents" (`docs/06-prompting.md`, added to nav): in-context learning, chain-of-thought, self-consistency.
  - [ ] L2-3: Chapter 3 → new subsection "Constrained decoding" with the FRAUD / OK example.
  - [x] L2-4: Chapter 2 → new section "Another training objective: masked language modelling"; link from Chapter 1.
  - [ ] L2-5: Chapter 3 → long-context advice; Chapter 5 → needle-in-a-haystack benchmark.
  - [ ] L2-6: Chapter 4 → RoPE: paragraph on context extension.
  - [ ] L2-7: Chapter 3 → tokens-to-words-to-pages table.
- [x] Lecture 3 — Pre-training data, SFT, LoRA, RLHF vs DPO, GRPO, distillation. Gap analysis in `fix.md`; L3-1 to L3-9 approved.
  - [x] L3-1: Chapter 2 → section "Where pre-training data comes from" (pipeline diagram, filtering example).
  - [x] L3-2: Chapter 2 → model collapse, inside L3-1's section.
  - [ ] L3-3: Chapter 5 → bits per byte after the two-tokenizer example.
  - [x] L3-4: Chapter 2 → "Packing: the alternative to padding".
  - [ ] L3-5: Chapter 2 → subsection "Cheap fine-tuning: LoRA"; "instruction tuning" in Stage 2.
  - [x] L3-6: Chapter 2 → Stage 3 expanded: reward model, KL penalty, DPO loss, 4 vs 2 models.
  - [ ] L3-7: Chapter 6 → section "Reasoning models: thinking before answering" (test-time compute, RLVR, GRPO).
  - [ ] L3-8: Chapter 2 → section "Distillation"; link from Chapter 4.
  - [ ] L3-9: Chapter 5 → benchmark families, Elo and pass@k.
- [ ] Lecture 4 (when published).
- [ ] Lecture 5 (when published).
- [ ] Lecture 6 (when published).
- [ ] Lecture 7 (when published).
- [ ] Lecture 8 (when published).
- [ ] Lecture 9 (when published), then the glossary (Phase 6).
- [ ] User read-through of every chapter; rewrite the parts flagged as unclear.
