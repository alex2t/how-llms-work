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
- [ ] Shorten Chapter 1's "Training and generation" section into a summary that links to Chapters 2 and 3. **Propose the new text first** (it replaces existing content).

## Phase 1 — Introduction: foundations and history (`docs/00-introduction.md`)

Goal: give the reader the background needed before Chapter 1, and the story of why Transformers replaced earlier approaches.

- [ ] **What you need to know first** (prerequisites), each with a tiny worked example:
  - [ ] Linear algebra: vectors, matrices, shapes, matrix multiplication (2×3 times 3×2 by hand), dot product as similarity, transpose.
  - [ ] Basic machine learning: what a neural network is (weights, layers, activation), what a loss function is, training as "adjust weights to reduce the loss", what an embedding is (a learned vector per item). Point forward to Chapters 1 and 2 for detail.
- [ ] **Historical evolution of text modeling** (timeline table + short text):
  - [ ] Task-specific era (2010s): a separate specialised RNN model per task (sentiment analysis, translation, named-entity recognition); no sharing between tasks.
  - [ ] The Transformer turning point (2017): "Attention Is All You Need"; one architecture that scales with compute, data and parameters. Link to Chapter 1.
  - [ ] Scaling to modern LLMs (2022–2026): growing parameter counts and datasets lead to chat assistants and autonomous coding agents.
- [ ] **Vector embeddings: static representations**
  - [ ] One-hot encoding: each token is an orthogonal vector; worked example with a 5-word vocabulary showing every dot product is 0, so "cat" is as far from "dog" as from "bank" (no semantic relationship).
  - [ ] Word2Vec: proxy tasks CBOW (predict a word from its context) and Skip-gram (predict the context from a word); learned semantic space; vector arithmetic Paris − France + Germany ≈ Berlin, with a small 2-D illustrative example.
  - [ ] Word2Vec limitations: one vector per word, so no polysemy (river "bank" vs money "bank"); ignores word order. Link to Chapter 1 "One fixed vector per token, different meaning in context".
- [ ] **Sequence architectures: RNNs and LSTMs**
  - [ ] RNN: hidden state carried from step to step; small diagram (x1 → RNN → y1, state passed to the next step); worked 3-step example.
  - [ ] LSTM: adds a cell state and gates to keep information longer (high level only).
  - [ ] Limitations: vanishing/exploding gradients, long-range dependencies are hard, sequential processing blocks parallel training. Close with a bridge to Chapter 1: attention removes all three.
- [ ] Cheat sheet + Sources.

## Phase 2 — Chapter 2: Training (`docs/02-training.md`)

- [x] Convert every ```` ```latex ```` block to a `$$ … $$` math block (8 blocks).
- [ ] What training is: predict the next token, measure the error, adjust every weight. Link to Chapter 1 "What is learned, and what changes after training?".
- [ ] Training data and batches: what a batch is; why sequences in a batch must have the same length.
- [ ] Padding and the padding mask: add `<pad>` tokens so a batch is rectangular, then mask them so attention and the loss ignore them. 2-sentence batch + its padding mask; contrast with the causal mask in Chapter 1.
- [ ] Decoder input "shifted right": `<SOS>` / `<EOS>`. Encoder input "`<SOS>` I love you very much `<EOS>`", decoder input "`<SOS>` Ti amo molto", target "Ti amo molto `<EOS>`". Input/target alignment table.
- [ ] Teacher forcing, and why training happens in one time step: the whole target sentence is fed at once and the causal mask hides each position's answer, so all positions train in parallel. Contrast with word-by-word generation (Chapter 3).
- [ ] The loss: cross-entropy, worked example with the CAT/DOG/IS/LOVELY probabilities. Two-line teaser for perplexity → Chapter 5.
- [ ] Backpropagation and gradient descent in plain words; Adam, learning rate, warm-up schedule from the paper; dropout and label smoothing.
- [ ] Stages of an LLM's life: pre-training, fine-tuning, RLHF/preference training; what changes in each.
- [ ] Cheat sheet + Sources.

## Phase 3 — Chapter 3: Inference and decoding (`docs/03-inference.md`)

- [ ] Inference vs training: frozen weights, no loss, no backpropagation.
- [ ] Generation step by step: the translation produced one word per time step (Ti → amo → molto → `<EOS>`), reusing the encoder output; then the same idea for decoder-only chat models.
- [ ] Prefill (prompt processing) vs decode (token generation); why the first token is slower.
- [ ] The KV cache: why earlier keys and values are stored; memory estimate.
- [ ] Choosing the next token, on one shared probability example:
  - [ ] Greedy decoding and beam search (beam of 2 worked through).
  - [ ] Temperature: divide logits by T before softmax; T = 0.5, 1, 2 on the same logits.
  - [ ] Top-k: keep the k most likely tokens, renormalise.
  - [ ] Top-p (nucleus): keep the smallest set summing to p, renormalise.
  - [ ] How they combine; typical settings for factual vs creative output.
- [ ] Stopping: `<EOS>`, max tokens, stop sequences.
- [ ] Cheat sheet + Sources.

## Phase 4 — Chapter 4: Quantization, efficiency and the modern Transformer (`docs/04-quantization.md`)

- [ ] Why size matters: parameters × bytes per parameter = memory; worked example for a 7B model.
- [ ] Number formats: FP32, FP16, BF16, INT8, INT4 — size and precision table.
- [ ] How quantization works: scale and zero-point; worked example rounding a few weights to INT8 and back; the rounding error.
- [ ] Kinds of quantization: post-training vs quantization-aware; weights-only vs weights + activations; KV-cache quantization; GPTQ, AWQ, GGUF at a high level.
- [ ] Quality vs speed trade-offs; when quality drops noticeably.
- [ ] "What changed since 2017" box (table: what it is, why adopted, which Chapter 1 section it modifies):
  - [ ] decoder-only instead of encoder–decoder
  - [ ] pre-norm instead of post-norm; RMSNorm instead of LayerNorm
  - [ ] RoPE instead of sinusoidal positional encoding
  - [ ] grouped-query attention (GQA) to shrink the KV cache
  - [ ] SwiGLU instead of ReLU in the feed-forward network
  - [ ] larger context windows and vocabularies
- [ ] Cheat sheet + Sources.

## Phase 5 — Chapter 5: Evaluation (`docs/05-evaluation.md`)

- [ ] Three levels of evaluation (opening table): training loss → model metrics → business outcomes.
- [ ] Loss vs metric; link to cross-entropy in Chapter 2.
- [ ] Perplexity: e^cross-entropy, worked example from CAT/DOG/IS/LOVELY; what a good value means; only comparable with the same tokenizer.
- [ ] MSE loss: worked example on predicted vs true numbers; why it suits regression, not next-token prediction.
- [ ] Classification metrics on the fraud example (100 transactions):
  - [ ] Confusion matrix (TP, FP, TN, FN).
  - [ ] Accuracy and why it misleads on imbalanced data (2% fraud).
  - [ ] Precision, recall, F1 from the same matrix; when to favour each.
  - [ ] ROC curve and AUC: moving the threshold, TPR vs FPR, AUC = 0.5 vs 1.0; small threshold table.
- [ ] Evaluating generative LLMs: benchmarks (limits, contamination), human evaluation, LLM-as-judge.
- [ ] Business-centric (outcome) metrics, same fraud example:
  - [ ] KPIs tied to a business objective: objective → KPI → model metric chain table.
  - [ ] Time, cost and resource improvement: analyst hours saved, cost per case vs model running cost (link Chapter 4); worked ROI example.
  - [ ] Customer satisfaction: false positives as friction, CSAT/NPS, complaint rate; why precision matters to customers.
  - [ ] Benchmark comparison against a baseline (manual process, rules engine, previous model); A/B tests, before/after.
  - [ ] Trade-off table: threshold vs recall, precision, analyst workload, customer friction.
- [ ] Choosing metrics for a project: checklist (objective → KPI → model metric → baseline → target → post-go-live measurement and drift monitoring).
- [ ] Cheat sheet + Sources.

## Phase 6 — Polish

- [ ] Glossary page (`docs/glossary.md`) collecting every defined term with a link to where it is explained; add to the nav.
- [ ] Cross-link check: every "see Chapter N" is a working link.
- [ ] Read-through for consistency of running examples and notation across all chapters.
