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
  - [x] L1-1: Introduction → "Why RNNs hit a wall": the context bottleneck as a fourth problem, plus a row in the bridge table.
  - [x] L1-2: Chapter 1 → BPE tokenizer: subsection "Word, character, subword or byte?" with a table and worked example.
  - [x] L1-3: Chapter 1 special-tokens table; Chapter 3 worked chat example with role tokens.
  - [x] L1-4: Chapter 1 → Practical consequences: short note on WordPiece and Unigram.
  - [x] L1-5: Introduction → Word2Vec: the network and negative sampling.
- [x] Lecture 2 — Model families, Mixture of Experts, normalisation, context, decoding, prompting. Gap analysis in `fix.md`; L2-1 to L2-7 approved.
  - [x] L2-1: Chapter 4 → new section "Mixture of Experts: more weights, same compute" + row in the overview table.
  - [x] L2-2: new Chapter 6 "Prompting, reasoning and agents" (`docs/06-prompting.md`, added to nav): in-context learning, chain-of-thought, self-consistency.
  - [x] L2-3: Chapter 3 → new subsection "Constrained decoding" with the FRAUD / OK example.
  - [x] L2-4: Chapter 2 → new section "Another training objective: masked language modelling"; link from Chapter 1.
  - [x] L2-5: Chapter 3 → long-context advice; Chapter 5 → needle-in-a-haystack benchmark.
  - [x] L2-6: Chapter 4 → RoPE: paragraph on context extension.
  - [x] L2-7: Chapter 3 → tokens-to-words-to-pages table.
- [x] Lecture 3 — Pre-training data, SFT, LoRA, RLHF vs DPO, GRPO, distillation. Gap analysis in `fix.md`; L3-1 to L3-9 approved.
  - [x] L3-1: Chapter 2 → section "Where pre-training data comes from" (pipeline diagram, filtering example).
  - [x] L3-2: Chapter 2 → model collapse, inside L3-1's section.
  - [x] L3-3: Chapter 5 → bits per byte after the two-tokenizer example.
  - [x] L3-4: Chapter 2 → "Packing: the alternative to padding".
  - [x] L3-5: Chapter 2 → subsection "Cheap fine-tuning: LoRA"; "instruction tuning" in Stage 2.
  - [x] L3-6: Chapter 2 → Stage 3 expanded: reward model, KL penalty, DPO loss, 4 vs 2 models.
  - [x] L3-7: Chapter 6 → section "Reasoning models: thinking before answering" (test-time compute, RLVR, GRPO).
  - [x] L3-8: Chapter 2 → section "Distillation"; link from Chapter 4.
  - [x] L3-9: Chapter 5 → benchmark families, Elo and pass@k.
  - [x] Follow-up: link "as Chapter 6 explains" in Chapter 5 (Benchmark families) to the reasoning-models section once Chapter 6 exists.
  - [x] Follow-up: link "Chapter 6 covers them" (Chapter 1, Special tokens) to the reasoning-models section once Chapter 6 exists.
- [ ] Lecture 4 (when published).
- [ ] Lecture 5 (when published).
- [ ] Lecture 6 (when published).
- [ ] Lecture 7 (when published).
- [ ] Lecture 8 (when published).
- [ ] Lecture 9 (when published), then the glossary (Phase 6).
- [ ] When the CME295 lecture on RAG and agents is reviewed, compare it with Chapters 7, 8 and 9 and add only the gaps (Phase 8, D4).
- [ ] User read-through, one chapter at a time; rewrite the parts flagged as unclear; one commit, push and deploy per chapter (`CLAUDE.md`, rule 8).

## Phase 8 — Stanford CS230 Lecture 8: Agents, Prompts and RAG

Source: `lecture8.md`, a summary of **CS230: Deep Learning (Autumn 2025), Lecture 8**. This is a different course from CME295, so it gets its own phase. The summary is a pointer, not a fact: verify every claim and number against the original paper or documentation before it goes into the notes; paraphrase and credit CS230 in the Sources of every chapter it shapes.

IDs: `C8-n` (never reused). Priority: **H** high, **M** medium, **L** low.

### Decisions for the user (before writing)

- [x] **D1. Agents: in Chapter 6 or a new Chapter 8?** Modules 5–7 (agents, agent evaluation, multi-agent systems) are as large as RAG. Recommendation: a new **Chapter 8 "Agents"** (`docs/08-agents.md`), written after Chapter 7 because an agent uses retrieval as one of its tools. This changes Chapter 6: its title "Prompting, Reasoning and Agents" becomes "Prompting and Reasoning", and its "Work in progress" note (`docs/06-prompting.md:25`) is replaced by a link to Chapters 7 and 8 (rule 1: needs approval).
- [x] **D2. Running example for Chapter 7.** Recommendation: continue the bank from Chapter 5. A **fraud-policy assistant** answers analysts' questions from the bank's internal policy documents, e.g. *"How many days does a customer have to dispute a card payment?"*. Documents, chunks and vectors are made up (and labelled so); all arithmetic checked with a script. Once approved, add it to the running examples in `CLAUDE.md`.
- [x] **D3. `lecture8.md` sits in the repository root, untracked.** Move it into `sources/` (git-ignored), so it is never committed.
- [x] **D4. CME295 overlap.** CME295's later lectures also cover RAG and agents. Write Chapter 7 now, then compare it with the matching CME295 lecture when it is reviewed (Phase 7) and add only the gaps.

### Already covered (no new text needed)

| Lecture topic | Where in the notes |
| --- | --- |
| Zero-shot and few-shot prompting | `docs/06-prompting.md:37` |
| Chain-of-thought | `docs/06-prompting.md:95` |
| Needle in a haystack, lost in the middle | `docs/03-inference.md:711`, `docs/05-evaluation.md:428` |
| Distribution shift (as drift after go-live) | `docs/05-evaluation.md:693` |
| Control and alignment (preference training) | `docs/02-training.md:886` |
| LLM-as-judge with a rubric, pairwise position bias | `docs/05-evaluation.md:494` |
| Larger models as the "horizontal axis" of progress | `docs/00-introduction.md:443` |

### New Chapter 7: Retrieval-augmented generation (`docs/07-rag.md`)

Follow rule 14 of `CLAUDE.md` (nav, README Chapters, status table, build check). One section at a time (rule 3). Sections marked **(beyond the lecture)** are not in CS230 Lecture 8; see "RAG topics beyond the lecture" below.

- [x] C8-1 **(H) Create the chapter**: placeholder page, `nav:` entry `"7 · RAG": 07-rag.md`, README Chapters row, `CLAUDE.md` status table, `mkdocs build --strict`.
- [x] C8-2 **(H) The big picture**: what RAG is in one paragraph (look facts up, then answer from them); Mermaid diagram of the two phases: indexing (documents → chunks → embeddings → vector index) and answering (question → embedding → top-k chunks → prompt → answer with citations). Original paper: Lewis et al., 2020.
- [x] C8-3 **(H) Why a model needs outside knowledge** (Module 1): knowledge cutoff (the model knows nothing after its training data ends), private data it never saw, hallucination (define: fluent text that is false, e.g. a made-up citation), no sources to check. The options compared in one table: bigger model, prompting, prompt chaining, RAG, fine-tuning, agents (the lecture's "two axes"), each linked to its section.
- [x] C8-4 **(H) Chunking** (Module 4, step 1): why documents are cut (embedding models have an input limit; small chunks retrieve precisely); chunk size and overlap with a worked example on a policy document (tokens → number of chunks); splitting on headings and paragraphs instead of fixed lengths. Link to "chunking" for training data (`docs/02-training.md:214`) and say how the two differ.
- [x] C8-5 **(H) Embeddings for retrieval** (Module 4, step 2) **(partly beyond the lecture)**: an embedding model turns a whole chunk into one vector; how it differs from token embeddings (Chapter 1) and Word2Vec (Introduction); bi-encoder; trained with contrastive learning (pull matching question/passage pairs together, push others apart) at a high level (Sentence-BERT, DPR).
- [x] C8-6 **(H) Similarity search and vector databases** (Module 4, steps 3–4): cosine similarity vs dot product, worked example with made-up 4-dimensional vectors for one question and three chunks, top-k; what a vector database stores; **(beyond the lecture, M)** exact vs approximate nearest-neighbour search (HNSW) and why exact search is too slow for millions of chunks.
- [x] C8-7 **(H) Keyword search and hybrid search (beyond the lecture)**: why embeddings miss exact strings (an error code, a policy number); BM25 in plain words; combining both rankings with reciprocal rank fusion, worked example. The lecture's Lab 2 names hybrid search but does not explain it.
- [x] C8-8 **(H) Re-ranking (beyond the lecture)**: a cross-encoder reads the question and each chunk together and scores them; retrieve 50 cheaply, re-rank, keep 5; cost vs quality.
- [x] C8-9 **(H) Building the prompt and citing sources** (Module 4, steps 5–6): the prompt template ("answer only from these documents; if the answer is not there, say you do not know"); citations (document, section, page); chunk order and lost in the middle (link `docs/03-inference.md:716`); **(beyond the lecture, H)** prompt injection: a retrieved document that contains instructions, and why the model may follow them.
- [x] C8-10 **(M) Better retrieval** (Module 4, advanced): query rewriting **(beyond the lecture)**; HyDE (Gao et al., 2022), worked through on the running question; multi-level chunking (a summary vector per document plus vectors per paragraph); **(beyond the lecture)** contextual retrieval (prefix each chunk with a short description of its document; Anthropic, 2024).
- [x] C8-11 **(H) Evaluating a RAG system (beyond the lecture)**: evaluate retrieval and generation separately. Retrieval: recall@k and mean reciprocal rank, worked example on 5 test questions. Generation: faithfulness (is every claim supported by the retrieved text?) and answer relevance, scored with an LLM-as-judge (link `docs/05-evaluation.md:494`); RAGAS as an example tool.
- [x] C8-12 **(H) RAG, long context or fine-tuning?** (Module 1 context bottleneck, Module 3, Module 4 "RAG vs infinite context"): worked cost comparison, whole manual in the prompt vs 5 chunks (tokens per question × questions per month; prefill time, link `docs/03-inference.md:223`); decision table: when to prompt, when to retrieve, when to fine-tune (consistent format and specialised vocabulary that prompting cannot fix), and their costs.
- [x] C8-13 **(L) Beyond basic RAG (beyond the lecture)**: agentic RAG (the model decides when and what to search; bridge to Chapter 8); GraphRAG (Edge et al., 2024) in two paragraphs.
- [x] C8-14 **(H) Cheat sheet + Sources** (credit CS230 Lecture 8; Lewis 2020, HyDE, Sentence-BERT, DPR, BM25, HNSW, RRF, RAGAS, contextual retrieval, lost in the middle).

### Additions to existing chapters

New sections and subsections can be written directly; changes to existing text need approval first (rule 1).

- [x] C8-15 **(M) Chapter 6 → new section "Writing a clear prompt"** (Module 2), placed before "In-context learning": state the audience, constraints, length and format; role (persona) framing; the system prompt (link to the chat roles example in Chapter 3). Worked before/after prompt on the bank example.
- [x] C8-16 **(M) Chapter 6 → new section "Self-critique and refinement"** (Module 2), after "Self-consistency": the model drafts, critiques its own draft and rewrites it (Self-Refine, Madaan et al., 2023); its limits (a model that cannot find an error will not fix it).
- [x] C8-17 **(H) Chapter 6 → new section "Prompt chaining"** (Module 2), before "Reasoning models": split a task into single-purpose prompts; benefits (control, debugging one step, a metric per step); worked example as a 3-step chain. Bridges to agents.
- [x] C8-18 **(L) Chapter 6 → short note "Where models help and where they fail: the jagged frontier"** (Module 2): Dell'Acqua et al., 2023 (Harvard Business School field experiment with consultants); "centaurs" and "cyborgs" as two ways of working with a model. Verify the numbers in the paper before using any.
- [ ] C8-19 (moved to `fix.md`, item C8, for the next session) **(M) Chapter 3 → "A bigger window is not a perfect memory"** (`docs/03-inference.md:711`, existing text: propose first): advice for long conversations, summarise earlier turns and start afresh. The lecture's "about 8 turns" is unverified; use it only if a source supports it.
- [ ] C8-20 (moved to `fix.md`, item C8, for the next session) **(M) Chapter 5 → LLM-as-judge** (`docs/05-evaluation.md:494`, existing text: propose first): table of the three judge set-ups, pairwise comparison, single-answer grading, reference-guided grading (Zheng et al., 2023, MT-Bench).
- [ ] C8-21 (moved to `fix.md`, item C8, for the next session) **(M) Chapter 5 → new subsection "Testing prompts automatically"** (Module 2): a fixed test set run on every prompt change, with code checks plus an LLM-as-judge; PromptFoo as an example tool.
- [ ] C8-22 (moved to `fix.md`, item C8, for the next session) **(M) Chapter 2 → Stage 2 (SFT): new subsection "What fine-tuning costs"** (Module 3): catastrophic forgetting (define, with an example), need for labelled data, the risk that a newer base model overtakes the fine-tuned one; link to the decision table in C8-12.

### New Chapter 8: Agents (`docs/08-agents.md`)

Follow rule 14 of `CLAUDE.md`.

- [x] C8-23 **(H) What an agent is** (Module 5): Andrew Ng's "agentic workflow"; the loop: plan, call a tool, read the result, repeat; how this differs from ordinary software (fuzzy outputs, so guardrails and a human fallback).
- [x] C8-24 **(H) Tools and tool calling** (Module 5): how a model asks for a tool (structured output, link to constrained decoding in Chapter 3); the Model Context Protocol (MCP), verified against the official specification.
- [x] C8-25 **(H) Memory** (Module 5): working memory (the context window) vs long-term memory (stored outside, often retrieved with RAG, link to Chapter 7).
- [x] C8-26 **(M) Degrees of autonomy** (Module 5): fixed steps; fixed tools, model chooses the steps; model plans and writes its own tools. Table with an example of each.
- [x] C8-27 **(H) Evaluating an agent** (Module 6): the customer-support case study as a running example; tracing; end-to-end vs per-step evals; code checks vs LLM-as-judge; cost, latency and success rate per run. Link to Chapter 5.
- [x] C8-28 **(M) Multi-agent systems** (Module 7): why (parallel work, reusable specialists); orchestrator (hierarchical) vs peer-to-peer; the smart-home example.
- [x] C8-29 **(H) Cheat sheet + Sources** (credit CS230 Lecture 8).

### Not planned (recommendation: skip)

- **Frontier trends** (Module 7: scaling plateaus, multimodal transfer, unifying learning paradigms, forward-only learning): opinion and research directions, not how LLMs work today. At most one sentence on scaling laws in the Introduction (`docs/00-introduction.md:443`), if the user wants it.
- **Hands-on labs**: the notes do not include programming exercises.
- **Efficiency of very large models for small tasks** (Module 1): already covered by distillation (`docs/02-training.md:1065`) and model size (`docs/04-quantization.md:36`).

## Phase 9 — New Chapter 8 "RAG security and production"; Agents becomes Chapter 9

Sources: `sources/rag_security_attacks_and_defenses.md` and `sources/production_rag_recommendations.md` (git-ignored, author unknown). They are pointers, not facts: verify every claim, code snippet and model name against a primary source before it goes into the notes. Primary sources to check against: OWASP Top 10 for LLM Applications 2025 (prompt injection, sensitive information disclosure, data and model poisoning, excessive agency, vector and embedding weaknesses), Greshake et al. 2023 (indirect prompt injection), PoisonedRAG (Zou et al., 2024), the MCP specification's security section, NIST definitions of RBAC and ABAC, and the official documentation of the vector database and embedding models used in examples.

Decided by the user (2026-10-20):

- RAG security gets its **own chapter**, because it is important and Chapter 7 is long enough.
- The production RAG recommendations go into the **same new chapter**.
- The current Chapter 8 "Agents" becomes **Chapter 9**; the agent-specific defences go there.
- No product name from the sources ("Trusteer") is used anywhere in the notes.
- Vector-database query syntax and encryption at rest and in transit **are kept**.

IDs: `P9-n` (never reused). Priority: **H** high, **M** medium, **L** low. Best practice for every task: follow `CLAUDE.md` (rule 3 one section at a time with `mkdocs build --strict`; rule 5 every number computed with a script; rules 1, 2 and 9 propose changes to existing text in `fix.md` first; rule 14 new-chapter checklist; Writing style and Markdown conventions). Commit messages: `ch8: <section> — <what changed>` and `ch9: …`.

### Decisions for the user (before writing)

- [x] **D5. Title and file name.** Recommendation: title "RAG security and production", file `docs/08-rag-security.md`, nav label `"8 · RAG security": 08-rag-security.md`.
- [x] **D6. Running example.** Recommendation: keep the bank's fraud-policy assistant from Chapter 7. Made-up additions: an old version of the manual (90-day dispute limit, valid until 2025-12-31) for validity dates; a poisoned chunk claiming 365 days for data poisoning; chunks tagged "fraud team only" and an HR salary document for access control; the dispute agent from Chapter 9 for unauthorised actions. Add them to the running examples in `CLAUDE.md` once approved.
- [x] **D7. Prompt injection in Chapter 7.** Recommendation: keep the short subsection "Prompt injection through documents" (`docs/07-rag.md:482`) as it is, and add one sentence linking to the full treatment in Chapter 8 (existing text: propose first). Chapter 8 explains the attack again in full (a link never replaces an explanation).
- [x] **D8. Old links to `08-agents/`.** Renaming the file changes the published URL, so bookmarks to the old page break. Recommendation: add the `mkdocs-redirects` plugin (to `requirements.txt` and `mkdocs.yml`) with a redirect from `08-agents.md` to `09-agents.md`. Alternative: accept the broken old URL.

### Step 1: renumber Agents to Chapter 9 (before Chapter 8 is created)

The approved move of a whole chapter; no text is removed. Section numbers update automatically (`docs/javascripts/numbering.js` reads the chapter number from the file name).

- [x] P9-1 **(H) Rename** `docs/08-agents.md` to `docs/09-agents.md` with `git mv` (keeps the file's history).
- [x] P9-2 **(H) Update every reference**, then check that each anchor still matches a heading:
  - `mkdocs.yml:48` → `"9 · Agents": 09-agents.md`
  - `docs/index.md:23` → `[9 · Agents](09-agents.md)`
  - `docs/06-prompting.md:31`, `:368`, `:395` → "Chapter 9" and `09-agents.md#…`
  - `docs/07-rag.md:89`, `:491`, `:698` → "Chapter 9" and `09-agents.md#…`
  - `README.md:39` (Chapters row) and `README.md:55` (CS230 credit: "Chapters 6, 7 and 9")
  - `CLAUDE.md`: status table, repository layout (`… 09-agents.md`), running example "Dispute agent (Chapter 9)"
  - `AGENTS.md:23`: path `docs/09-agents.md` (path only; the file belongs to Antigravity)
  - this file: Phase 7's CME295 task on RAG and agents → "Chapters 7, 8 and 9"
- [x] P9-3 **(H) Check**: `grep` for any remaining `08-agents` or "Chapter 8" meaning agents; `mkdocs build --strict`; `mkdocs serve` and confirm Chapter 9 opens with numbers 9.1, 9.2, … Commit: `ch9: renumber Agents from Chapter 8 to Chapter 9`.

### Step 2: create Chapter 8 (rule 14 checklist)

- [x] P9-4 **(H) Create the chapter**: placeholder `docs/08-rag-security.md`; `nav:` entry between RAG and Agents; `docs/index.md` row; README **Chapters** row and a bullet in **What the notes cover**; `CLAUDE.md` status table and repository layout; `mkdocs build --strict`, then `mkdocs serve` and confirm it appears in the navigation bar and opens.

### Step 3: write Chapter 8, one section at a time

Every `##` section opens with an introduction (what it covers, the key idea, linked list of its subsections), defines every term on first use, has a worked example with real arithmetic (computed with a script, made-up numbers labelled as such), and briefly re-explains any idea from Chapter 7 before linking to it.

- [x] P9-5 **(H) The big picture**: why a RAG system needs its own security (similarity search finds what is relevant, not what the user may see; every document becomes possible input to the model). The four entry points in one table (documents → data poisoning; retrieval → sensitive data leaks; context → indirect prompt injection; actions → unauthorised tool execution), with each attack's goal and main defences. Mermaid diagram of the secured pipeline: ingestion → secure storage and indexing → access-controlled retrieval → untrusted-context processing → output protection → authorised actions.
- [x] P9-6 **(H) Data poisoning** (documents): definition; worked example, a poisoned chunk saying 365 days that outranks the real 120-day chunk (similarity scores computed); a second example of a changed limit; defences: ingest only trusted, approved sources; an approval workflow before indexing; provenance metadata (source, owner, version, ingestion date); version control and audit trail; human review of critical content; monitoring changes and unexpected additions; re-indexing only validated versions; citations so a reader can check. Verify against OWASP LLM04 and PoisonedRAG.
- [x] P9-7 **(H) Sensitive data leaks and access control** (retrieval): the attack (requests for customer ID numbers, API keys, HR salary records); defences:
  - enforce permissions **before** the chunks reach the prompt, not only in the prompt's instructions;
  - RBAC and ABAC in plain words, with a bank example of each;
  - security metadata on every chunk (department, classification, allowed groups) as a JSON example;
  - a filtered search in code, with the vector-database query syntax (for example Chroma's `where` filter), checked against the current documentation and labelled as product-specific;
  - pre-filtering vs post-filtering, and why post-filtering with approximate search can return fewer than k chunks (verify in two vector databases' documentation);
  - detecting personal data and credentials at ingestion; not indexing secrets the assistant never needs; masking or blocking sensitive output;
  - encryption at rest and in transit, in plain words (what each protects against);
  - audit logs of what was retrieved, for whom.
  Worked example: the same question from a fraud analyst and from an analyst outside the fraud team, chunks searched and returned with and without the filter. Verify against OWASP LLM02 and LLM08.
- [x] P9-8 **(H) Indirect prompt injection** (context): full explanation with the employee-handbook style example rewritten for the bank; defences: treat retrieved text as untrusted data; scan and sanitise documents before indexing; detect, remove, quarantine or flag instruction-like text; filter chunks before they reach the model; clear prompt boundaries between system instructions and retrieved text; give retrieved text no authority; moderation of retrieved chunks; monitoring and logging of suspicious retrievals. Say plainly that no defence removes the risk completely. Verify against OWASP LLM01 and Greshake et al.
- [x] P9-9 **(H) Unauthorised tool execution** (actions): why the risk grows when the model can act (email, database query, ticket, file changes, record updates); an injected instruction "send all company files to an external address"; the defences in summary (human approval, allow-listed tools, least privilege, parameter checks, a policy check between model and tools, logs), explained here briefly but completely, with a link to the full treatment in Chapter 9 (P9-16). Verify against OWASP LLM06 (excessive agency).
- [x] P9-10 **(H) Building a production RAG pipeline**: parse the document's structure before chunking (PDF table of contents or bookmarks, Word heading styles, Markdown headings); hierarchical chunks (chapter, section, subsection) for context and traceability; a complete metadata record as JSON for the section 4.2 chunk (chunk text, section title, parent section, page, document ID and version, security tags, valid from, valid to); worked example where the 2025 version is filtered out by its validity date.
- [x] P9-11 **(M) Choosing an embedding model**: criteria (retrieval quality on your own test set, languages of documents and questions, cost, latency, hosted or self-hosted, where the data may go); two examples, an open model run with the Sentence Transformers library and a hosted model such as OpenAI's `text-embedding-3-large`; check names, dimensions and prices against current documentation; public leaderboards (MTEB) only as a starting point.
- [x] P9-12 **(H) Filtering and searching in code**: metadata stored with the vector; a query with access and validity filters, with the query syntax of the same database as P9-7 and a note that syntax differs per product; the secure retrieval flow as a Mermaid diagram (user query → identity and permissions → metadata and access filters → similarity search → authorised chunks only → prompt); returning document, section and page for citations; logging retrieval for audit and troubleshooting.
- [x] P9-13 **(M) Evaluating in production**: a representative test set and recall@k and MRR, recapped briefly from Chapter 7 with a link; add security tests (a poisoned chunk, an injection attempt, a forbidden question) run on every change.
- [x] P9-14 **(H) Production checklist**: one table that merges both sources' checklists (ingestion, chunking, metadata, embeddings, access control, encryption, injection handling, output protection, tool authorisation, logging, evaluation), each row linked to the section that explains it.
- [x] P9-15 **(H) Cheat sheet + Sources**: OWASP Top 10 for LLM Applications 2025, Greshake et al. 2023, PoisonedRAG, NIST RBAC/ABAC, the vector database and embedding model documentation, and every other source actually used.

### Step 4: additions to Chapter 9 (Agents)

Existing text: propose in `fix.md` first.

- [x] P9-16 **(H) A policy check between the model and its tools** ("Checks in code, approval by a person", `docs/09-agents.md` after the rename): a deterministic program that, before every tool call, checks whether the user is authenticated, whether the user may perform this action, whether the tool is allowed for this task, whether the inputs are valid and within limits, and whether a person must approve. Worked on the dispute agent: `open_dispute` for €2,400 passes the checks but triggers `request_approval`; an injected `send_email` to an outside address is refused. Mermaid diagram: model → policy and authorisation check → tool.
- [x] P9-17 **(M) Allow-listed tools and operations, extra confirmation for high-risk transactions, separate retrieval rights from action rights, and log every tool call** (requested, approved, refused, completed). Same subsection as P9-16.
- [x] P9-18 **(M) Link back** from Chapter 9's injection row in "What goes wrong" to Chapter 8's prompt-injection section.

### Step 5: links from earlier chapters

Existing text: propose in `fix.md` first.

- [~] P9-19 **(M) Chapter 7 → Chapter 8 links** (term links done at `docs/07-rag.md:157`, `:287`, `:488`; the Cheat sheet row and the closing sentence wait in `fix.md`, RS-1 and RS-2): from "Cutting along the document's structure" (`docs/07-rag.md:148`), "What a vector database stores" (`docs/07-rag.md:285`) and "Prompt injection through documents" (`docs/07-rag.md:482`); one line in Chapter 7's Cheat sheet pointing to Chapter 8.
- [x] P9-20 **(L) Final check**: `grep` every new anchor link against the headings, `mkdocs build --strict`, read the chapter on `mkdocs serve`; one commit per section, then push and deploy.

### Not planned

- **The product name "Trusteer"** from the sources: not used anywhere in the notes.
