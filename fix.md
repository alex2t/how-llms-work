# fix.md — changes to existing content, waiting for approval

Each item below changes text that is already written, so it needs the user's approval first (`CLAUDE.md`, rules 1 and 2). Rewrites, number corrections, layout changes and moved text all go here.

How to use it:

- The user marks an item **Approved**, **Rejected** or **Change: …**.
- Claude applies approved items one at a time, runs `mkdocs build --strict`, then moves the item to "Done" at the bottom with the date.

Legend: `[ ]` waiting · `[x]` approved and applied · `[-]` rejected

---

## Waiting for approval

(Nothing waiting.)

---

## CME295 gap analysis

Topics from Stanford's [CME295](https://cme295.stanford.edu/syllabus/) lectures that the book does not cover yet, one lecture at a time (see `CLAUDE.md`, "CME295 lecture review"). For each item: what is missing, the recommendation (new chapter, or which existing chapter and section), a priority, and a decision line for the user. Approved items become tasks in `plan.md`, Phase 7.

Item IDs: `L<lecture>-<number>`, for example L1-2. Priority: **High** = the book is incomplete without it; **Medium** = a clear improvement; **Low** = nice to have.

### Lecture 1 — Transformer foundations, tokenization, embeddings, attention

Reviewed 2026-10-17. Lecture 1 overlaps heavily with the Introduction and Chapter 1. No new chapter is needed.

**Already covered** (no action):

| Lecture 1 topic | Where the book covers it |
| --- | --- |
| Task-specific era, the 2017 turning point, parallel training | Introduction: Historical evolution; RNNs and LSTMs |
| RNN/LSTM limits: sequential processing, vanishing and exploding gradients | Introduction: Why RNNs hit a wall |
| BPE tokenization; why not plain words or characters | Chapter 1: BPE tokenizer |
| One-hot vectors, Word2Vec (CBOW, skip-gram), polysemy | Introduction: Vector embeddings |
| Query, key and value; Q = XW\_Q and so on | Chapter 1: Where Q, K and V come from |
| Scaled dot-product attention and the reason for √d\_k | Chapter 1: Scaled dot-product attention, step by step |
| Why position is needed (attention ignores order); sinusoidal encoding | Chapter 1: Positional encoding |
| Multi-head attention, Add & Norm, feed-forward network | Chapter 1: Multi-head attention; The full Transformer block |
| Encoder vs decoder, causal mask, cross-attention | Chapter 1; Chapter 2 (masks) |
| Translation walkthrough, one token per step until the end token | Chapter 3: Generation step by step (with "Ti amo molto") |
| GQA, MQA, RoPE, SwiGLU, LoRA, DPO, RLHF, PPO, perplexity, LLM-as-judge, MMLU, time to first token, time per output token | Chapters 2 to 5 |

**Gaps:**

#### L1-1. The context bottleneck of sequence-to-sequence RNNs `[ ]`

- **Missing:** an RNN encoder–decoder squeezes the whole source sentence into **one fixed-size vector** (the last hidden state) before the decoder starts. Long sentences lose information. Attention was first added to RNN translation (Bahdanau et al., 2014) to fix exactly this, by letting the decoder look back at every encoder state. The Introduction mentions 2014 attention in its timeline, but never explains the problem it solved, so the step from RNNs to the Transformer is less clear than it could be.
- **Recommendation:** existing chapter. **Introduction → "Why RNNs hit a wall"**: add a fourth problem, "One fixed-size summary", with a small example (a 30-word sentence squeezed into one vector), and add a row to the bridge table ("decoder reads all encoder states through attention"). Link to Chapter 1's cross-attention.
- **Priority:** High. It is the historical origin of attention.

**Decision:**

#### L1-2. Word, character, subword or byte tokens: the comparison `[ ]`

- **Missing:** Chapter 1 has one paragraph, "Why not just use words or characters?". The lecture compares four granularities (word, character, subword, byte) on vocabulary size, sequence length and unknown words. The book never names **out-of-vocabulary (OOV)** words, and does not cover **byte-level** tokenization (used by GPT-2 and most modern models: 256 base symbols, so any text, including emoji, can be encoded and no token is ever "unknown").
- **Recommendation:** existing chapter. **Chapter 1 → "BPE tokenizer: from text to token IDs"**: expand that paragraph into a subsection "Word, character, subword or byte?" with a comparison table and one worked example (the same word split at each granularity, with the token count for each).
- **Priority:** Medium.

**Decision:**

#### L1-3. Special tokens in one place, including `[UNK]` and chat role tokens `[ ]`

- **Missing:** the book explains `<SOS>`, `<EOS>` and `<pad>` (Chapter 2) and mentions in one sentence that chats use "special tokens that mark who is speaking" (Chapter 3). The **unknown token** `[UNK]` is not covered, and no example shows what a chat looks like as tokens. The lecture also uses other names: `[BOS]` (beginning of sequence) for the book's `<SOS>`, and `[PAD]` for `<pad>`, which readers will meet elsewhere.
- **Recommendation:** existing chapters.
    - **Chapter 1 → BPE tokenizer**: a short "Special tokens" table (token, other common names, purpose, link to where it is explained).
    - **Chapter 3 → "Decoder-only: how chat models generate"**: a worked chat example showing role tokens around a user message and an assistant answer.
- **Priority:** Medium.

**Decision:**

#### L1-4. Other subword algorithms: WordPiece and Unigram `[ ]`

- **Missing:** the book presents BPE only. The lecture mentions Unigram as an alternative; WordPiece (used by BERT) is the other common one.
- **Recommendation:** existing chapter. **Chapter 1 → BPE tokenizer → "Practical consequences"**: a 3- or 4-sentence note with a small table (algorithm, main idea, example models). Model names to be checked against the original papers before writing.
- **Priority:** Low.

**Decision:**

#### L1-5. How Word2Vec is trained: the network and negative sampling `[ ]`

- **Missing:** the Introduction explains Word2Vec's prediction tasks but not the network behind them (vocabulary size → d numbers → vocabulary size, where the word vectors are the weights of the first layer), or **negative sampling** (instead of a softmax over the whole vocabulary, the model answers yes/no for the true neighbour and a few random words, which makes training fast).
- **Recommendation:** existing chapter. **Introduction → "Word2Vec: learning vectors from context"**: one paragraph with a tiny example of one positive and two negative pairs.
- **Priority:** Low.

**Decision:**

#### L1-6. Terms named in Lecture 1 but taught in later lectures `[ ]`

- **Noted, no action now:** Lecture 1 lists acronyms that later lectures teach: MLA, sliding window attention, ALiBi, QKNorm, Mixture of Experts, PEFT, QLoRA, RLVR, GRPO, GSPO, KL, Pass@k, Elo, and the benchmarks AIME, HLE, SWE-bench and GPQA. Each will be reviewed with the lecture that teaches it; all of them go into the glossary after Lecture 9.
- **Recommendation:** none for Lecture 1.

**Decision:**

#### L1-7. Hands-on exercises `[ ]`

- **Not in the book:** the lecture suggests labs (build a BPE tokenizer, attention from scratch, a mini Transformer). The book is a reading book with worked examples, so this is out of its current scope.
- **Recommendation:** optional and not needed for completeness. If wanted later, add a short "Try it yourself" box at the end of each chapter pointing to small exercises.
- **Priority:** Low (your call).

**Decision:**

---

## Done

- 2026-10-17 — F2 — Ch 2, line 46: "about 0.808" corrected to "about 0.807".
- 2026-10-17 — F3 — Ch 2, gradient-descent table: DOG new logit 4.794 corrected to 4.795.
- 2026-10-17 — F4 — Ch 2, label-smoothing table: CAT term 0.239 → 0.240; loss 0.525 → 0.526 (table and text).
- 2026-10-17 — F5 — Ch 2, perplexity paragraph: added a sentence linking to Chapter 5.
- 2026-10-17 — F6 — Ch 2: 23 chapter references turned into links (all anchors checked). Left as plain text on purpose: two table headers (lines 166, 269) and one heading (line 305).
- 2026-10-17 — F7 — Home page: added the logits and fraud running examples.
- 2026-10-17 — F1 — Ch 1, "Training and generation": the two 4-step lists replaced by a 2-bullet summary linking to exact sections of Chapters 2 and 3 (approved wording; links made section-specific once Chapter 3 existed).
- 2026-10-17 — F8 — Ch 3: four links to the Chapter 4 page now point to exact sections (number formats ×2, faster generation, KV-cache quantization), and "grouped-query attention" is now a link to its section. Wording unchanged.
- 2026-10-17 — F9 — Ch 2, 3, 4: three links to the Chapter 5 page now open the perplexity, fraud-example and classification-metrics sections. Wording unchanged.
