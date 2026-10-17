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

Topics the book already covers are checked but not listed. Item IDs: `L<lecture>-<number>`, for example L1-2. Priority: **High** = the book is incomplete without it; **Medium** = a clear improvement; **Low** = nice to have.

### Lecture 1 — Transformer foundations, tokenization, embeddings, attention

Reviewed 2026-10-17. Lecture 1 overlaps heavily with the Introduction and Chapter 1. No new chapter is needed.

#### L1-1. The context bottleneck of sequence-to-sequence RNNs `[ ]`

- **Missing:** an RNN encoder–decoder squeezes the whole source sentence into **one fixed-size vector** (the last hidden state) before the decoder starts. Long sentences lose information. Attention was first added to RNN translation (Bahdanau et al., 2014) to fix exactly this, by letting the decoder look back at every encoder state. The Introduction mentions 2014 attention in its timeline, but never explains the problem it solved, so the step from RNNs to the Transformer is less clear than it could be.
- **Recommendation:** existing chapter. **Introduction → "Why RNNs hit a wall"**: add a fourth problem, "One fixed-size summary", with a small example (a 30-word sentence squeezed into one vector), and add a row to the bridge table ("decoder reads all encoder states through attention"). Link to Chapter 1's cross-attention.
- **Priority:** High. It is the historical origin of attention.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L1-2. Word, character, subword or byte tokens: the comparison `[ ]`

- **Missing:** Chapter 1 has one paragraph, "Why not just use words or characters?". The lecture compares four granularities (word, character, subword, byte) on vocabulary size, sequence length and unknown words. The book never names **out-of-vocabulary (OOV)** words, and does not cover **byte-level** tokenization (used by GPT-2 and most modern models: 256 base symbols, so any text, including emoji, can be encoded and no token is ever "unknown").
- **Recommendation:** existing chapter. **Chapter 1 → "BPE tokenizer: from text to token IDs"**: expand that paragraph into a subsection "Word, character, subword or byte?" with a comparison table and one worked example (the same word split at each granularity, with the token count for each).
- **Priority:** Medium.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L1-3. Special tokens in one place, including `[UNK]` and chat role tokens `[ ]`

- **Missing:** the book explains `<SOS>`, `<EOS>` and `<pad>` (Chapter 2) and mentions in one sentence that chats use "special tokens that mark who is speaking" (Chapter 3). The **unknown token** `[UNK]` is not covered, and no example shows what a chat looks like as tokens. The lecture also uses other names: `[BOS]` (beginning of sequence) for the book's `<SOS>`, and `[PAD]` for `<pad>`, which readers will meet elsewhere.
- **Recommendation:** existing chapters.
    - **Chapter 1 → BPE tokenizer**: a short "Special tokens" table (token, other common names, purpose, link to where it is explained).
    - **Chapter 3 → "Decoder-only: how chat models generate"**: a worked chat example showing role tokens around a user message and an assistant answer.
- **Priority:** Medium.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L1-4. Other subword algorithms: WordPiece and Unigram `[ ]`

- **Missing:** the book presents BPE only. The lecture mentions Unigram as an alternative; WordPiece (used by BERT) is the other common one.
- **Recommendation:** existing chapter. **Chapter 1 → BPE tokenizer → "Practical consequences"**: a 3- or 4-sentence note with a small table (algorithm, main idea, example models). Model names to be checked against the original papers before writing.
- **Priority:** Low.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L1-5. How Word2Vec is trained: the network and negative sampling `[ ]`

- **Missing:** the Introduction explains Word2Vec's prediction tasks but not the network behind them (vocabulary size → d numbers → vocabulary size, where the word vectors are the weights of the first layer), or **negative sampling** (instead of a softmax over the whole vocabulary, the model answers yes/no for the true neighbour and a few random words, which makes training fast).
- **Recommendation:** existing chapter. **Introduction → "Word2Vec: learning vectors from context"**: one paragraph with a tiny example of one positive and two negative pairs.
- **Priority:** Low.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L1-6. Terms named in Lecture 1 but taught in later lectures `[ ]`

- **Noted, no action now:** Lecture 1 lists acronyms that later lectures teach: MLA, sliding window attention, ALiBi, QKNorm, Mixture of Experts, PEFT, QLoRA, RLVR, GRPO, GSPO, KL, Pass@k, Elo, and the benchmarks AIME, HLE, SWE-bench and GPQA. Each will be reviewed with the lecture that teaches it; all of them go into the glossary after Lecture 9.
- **Recommendation:** none for Lecture 1.

**Decision:**

#### L1-7. Hands-on exercises `[ ]`

- **Not in the book:** the lecture suggests labs (build a BPE tokenizer, attention from scratch, a mini Transformer). The book is a reading book with worked examples, so this is out of its current scope.
- **Recommendation:** optional and not needed for completeness. If wanted later, add a short "Try it yourself" box at the end of each chapter pointing to small exercises.
- **Priority:** Low (your call).

**Decision:**

### Lecture 2 — Model families, Mixture of Experts, normalisation, context, decoding, prompting

Reviewed 2026-10-17. Normalisation (RMSNorm, pre-norm), RoPE, attention's quadratic cost, the 0.75-words-per-token rule and temperature, top-k and top-p sampling are already covered. Two topics are large enough to need real space: Mixture of Experts (in Chapter 4) and prompting and reasoning (a new chapter is recommended).

#### L2-1. Mixture of Experts (MoE) `[ ]`

- **Missing:** the book does not mention MoE at all, yet many large modern models use it. An MoE layer replaces the single feed-forward network with several **experts** (smaller feed-forward networks) and a **router** that sends each token to only the top-k of them (for example 2 of 8). The model has many weights in total but uses only a fraction for each token, so it gains knowledge without a matching rise in compute. Also missing: **routing collapse** (the router keeps choosing the same few experts) and the **load-balancing loss** that prevents it; **shared experts** and **fine-grained experts** in recent designs.
- **Recommendation:** existing chapter. **Chapter 4**, a new section "**Mixture of Experts: more weights, same compute**" after "What changed since 2017", plus a row in that section's overview table. Worked example: one token, router scores for 8 experts, softmax, top-2 selection and the weighted sum; then total vs active weights for a real model (figures to be checked against its paper), and why every expert must still sit in memory, which links back to quantization.
- **Priority:** High.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L2-2. Prompting and reasoning: in-context learning, chain-of-thought, self-consistency `[ ]`

- **Missing:** the Introduction mentions in one line that GPT-3 learned "from a few examples in the prompt"; nothing else in the book covers how prompts steer a model. Missing topics:
    - **In-context learning:** zero-shot (instructions only) and few-shot (a few solved examples in the prompt); the weights do not change.
    - **Chain-of-thought (CoT):** asking the model to write intermediate steps before the answer, which improves multi-step maths and logic.
    - **Self-consistency:** sample several chains of thought at a temperature above 0 and take the majority answer, paying more tokens for accuracy.
- **Recommendation:** **new chapter**, "**Chapter 6 — Prompting, reasoning and agents**". Lecture 3 adds reasoning models (`<think>` tokens, test-time compute, GRPO), and later lectures are likely to add agents, tools and retrieval, so the subject will outgrow a section. Placing it **after Chapter 5** avoids renumbering the existing files and links. Worked examples on the running examples: a zero-shot and a few-shot fraud prompt; self-consistency with 5 sampled answers and a majority vote (reusing Chapter 3's sampling), and its cost in tokens.
- **Alternative:** a section in Chapter 3 now, moved to its own chapter later. Not recommended, since moving text needs approval again.
- **Priority:** High.

**Decision:** Approved: new Chapter 6 after Chapter 5 (user, 2026-10-17). Not written yet.

#### L2-3. Constrained (guided) decoding `[ ]`

- **Missing:** forcing the output to follow a format (valid JSON, a fixed list of labels, SQL syntax) by **masking** every token that would break the format at each step, then renormalising. The book's decoding rules (Chapter 3) only choose among all tokens.
- **Recommendation:** existing chapter. **Chapter 3 → "Choosing the next token"**, a new subsection "**Constrained decoding**" after "How the rules combine". Worked example on the fraud prompt: only `FRAUD` and `OK` are allowed, every other token is masked, and the two probabilities are renormalised, the same arithmetic as top-k. Link from Chapter 5's fraud example, where a guaranteed valid label matters.
- **Priority:** Medium.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L2-4. Encoder-only models and masked language modelling (BERT) `[ ]`

- **Missing:** Chapter 1's table lists encoder-only models (BERT) but never says how they are trained. Missing: **masked language modelling** (hide about 15% of the tokens and predict them from both sides), **next sentence prediction**, and the special tokens `[CLS]` (one vector for the whole text) and `[SEP]` (boundary between segments). Encoder-only models remain common for classification and for the text embeddings used in search and retrieval, which later lectures are likely to build on.
- **Recommendation:** existing chapters. **Chapter 2**, a new section "**Another training objective: masked language modelling**" before "Stages of an LLM's life", with a worked example that masks one token of `YOUR CAT IS A LOVELY CAT`. Link to it from **Chapter 1 → "Three ways the block is used today"**.
- **Priority:** Medium.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L2-5. Long context: a bigger window is not a perfect memory `[ ]`

- **Missing:** the book explains the context window as a hard limit (Chapter 3) but not how well models actually use long contexts. Missing: the **needle-in-a-haystack** test (hide one fact at different depths of a long document and ask for it) and the **lost-in-the-middle** finding (facts in the middle of a long prompt are retrieved less reliably than facts near the start or end). Practical consequence: put key instructions and facts at the start or end of a long prompt.
- **Recommendation:** existing chapters. **Chapter 3 → "The context window: a hard limit"**: a short subsection with the practical advice. **Chapter 5 → "Evaluating generative LLMs"**: needle-in-a-haystack as a long-context benchmark. Facts to be checked against the original papers (the summary's "10%–50%" region is not reliable).
- **Priority:** Medium.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L2-6. Extending the context of a trained model (RoPE scaling) `[ ]`

- **Missing:** the lecture notes that RoPE "enables context length extension". Chapter 4 explains RoPE but not how a model trained on, say, 8,192 tokens is stretched to much longer contexts (for example position interpolation, which slows the rotation so longer texts reuse the angles seen in training).
- **Recommendation:** existing chapter. **Chapter 4 → "RoPE: rotary position embedding"**: one paragraph with a small example reusing that section's 0.1-radian rotation.
- **Priority:** Low.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L2-7. Tokens to words to pages `[ ]`

- **Missing:** Chapter 1 gives the rule "about 0.75 English words per token". The lecture adds a scale that makes context sizes concrete: 1,000 tokens ≈ 750 words ≈ 1.5 pages; 100,000 tokens ≈ a novel.
- **Recommendation:** existing chapter. **Chapter 3 → "The context window: a hard limit"**: a 4-row table converting common window sizes (8,192, 128,000, 1,000,000 tokens) into words and pages, computed rather than copied.
- **Priority:** Low.

**Decision:** Approved (user, 2026-10-17). Not written yet.

### Lecture 3 — Pre-training data, SFT, LoRA, RLHF vs DPO, GRPO, distillation

Reviewed 2026-10-17. The next-token objective, teacher forcing, and supervised fine-tuning with the loss on answer tokens only are already covered (Chapter 2). RLHF, DPO and LoRA are each covered by one short paragraph; the lecture goes much deeper. Most gaps belong in Chapter 2; reasoning models belong in the new Chapter 6.

#### L3-1. Where pre-training data comes from `[x]`

- **Missing:** Chapter 2 says only that the model learns from "books, web pages, code". Missing: how a pre-training corpus is built (web crawls such as Common Crawl, code repositories, text extracted from PDFs), how it is cleaned (rule-based filters, **deduplication**, language identification, quality classifiers), training in **stages** (broad data first, then more code and reasoning data), and how the **data mixture** (the share of each source) is chosen by training small proxy models on different mixtures.
- **Recommendation:** existing chapter. **Chapter 2 → "Training data and batches"**: expand "Training data" into a section "**Where pre-training data comes from**", with a pipeline diagram (Mermaid) and a worked example of how deduplication and filtering shrink a corpus (made-up numbers). Real corpus sizes to be checked against the papers.
- **Priority:** Medium.

**Decision:** Approved (user, 2026-10-17). Written 2026-10-17.

#### L3-2. Model collapse: training on AI-generated text `[x]`

- **Missing:** more and more web text is written by LLMs. Models trained on it can lose the rare cases (the "tails" of the distribution) and become less varied from one generation of models to the next.
- **Recommendation:** existing chapter. **Chapter 2 → "Where pre-training data comes from"** (L3-1): a short subsection with a small illustration of how rare tokens disappear when a model's own samples become the next training set.
- **Priority:** Low.

**Decision:** Approved (user, 2026-10-17). Written 2026-10-17.

#### L3-3. Bits per byte: a loss that does not depend on the tokenizer `[ ]`

- **Missing:** Chapter 5 shows that perplexity can only be compared between models with the same tokenizer, but gives no fix. **Bits per byte (BPB)** divides the total loss by the number of bytes of text instead of the number of tokens (and converts it to bits), so models with different tokenizers can be compared fairly.
- **Recommendation:** existing chapter. **Chapter 5 → "Perplexity" → "Only comparable with the same tokenizer"**: add BPB right after the two-tokenizer example. Reusing that example (total loss 3.0 for the same sentence), both models get the same BPB: if the sentence is 30 bytes long, 3.0 ÷ ln 2 ÷ 30 ≈ 0.144 bits per byte, which shows exactly the fairness that perplexity lacked.
- **Priority:** Medium.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L3-4. Packing sequences instead of padding them `[ ]`

- **Missing:** Chapter 2 explains padding, which wastes compute on `<pad>` tokens. Training pipelines usually **pack** several short examples into one fixed-length sequence, separated by end and start tokens, so almost no positions are wasted.
- **Recommendation:** existing chapter. **Chapter 2 → "Padding and the padding mask"**: a final subsection "**Packing: the alternative to padding**", reusing sentences A (6 tokens) and B (4 tokens): padded, the batch wastes 2 of 12 positions; packed into one sequence, it wastes none. Mention that attention must still be kept from crossing between packed examples.
- **Priority:** Low.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L3-5. LoRA in depth, and the term "instruction tuning" `[ ]`

- **Missing:** Chapter 2 gives LoRA one paragraph (frozen weights, a 512 × 8 and an 8 × 512 matrix). Missing: the formula W = W₀ + B·A with every letter explained; **where** adapters are placed (the original paper: W\_Q and W\_V only; common practice today: all linear layers); **swapping adapters** so one base model serves many tasks; **QLoRA** (LoRA on a 4-bit quantized base model, linking to Chapter 4); and a one-line mention of other parameter-efficient methods (adapters, prefix tuning). Also, Stage 2 never uses the common name **instruction tuning** for supervised fine-tuning on instruction-following examples.
- **Recommendation:** existing chapter. **Chapter 2 → "Stages of an LLM's life"**: turn the LoRA paragraph into a subsection "**Cheap fine-tuning: LoRA**" with a worked example (a 2 × 2 weight matrix plus B·A with rank 1). Add "instruction tuning" to the Stage 2 text.
- **Priority:** Medium.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L3-6. Preference training in detail: reward model, KL penalty, DPO `[ ]`

- **Missing:** Stage 3 in Chapter 2 describes RLHF and DPO in words only. Missing:
    - **Reward model training** (the Bradley–Terry loss): the loss is small when the preferred answer gets the higher score. Example: scores 2.0 (preferred) and 0.5 (rejected) give σ(1.5) = 0.818 and a loss of −ln 0.818 = 0.201.
    - **The KL penalty** in RLHF: a term that keeps the trained model close to its starting point, to prevent **reward hacking** (the model exploiting flaws in the reward model).
    - **The DPO loss** in plain words: raise the probability of the preferred answer, relative to the starting model, more than that of the rejected one.
    - **Cost comparison:** RLHF with PPO keeps 4 models in memory (the model being trained, its frozen reference copy, the reward model, a value model); DPO needs 2.
- **Recommendation:** existing chapter. **Chapter 2 → "Stage 3: preference training"**: expand with the worked reward-model example, the KL idea, the DPO loss with every letter explained, and a 4-models-vs-2 table.
- **Priority:** Medium.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L3-7. Reasoning models and how they are trained (test-time compute, GRPO) `[ ]`

- **Missing:** none of this is in the book:
    - **Reasoning models** write a hidden chain of thought between `<think>` and `</think>` before the final answer. Spending more output tokens at answer time to get better answers is called **test-time compute**.
    - **Reinforcement learning with verifiable rewards (RLVR):** for maths and code, an answer can be checked automatically (is the number right, do the tests pass?), so no human labels or reward model are needed.
    - **GRPO** (group relative policy optimisation, used for DeepSeek-R1): for each question the model writes a group of answers; each answer's **advantage** is its reward minus the group's average, divided by the group's spread. Example: rewards 1, 0, 1, 1 give a mean of 0.75, a spread of 0.433, and advantages of +0.577 for the right answers and −1.732 for the wrong one. No value model is needed, unlike PPO.
    - **What emerges:** longer answers, self-checking and corrections appear during training without being taught directly.
- **Recommendation:** **new Chapter 6** (approved in L2-2), section "**Reasoning models: thinking before answering**", after chain-of-thought and self-consistency. Link from Chapter 2's Stage 3 and from Chapter 3 (thinking tokens make answers longer, so they cost decode time and KV-cache memory).
- **Priority:** High.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L3-8. Distillation: a small model learns from a large one `[ ]`

- **Missing:** **knowledge distillation**: a small **student** model is trained to match the probabilities of a large **teacher** model, instead of one-hot targets. Variants from the lecture: distilling a teacher's chains of thought into a smaller model, and **on-policy distillation** (the teacher scores answers that the student itself wrote).
- **Recommendation:** existing chapter. **Chapter 2**, a new section "**Distillation**" after "Stages of an LLM's life". Worked example reusing Chapter 1's probabilities as the teacher's soft target (CAT 0.772, DOG 0.211, IS 0.013, LOVELY 0.004) and comparing the student's loss with the one-hot loss, which connects to label smoothing. Link from **Chapter 4** as another way to get a smaller, cheaper model.
- **Priority:** Medium.

**Decision:** Approved (user, 2026-10-17). Not written yet.

#### L3-9. Benchmarks for maths, code and preferences; Elo and pass@k `[ ]`

- **Missing:** Chapter 5 names MMLU, HumanEval and Chatbot Arena only. Missing: **GSM8K**, **MATH** and **AIME** (maths), **SWE-bench** (fixing real software issues); how Chatbot Arena turns votes into an **Elo rating** (a 100-point gap means the stronger model is expected to win about 64% of comparisons: 1 ÷ (1 + 10^(−100/400)) = 0.640); and **pass@k** (the chance that at least one of k attempts is correct), the usual score for code. Pass@k was noted under L1-6.
- **Recommendation:** existing chapter. **Chapter 5 → "Evaluating generative LLMs" → "Benchmarks"**: a table of benchmark families (what each measures, how it is scored) and two short worked examples (Elo, pass@k). Benchmark sizes to be checked against their papers.
- **Priority:** Medium.

**Decision:** Approved (user, 2026-10-17). Not written yet.

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
