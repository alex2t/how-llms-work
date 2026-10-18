# fix.md — changes to existing content, waiting for approval

Each item below changes text that is already written, so it needs the user's approval first (`CLAUDE.md`, rules 1 and 2). Rewrites, number corrections, layout changes and moved text all go here.

How to use it:

- The user marks an item **Approved**, **Rejected** or **Change: …**.
- Claude applies approved items one at a time, runs `mkdocs build --strict`, then **removes** the item (or the row, in a table of issues) from this file. Git history keeps the record.
- Every list of issues is a **numbered table** (first column `#`, for example S2-1, S2-2), so the list can be worked down in order. Numbers are never reused or renumbered.

Legend: `[ ]` waiting · `[x]` approved and applied · `[-]` rejected

---

## Waiting for approval

### S-2. Links that replace an explanation `[ ]`

- **Rule:** `CLAUDE.md`, Writing style: "A link never replaces an explanation." The reader must understand the passage without clicking; a link to another chapter only shows where else the topic is discussed.
- **Checked:** all 189 links between chapters in the text of Chapters 0–6 (Sources and Cheat sheets left out). Most already explain the term next to the link, or only point forward to where a topic is developed ("Chapter 3 covers generation in detail"); those are fine.
- **Below:** the places still to fix, where a term is named with a link but not explained in the sentence, so a reader who does not click would not understand it.
- **Proposed change:** for each one, add a short definition (one or two sentences, with a small example where useful) next to the link, and keep the link. Nothing is removed.

| # | Location | Section | Term used without explanation | What to add | Priority |
| --- | --- | --- | --- | --- | --- |
| S2-1 | `docs/01-transformers.md:817` | Training and generation | **cross-entropy**, **Adam optimiser** | Cross-entropy: −ln of the probability given to the correct token. Adam: the rule that turns gradients into weight changes, with a separate step size for every weight | Low |
| S2-2 | `docs/02-training.md:797` | Another training objective: masked language modelling | **encoder-only models** | A model made only of the encoder stack: it reads a whole text at once (both directions) and produces vectors, but does not generate text | Low |
| S2-3 | `docs/03-inference.md:43` | Inference vs training › Definitions | **E, W\_Q, W\_K, W\_V, W\_O, feed-forward layers** (names only) | One short phrase per matrix, as in `docs/02-training.md:43` | Medium |
| S2-4 | `docs/03-inference.md:72` | Why inference needs less memory | Adam's **m** and **v** | m: running average of each weight's recent gradients (the direction); v: running average of their squares (how large and noisy they are) | Medium |
| S2-5 | `docs/03-inference.md:293` | What this means in practice | **quantization** (first use, before Chapter 4) | Storing each weight in fewer bits, for example 1 byte instead of 2, so there is less to read at each step | Low |
| S2-6 | `docs/03-inference.md:645` | Constrained decoding | **threshold** | The cut-off score above which the application answers `FRAUD` (0.5 in Chapter 5) | Low |
| S2-7 | `docs/04-quantization.md:382` | How to check a quantized model | **perplexity** | e raised to the average loss; read as "how many tokens the model hesitates between"; lower is better | Medium |
| S2-8 | `docs/04-quantization.md:434` | Decoder-only | "Chapter 1 described **both stacks**" | Name the two stacks: the encoder reads the input, the decoder generates the output one token at a time | Low |

**Decision:** S2-1 to S2-8 waiting.

---

### L1-7. Hands-on exercises `[ ]`

- **Not in the book:** the lecture suggests labs (build a BPE tokenizer, attention from scratch, a mini Transformer). The book is a reading book with worked examples, so this is out of its current scope.
- **Recommendation:** optional and not needed for completeness. If wanted later, add a short "Try it yourself" box at the end of each chapter pointing to small exercises.
- **Priority:** Low (your call).

**Decision:**

---

## CME295 gap analysis

Topics from Stanford's [CME295](https://cme295.stanford.edu/syllabus/) lectures that the book does not cover yet, one lecture at a time (see `CLAUDE.md`, "CME295 lecture review"). For each item: what is missing, the recommendation (new chapter, or which existing chapter and section), a priority, and a decision line for the user. Approved items become tasks in `plan.md`, Phase 7.

Topics the book already covers are checked but not listed. Item IDs: `L<lecture>-<number>`, for example L1-2. Priority: **High** = the book is incomplete without it; **Medium** = a clear improvement; **Low** = nice to have.

### Lecture 1 — Transformer foundations, tokenization, embeddings, attention

Reviewed 2026-10-17. Lecture 1 overlaps heavily with the Introduction and Chapter 1. No new chapter is needed.

#### L1-6. Terms named in Lecture 1 but taught in later lectures `[ ]`

- **Noted, no action now:** Lecture 1 lists acronyms that later lectures teach: MLA, sliding window attention, ALiBi, QKNorm, Mixture of Experts, PEFT, QLoRA, RLVR, GRPO, GSPO, KL, Pass@k, Elo, and the benchmarks AIME, HLE, SWE-bench and GPQA. Each will be reviewed with the lecture that teaches it; all of them go into the glossary after Lecture 9.
- **Recommendation:** none for Lecture 1.

**Decision:**

#### L1-7. Hands-on exercises: moved to the top of this file, under "Waiting for approval".

### Lecture 2 — Model families, Mixture of Experts, normalisation, context, decoding, prompting

Reviewed 2026-10-17. Normalisation (RMSNorm, pre-norm), RoPE, attention's quadratic cost, the 0.75-words-per-token rule and temperature, top-k and top-p sampling are already covered. Two topics are large enough to need real space: Mixture of Experts (in Chapter 4) and prompting and reasoning (a new chapter is recommended).

No open items: every gap found in this lecture has been written.

### Lecture 3 — Pre-training data, SFT, LoRA, RLHF vs DPO, GRPO, distillation

Reviewed 2026-10-17. The next-token objective, teacher forcing, and supervised fine-tuning with the loss on answer tokens only are already covered (Chapter 2). RLHF, DPO and LoRA are each covered by one short paragraph; the lecture goes much deeper. Most gaps belong in Chapter 2; reasoning models belong in the new Chapter 6.

No open items: every gap found in this lecture has been written.
