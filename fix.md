# fix.md — changes to existing content, waiting for approval

Each item below changes text that is already written, so it needs the user's approval first (`CLAUDE.md`, rules 1 and 2). Rewrites, number corrections, layout changes and moved text all go here.

How to use it:

- The user marks an item **Approved**, **Rejected** or **Change: …**.
- Claude applies approved items one at a time, runs `mkdocs build --strict`, then moves the item to "Done" at the bottom with the date.

Legend: `[ ]` waiting · `[x]` approved and applied · `[-]` rejected

---

## Chapter 1 — How Transformers Work

### F1. Shorten "Training and generation" into a summary with links `[ ]` (on hold)

**Where:** `docs/01-transformers.md`, section `## Training and generation` (line 518).

**Why:** Chapter 2 now explains training in full, and Chapter 3 will explain generation. The two numbered lists in Chapter 1 repeat that content in less detail.

**Will be removed:** the `### Training` heading and its 4-step list, and the `### Generation` heading and its 4-step list.

**Kept:** the opening sentence and the key idea "where do the Q, K, V weights come from".

**Proposed new text:**

```markdown
## Training and generation

Every weight in the model, including W\_Q, W\_K, W\_V, W\_O, the embeddings and the feed-forward layers, is learned the same way: predict the next token, measure the error, nudge all weights to reduce it. This is the answer to "where do the Q, K, V weights come from": they are whatever values made next-token prediction most accurate over the training data.

- **Training** feeds the whole text in at once. The causal mask hides each position's answer, so every position is trained in parallel in one pass. The loss is cross-entropy, and the Adam optimiser adjusts every weight slightly after each batch. [Chapter 2](02-training.md) walks through each step with numbers.
- **Generation** runs the model once per new token: pick the next token from the probabilities, append it to the input, and repeat until an end-of-sequence token or a length limit. The KV cache stores the K and V of earlier tokens so they are not recomputed. [Chapter 3](03-inference.md) covers this in detail.
```

**Decision:** On hold until Chapter 3 is written (user, 2026-10-17).

---

## Done

- 2026-10-17 — F2 — Ch 2, line 46: "about 0.808" corrected to "about 0.807".
- 2026-10-17 — F3 — Ch 2, gradient-descent table: DOG new logit 4.794 corrected to 4.795.
- 2026-10-17 — F4 — Ch 2, label-smoothing table: CAT term 0.239 → 0.240; loss 0.525 → 0.526 (table and text).
- 2026-10-17 — F5 — Ch 2, perplexity paragraph: added a sentence linking to Chapter 5.
- 2026-10-17 — F6 — Ch 2: 23 chapter references turned into links (all anchors checked). Left as plain text on purpose: two table headers (lines 166, 269) and one heading (line 305).
- 2026-10-17 — F7 — Home page: added the logits and fraud running examples.
