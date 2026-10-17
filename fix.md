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
