# fix.md — changes to existing content, waiting for approval

Each item below changes text that is already written, so it needs the user's approval first (`CLAUDE.md`, rules 1 and 2). Rewrites, number corrections, layout changes and moved text all go here.

How to use it:

- The user marks an item **Approved**, **Rejected** or **Change: …**.
- Claude applies approved items one at a time, runs `mkdocs build --strict`, then **removes** the item (or the row, in a table of issues) from this file. Git history keeps the record.
- Every list of issues is a **numbered table** (first column `#`, for example S2-1, S2-2), so the list can be worked down in order. Numbers are never reused or renumbered.

Legend: `[ ]` waiting · `[x]` approved and applied · `[-]` rejected

---

## Waiting for approval

### FT. New Chapter 13 (Fine-tuning): follow-up changes in other chapters `[ ]`

Chapter 13 is written. These changes touch existing text, so they wait for approval.

| # | File and place | Proposed change | Why | Priority |
| --- | --- | --- | --- | --- |
| FT-2 | `docs/07-rag.md:679`, end of "When to use which" | Add one sentence linking to `13-fine-tuning.md#when-to-fine-tune-the-decision-order` | The decision is now treated in full in Chapter 13 | Low |
| FT-3 | `docs/02-training.md`, "Cheap fine-tuning: LoRA" (line 1007) | Add one sentence linking to `13-fine-tuning.md#modern-fine-tuning-techniques` | Points the reader from how LoRA works to when to use it | Low |
| FT-4 | This file, row C8-22 | Remove the row: "What fine-tuning costs" (catastrophic forgetting, labelled data, being overtaken by a newer model) is now covered in Chapter 13 | Avoids explaining it twice | Low |

### RS. New Chapter 8 (RAG security and production): follow-up changes outside Chapters 8 and 9 `[ ]`

Chapter 8 is written and Chapter 9 has its new subsection "A policy check before every tool call". Chapter 7 received only term links to Chapter 8 (lines 157, 287 and 488). These changes touch other files or existing text, so they wait for approval.

| # | File and place | Proposed change | Why | Priority |
| --- | --- | --- | --- | --- |
| RS-1 | `docs/07-rag.md`, Cheat sheet (line 710) | Add a row: "RAG security \| Poisoning, access filters, prompt injection, tool misuse: see Chapter 8", linked to `08-rag-security.md#the-big-picture` | A reader of Chapter 7's summary should know the security topics continue in Chapter 8 | Medium |
| RS-2 | `docs/07-rag.md`, "Prompt injection through documents" (line 482) | Add one closing sentence: "Chapter 8 explains the attack in full, with ingestion scanning, output checks and security tests." (D7 in `plan.md`) | Today only the word "Defences" links to Chapter 8 | Low |
| RS-3 | `docs/06-prompting.md:31` | "Two later chapters build on these techniques" could mention Chapter 8 in one clause: "and Chapter 8 shows how to secure such a system" | The sentence lists Chapters 7 and 9 only; still correct, but skips Chapter 8 | Low |
| RS-4 | `requirements.txt` and the build banner | `mkdocs-redirects` 1.2.3 (added for D8) depends on the `properdocs` package, which prints a large "MkDocs 2" warning at every build. The build still passes. Options: (a) pin `mkdocs-redirects==1.2.2` if that version has no such dependency (check first); (b) set `DISABLE_MKDOCS_2_WARNING=true` in the deploy workflow; (c) leave it | Keeps the build log readable; the banner may also worry a reader of the GitHub Actions log | Medium |
| RS-6 | `README.md`, Credits (line 54) | Credit the OWASP Top 10 for LLM Applications 2025 as the reference list of risks behind Chapter 8 | Chapter 8 is organised around it; the README credits other main sources | Low |

### C8. Stanford CS230 Lecture 8: additions to Chapters 2, 3 and 5 `[ ]`

Approved in principle (`plan.md`, Phase 8); to be done next session. Rows C8-19 and C8-20 change existing text, so the exact wording is shown to the user before it is applied.

| # | Chapter and place | What to add | Priority |
| --- | --- | --- | --- |
| C8-19 | Chapter 3, "A bigger window is not a perfect memory" (`docs/03-inference.md:711`), existing text | Advice for long conversations: summarise earlier turns and start a fresh conversation. The lecture's "about 8 turns" is unverified; use a number only if a source supports it. | Medium |
| C8-20 | Chapter 5, "LLM-as-judge" (`docs/05-evaluation.md:494`), existing text | Table of the three judge set-ups: pairwise comparison, single-answer grading, reference-guided grading (Zheng et al., 2023, MT-Bench). | Medium |
| C8-21 | Chapter 5, new subsection "Testing prompts automatically", after "LLM-as-judge" | A fixed test set run on every prompt change, with code checks plus an LLM-as-judge; PromptFoo as an example tool. | Medium |
| C8-22 | Chapter 2, Stage 2 (SFT), new subsection "What fine-tuning costs" | Catastrophic forgetting (definition and example), the need for labelled data, the risk that a newer base model overtakes the fine-tuned one; link to Chapter 7's decision table. | Medium |

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


### Lecture 2 — Model families, Mixture of Experts, normalisation, context, decoding, prompting

Reviewed 2026-10-17. Normalisation (RMSNorm, pre-norm), RoPE, attention's quadratic cost, the 0.75-words-per-token rule and temperature, top-k and top-p sampling are already covered. Two topics are large enough to need real space: Mixture of Experts (in Chapter 4) and prompting and reasoning (a new chapter is recommended).

No open items: every gap found in this lecture has been written.

### Lecture 3 — Pre-training data, SFT, LoRA, RLHF vs DPO, GRPO, distillation

Reviewed 2026-10-17. The next-token objective, teacher forcing, and supervised fine-tuning with the loss on answer tokens only are already covered (Chapter 2). RLHF, DPO and LoRA are each covered by one short paragraph; the lecture goes much deeper. Most gaps belong in Chapter 2; reasoning models belong in the new Chapter 6.

No open items: every gap found in this lecture has been written.
