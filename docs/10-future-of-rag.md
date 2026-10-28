# The future of RAG

## The big picture

Each time a model with a larger context window is released, or a popular agent turns out not to use a vector database, commentators announce that RAG is dead. Others answer that "agentic RAG" is the future. Each side quotes sources. This chapter takes the three main claims one at a time, sets out what each one rests on, and compares it with the published evidence, using the bank's numbers. It matters because the bank, like any company, must decide where to invest, and because the right answer depends on the question, the collection and the rules around it, not on the latest announcement.

**The running example of this chapter.** The bank's fraud-policy assistant of [Chapter 7](07-rag.md#the-big-picture), [Chapter 8](08-rag-security.md#the-big-picture) and [Chapter 9](09-advanced-rag.md#the-big-picture) is in production. It answers analysts' questions from the 120-page fraud-policy manual: 80,000 tokens, cut into 229 chunks of 400 tokens. For each question it finds the best chunks with hybrid search (vector search plus keyword search), re-ranks them, and applies access-control filters, so that a user only sees chunks they are allowed to read. On Chapter 7's small test set it reaches recall@3 = 0.8: for 4 questions out of 5, the right chunk is among the top 3.

The bank's management has read two things. One is that million-token context windows make RAG unnecessary. The other is that "agentic RAG is the future". They want to know whether to keep investing in retrieval.

This chapter covers:

1. [Three claims](#three-claims): what is being argued, and by whom.
2. [Claim 1: long context windows will make RAG unnecessary](#claim-1-long-context-windows-will-make-rag-unnecessary): the needle test, harder benchmarks, routing, and the cost.
3. [Claim 2: agentic search replaces RAG](#claim-2-agentic-search-replaces-rag): what coding assistants actually did, and why.
4. [Claim 3: agentic RAG is the future](#claim-3-agentic-rag-is-the-future): what the gains are, and what they cost.
5. [What the evidence says](#what-the-evidence-says): a verdict per claim, and what does not change.
6. [From RAG to context engineering](#from-rag-to-context-engineering): the direction the field is taking.

## Three claims

| Claim | What it rests on |
| --- | --- |
| **1. Long context windows will make RAG unnecessary** | Models that read a million tokens or more, and near-perfect scores on "needle in a haystack" tests |
| **2. Agentic search replaces RAG** | Coding assistants such as Claude Code, which dropped their vector database and let the model search files with tools |
| **3. Agentic RAG is the future** | Surveys and research systems in which a model plans its own searches, often over several steps or with several agents |

!!! tip "Key idea: retrieval-augmented generation never meant 'vector search'"
    The original RAG paper ([Lewis and colleagues, 2020](07-rag.md#the-big-picture)) used a dense vector retriever. Its idea, however, is more general: **retrieve** relevant information from outside the model, **then generate** from it. Under that definition, all of these are retrievers:

    - BM25 keyword search;
    - a SQL query on a transactions table;
    - `grep` over a folder of files;
    - a web search;
    - a vector index.

    Much of the "RAG is dead" debate is really about two narrower questions:

    1. **Who decides what to retrieve and when**: a pipeline written by the developer, or the model itself in a loop?
    2. **How much retrieved text to put in the prompt**: a few chunks, or a whole document?

    Keeping these two questions apart makes the claims below much easier to judge.

## Claim 1: long context windows will make RAG unnecessary

**What it rests on.** In February 2024, Google introduced Gemini 1.5 Pro with a context window of up to 1 million tokens in production. In research it was tested up to 10 million. On the **needle-in-a-haystack** test, a single fact (the "needle") is hidden in a long text (the "haystack") and the model is asked to find it. Gemini 1.5 Pro found the needle 99% of the time in texts of up to a million tokens. The bank's manual is 80,000 tokens, so why search it at all? Why not put all of it in every prompt?

**What harder benchmarks show.** The needle test is the easiest possible case: one fact, worded almost like the question. Three later studies made the test more realistic:

- **RULER** (Hsieh and colleagues, NVIDIA, 2024) evaluated 17 long-context models on 13 tasks, including:
  - several needles at once;
  - **multi-hop tracing**, following a chain of references through the text;
  - **aggregation**, counting or collecting items spread across the text.

  The models scored almost perfectly on the plain needle test, yet all of them lost accuracy as the text grew. Although all 17 claimed context windows of 32,000 tokens or more, only half kept satisfactory performance at 32,000 tokens.
- **NoLiMa** (Modarressi and colleagues, 2025, published at ICML 2025) removed the **literal matches** between question and needle. The question no longer shares words with the answer, so the model has to make an association. For example, it must connect a question about someone who has been to Dresden with a needle about the Semperoper, the city's opera house.

  The paper evaluated 12 models (13 in the published version) that claim at least 128,000 tokens. At 32,000 tokens, 10 of them (11 in the published version) fell below half of their short-text score. GPT-4o, one of the best, dropped from 99.3% to 69.7%. The authors attribute the decline to the attention mechanism finding it harder to locate relevant tokens in long texts when there are no literal matches to latch onto.
- **Context Rot** (Hong, Troynikov and Huber, Chroma, 2025) tested 18 models, including GPT-4.1, Claude 4, Gemini 2.5 and Qwen3, on deliberately simple tasks while only the input length grew. Performance became less reliable as the input grew, in uneven ways:
  - distracting passages that resemble the answer hurt;
  - the less the question resembles the answer, the faster accuracy falls.

  On a conversation-memory benchmark (LongMemEval), the models did clearly better with a focused input of about 300 tokens than with the full history of about 113,000 tokens, although both contained the answer.

The bank's analysts rarely copy the manual's wording. *"How long does a cardholder have to contest a charge?"* shares almost no words with section 4.2. That is exactly the setting where NoLiMa and Context Rot found the steepest drops. This builds on [lost in the middle](07-rag.md#ordering-the-chunks): it is not only *where* the fact sits that matters, but *how much* else surrounds it.

**What direct comparisons show.** Two studies compared RAG and long context on the same questions:

- **Li and colleagues** (Google DeepMind and University of Michigan, 2024, published at EMNLP 2024) found that, when resourced sufficiently, long context beat RAG on average quality. RAG's much lower cost remained a clear advantage.

  They proposed **Self-Route**:
  1. Every question first goes through RAG, and the model is asked whether the retrieved chunks are enough to answer.
  2. Only the questions it judges unanswerable are sent again with the full long context.

  Self-Route kept quality comparable to long context while cutting cost by 65% for Gemini-1.5-Pro and 39% for GPT-4o.
- **LaRA** (Li and colleagues, Alibaba and Pennsylvania State University, 2025, published at ICML 2025) built 2,326 test cases across four kinds of question and three kinds of long text, and tested 11 models. Its title gives the conclusion: there is *no silver bullet*. The better choice depends on:

  - how capable the model is (RAG helps weaker models more);
  - the length of the text (RAG's advantage grows as texts get longer);
  - the type of question;
  - the quality of the retrieval.

**Worked example: the bank's monthly bill.** Use Chapter 7's assumptions: 20,000 questions a month, input at $3 per million tokens, a manual of 80,000 tokens, and 300 tokens of instructions and question. Suppose that with Self-Route, 15% of questions are routed to the full manual (made up):

| Approach | Input tokens per question | Monthly input cost |
| --- | --- | --- |
| RAG, top 5 chunks ([Chapter 7](07-rag.md#worked-comparison-the-whole-manual-vs-five-chunks)) | 2,300 | **$138** |
| Whole manual with prompt caching (cache reads at one tenth of the price; an assumed, typical discount) | 80,300, of which 80,000 cached | **$498** |
| Self-Route (all questions through RAG, 15% also through the whole manual) | 2,300 + 0.15 × 80,300 = 14,345 on average | **$861** |
| Agentic search, 5 model calls (see [Claim 2](#claim-2-agentic-search-replaces-rag)) | 27,500 | **$1,650** |
| Whole manual, no caching | 80,300 | **$4,818** |

!!! note "Quick refresher: how the Self-Route and caching costs are calculated"
    **Self-Route.** All 20,000 questions pay for the RAG step, and the 15% that are routed (3,000 questions) also pay for the full manual:

    $$
    20{,}000 \times 2{,}300 + 3{,}000 \times 80{,}300 = 46{,}000{,}000 + 240{,}900{,}000 = 286{,}900{,}000 \text{ tokens}
    $$

    At $3 per million tokens, that is 286.9 × $3 = **$860.70**: 82% less than the uncached long context ($4,818), but about 6 times more than plain RAG.

    **Prompt caching.** If the 80,000 tokens of the manual are cached ([Chapter 7](07-rag.md#worked-comparison-the-whole-manual-vs-five-chunks)) and cache reads cost $0.30 per million, one question costs:

    - 80,000 × $0.30 ÷ 1,000,000 = $0.024 for the cached manual;
    - 300 × $3 ÷ 1,000,000 = $0.0009 for the rest.

    That is about **$0.025** per question, or **$498** for 20,000 questions. This simplified figure ignores the cost of writing the cache and of refreshing it when it expires, which providers charge separately.

Prompt caching closes most of the **cost** gap for a collection the size of the manual. It does nothing for three other problems:

- **Quality.** The model still reads 80,000 tokens, with the risks RULER, NoLiMa and Context Rot measured.
- **Access rights.** A customer-service agent would receive the 31 fraud-team-only chunks in every prompt ([Chapter 8](08-rag-security.md#why-the-search-ignores-access-rights)). Avoiding that means a different cached prompt per access group.
- **Scale.** The bank's full collection of 10 million chunks fits in no context window.

## Claim 2: agentic search replaces RAG

**What it rests on.** Boris Cherny, who created Anthropic's coding assistant Claude Code, has explained publicly (in podcast interviews in 2025, and in posts since) how its search evolved:

- Early versions used RAG with a local vector database.
- The team found that **agentic search** worked better: letting the model search the code itself with simple tools such as `glob` (find files by name pattern) and `grep` (find lines containing a word), then read the files it chooses.
- They also tried more elaborate approaches, such as model-based indexing of the code, before settling on this.

The reasons given were:

- **Simplicity:** there is no index to build or maintain.
- **Freshness:** a file edited a second ago is searched as it is now, not as it was at the last re-indexing.
- **Security and privacy:** no copy of the code is stored in an index.
- **Reliability:** there are no stale or missing embeddings.

Anthropic's engineering guide *Effective context engineering for AI agents* (September 2025) generalises the idea as **just-in-time** retrieval. The agent keeps lightweight references, such as file paths and names, and loads content only when it needs it. The guide also describes a **hybrid** strategy: some context is placed in the prompt up front (Claude Code loads its project instruction files), and the rest is explored with tools.

**Why it works for code.** Several properties of a code repository favour agentic search:

- **Exact identifiers.** A question about `block_if_score_above` contains the exact name to search for, so `grep` acts as a perfect keyword retriever.
- **Organisation.** Code is organised in folders and files whose names are meaningful.
- **Change.** The files change constantly, so any index would be out of date.
- **Size and ownership.** The collection is local, owned by the user, and usually small enough to explore.
- **Tolerance for slow answers.** A developer accepts a task that takes a minute and a dozen model calls.

**Why the bank's assistant is different.** The fraud-policy assistant has the opposite profile:

- **Size.** Millions of chunks of prose across many systems.
- **Vocabulary.** Questions in everyday words rather than exact identifiers ("contest a charge").
- **Access rights per user.** `grep` knows nothing about rights unless the tool enforces them.
- **Volume and speed.** Thousands of simple questions a day, each expected in a second or two.

The cost alone differs by an order of magnitude. Apply [Chapter 11's formula](11-agents.md#working-memory-the-context-window) for the tokens of an agent run. Take N = 5 model calls, a starting prompt of B = 1,500 tokens, and s = 2,000 tokens added per step (each search returns about five chunks):

$$
N \cdot B + s \cdot \frac{N(N - 1)}{2} = 5 \times 1{,}500 + 2{,}000 \times 10 = 27{,}500 \text{ tokens}
$$

That is about **12 times** the 2,300 tokens of one RAG call: **$1,650** a month instead of $138, before the waiting time of five calls in a row.

The bank's version of agentic search is therefore not `grep` over raw files. It is an agent whose search **tool** is the filtered hybrid search of Chapters 7 and 8. That is, by the definition above, agentic RAG.

## Claim 3: agentic RAG is the future

**What it rests on.** In **agentic RAG**, retrieval is one or more tools, and the model decides when to search, what to search for, and whether the results are enough ([Chapter 7](07-rag.md#agentic-rag), [Chapter 11](11-agents.md#the-agent-loop)). The evidence behind it comes in three kinds:

- **Surveys.** The survey by Singh and colleagues (2025) describes how agentic design patterns enter the retrieval pipeline: reflection, planning, tool use and several agents working together. It proposes a taxonomy of architectures. It is a review of existing systems, with no new experiments, so it describes a trend rather than measuring it.
- **Models trained to search.** **Search-R1** (Jin and colleagues, 2025) uses **reinforcement learning**: instead of copying example answers, the model is trained by reward, receiving a reward only when its final answer is right. Through this training, it learns to write search queries in the middle of its reasoning and to search again when needed. The authors report gains over RAG baselines on seven question-answering datasets, for example 41% for a 7-billion-weight Qwen2.5 model.
- **Multi-agent research systems.** Anthropic reported that an orchestrator with sub-agents outperformed a single agent by 90.2% on its internal research evaluation. It also reported that it used about 15 times the tokens of a chat ([Chapter 11](11-agents.md#what-it-costs)).

**What it costs.** Every gain above is on questions that need several look-ups. Every one of them is paid for in model calls, tokens and waiting time. It also adds the risks of [Chapter 8](08-rag-security.md#unauthorised-tool-execution): retrieved text can now steer actions, so every tool call needs the policy check of [Chapter 11](11-agents.md#a-policy-check-before-every-tool-call).

Most of the bank's daily questions, such as *"How many days to dispute a payment?"*, need one look-up. For them, an agent adds cost and unpredictability and no quality.

## What the evidence says

| Claim | Where it holds | Where it fails | Verdict |
| --- | --- | --- | --- |
| **1. Long context makes RAG unnecessary** | Small, stable collections (under about 200,000 tokens, [Chapter 7](07-rag.md#worked-comparison-the-whole-manual-vs-five-chunks)), read whole, with prompt caching; questions about a whole document, such as a summary | Large or changing collections; questions worded unlike the text (NoLiMa); access rights per user; tight budgets | **Partly true.** Long context replaces retrieval for small collections; for the rest it becomes the *reading* step after retrieval |
| **2. Agentic search replaces RAG** | Code and other well-organised, local, fast-changing collections with exact identifiers | Large prose collections with per-user rights and high question volumes | **True for the vector index in some domains, not for retrieval.** `grep` is a retriever, and the agent still retrieves before it generates |
| **3. Agentic RAG is the future** | Questions with several dependent steps; research tasks; questions whose next search depends on the last result | Simple, high-volume questions, where it multiplies cost (about 12× here) and adds risk | **True for hard questions.** One-shot retrieval remains the default for simple ones, chosen by a router |

The studies point the same way from different directions:

- **Self-Route** shows that routing beats choosing one approach for everything.
- **LaRA** shows that the best approach depends on the model, the length and the task.
- **RULER, NoLiMa and Context Rot** show that a bigger window is not the same as better reading.
- **Claude Code** shows that the best retriever depends on the collection.

**What does not change**, whichever way the debate goes, is everything [Chapter 8](08-rag-security.md#production-checklist) built:

- access rights applied before text reaches the model;
- provenance and trusted sources;
- citations that point to a page;
- logs of what was retrieved for whom;
- measured recall and faithfulness.

A million-token prompt and an autonomous agent both still need these. In regulated industries such as banking, they are the main reason retrieval infrastructure stays, whatever the size of the context window.

## From RAG to context engineering

The direction the field is taking is often called **context engineering**: deciding, for every model call, what goes into the context window and what stays out. Anthropic's guide describes it as finding the smallest set of high-signal tokens that gets the job done. In that view, RAG is not a product to keep or abandon, but one tool among several for filling the context. Five trends follow from the evidence above. They are directions to watch, not certainties:

1. **Routing per question.** A cheap first step decides between no retrieval, one-shot RAG, long context or an agent, as Self-Route does, and as Self-RAG's `[Retrieve]` token does: Self-RAG is a model fine-tuned to write that token when it decides a search is needed ([Chapter 9](09-advanced-rag.md#self-rag-a-model-that-critiques-itself)). The bank's assistant could answer 85% of questions with one RAG call, and pass the rest to an agent.
2. **Retrieval as tools.** The same search becomes callable by any agent through a standard protocol such as MCP ([Chapter 11](11-agents.md#the-model-context-protocol-mcp)). Different retrievers sit side by side: hybrid search for the manual, SQL for transactions, ColPali for scanned pages.
3. **Models trained to retrieve.** Search behaviour is learned in training (Search-R1), not only written into prompts.
4. **Long context as the reading step.** Retrieval narrows millions of chunks down to 50 or 100 candidates. A long-context model then reads all of them, instead of only the top 5.
5. **Retrieval beyond text.** Pages are searched as images (ColPali, [Chapter 9](09-advanced-rag.md#retrieving-pages-as-images-colpali)), and the same late-interaction idea, one small vector per token or image patch instead of one per passage, extends to other media.

## Cheat sheet

| Concept | One-line meaning |
| --- | --- |
| Needle in a haystack | Find one fact in a long text; near-perfect scores hide harder failures |
| RULER, NoLiMa, Context Rot | Harder long-context tests: accuracy falls with length, especially without literal matches |
| Self-Route | RAG first; send only "unanswerable" questions to long context (65% and 39% cost cuts) |
| LaRA | 2,326 cases, 11 models: no silver bullet between RAG and long context |
| Agentic search | The model searches with tools (glob, grep, read) instead of a pre-built index |
| Agentic RAG | Retrieval as tools inside an agent loop; strong on multi-step questions, about 12× the tokens here |
| Context engineering | Choosing, for every call, the smallest set of high-signal tokens for the context |
| Verdict | Retrieval is not dying; it is becoming routed, tool-based, multimodal and still governed |

## Sources

- [Our next-generation model: Gemini 1.5 (Google, February 2024)](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024): 1 million tokens in production, 10 million in research, 99% on the needle test
- [RULER: What's the Real Context Size of Your Long-Context Language Models? (Hsieh et al., 2024)](https://arxiv.org/abs/2404.06654): 17 models, only half satisfactory at 32K
- [NoLiMa: Long-Context Evaluation Beyond Literal Matching (Modarressi et al., 2025; ICML 2025)](https://arxiv.org/abs/2502.05167): no literal matches; GPT-4o from 99.3% to 69.7% at 32K
- [Context Rot: How Increasing Input Tokens Impacts LLM Performance (Hong, Troynikov and Huber, Chroma, 2025)](https://research.trychroma.com/context-rot): 18 models; focused vs full input
- [Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach (Li et al., 2024; EMNLP 2024 Industry Track)](https://arxiv.org/abs/2407.16833): Self-Route, 65% and 39% cost reductions
- [LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs – No Silver Bullet for LC or RAG Routing (Li et al., 2025; ICML 2025)](https://arxiv.org/abs/2502.09977): 2,326 cases, 11 models
- [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG (Singh et al., 2025)](https://arxiv.org/abs/2501.09136): taxonomy of agentic RAG
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (Jin et al., 2025)](https://arxiv.org/abs/2503.09516): search learned through reinforcement learning
- [Effective context engineering for AI agents (Anthropic, September 2025)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): just-in-time retrieval, the hybrid strategy of Claude Code
- [Building Claude Code with Boris Cherny (The Pragmatic Engineer)](https://newsletter.pragmaticengineer.com/p/building-claude-code-with-boris-cherny): agentic search with glob and grep instead of a vector database
- [Claude Code: Anthropic's CLI Agent (Latent Space podcast, May 2025)](https://www.latent.space/p/claude-code): Boris Cherny and Cat Wu on why early RAG was dropped
- [How we built our multi-agent research system (Anthropic, 2025)](https://www.anthropic.com/engineering/multi-agent-research-system): 90.2% better than a single agent, at about 15× the tokens of a chat
