# Inference and Decoding: How the Model Generates Text

## The big picture

[Chapter 2](02-training.md) explained how the model's weights are learned. This chapter explains what happens when the trained model is used: you type a prompt, and the model writes an answer one token at a time. This use of a trained model is called **inference**.

Each answer is produced by a simple loop:

1. **Read the prompt.** Tokenize it and run it through the model once, exactly as in [Chapter 1](01-transformers.md).
2. **Score every token.** The last position gives a probability for every token in the vocabulary.
3. **Choose one token.** A decoding rule picks the next token from those probabilities: always the most likely one, or a random choice shaped by settings such as temperature.
4. **Append and repeat.** Add the chosen token to the input and go back to step 2.
5. **Stop.** End when the model produces an end token or a length limit is reached, then turn the token IDs back into text.

The rest of the chapter walks through each step. It starts with what makes inference different from training, then shows the loop on the translation example, explains why the first token is slower than the others and how the KV cache speeds things up, and ends with the decoding rules that turn probabilities into a choice.

## Inference vs training

Inference uses the same model and the same forward pass as training, but nothing is learned. Knowing exactly what is switched off explains why using a model is so much cheaper than training it, and why the same prompt can give the same answer every time.

### Definitions

- **Inference:** running a trained model on new input to get an output. In a chat model, it means generating an answer to a prompt.
- **Frozen weights:** weights that are only read, never changed. After the last training stage, every weight in E, W\_Q, W\_K, W\_V, W\_O and the feed-forward layers is frozen.
- **Forward pass:** one run of the input through the model, from token IDs to probabilities, as described in [Chapter 1](01-transformers.md).

### What is switched off

| | Training ([Chapter 2](02-training.md)) | Inference |
| --- | --- | --- |
| Weights | Change after every batch | Frozen: read only |
| Forward pass | Yes | Yes, the same computation |
| Correct answer known? | Yes, the next token in the training text | No: the model's output *is* the answer |
| Loss | Computed at every position | Not computed: there is nothing to compare with |
| Backward pass (backpropagation) | Yes | No |
| Optimiser (Adam) | Updates every weight | Not used |
| [Dropout](02-training.md#dropout-forcing-the-model-not-to-rely-on-any-single-number) | On: randomly zeroes 10% of numbers | Off: every number is kept |
| Decoder input | The true text ([teacher forcing](02-training.md#teacher-forcing-why-training-happens-in-one-time-step)) | The model's own previous outputs |
| Typical cost | Weeks on thousands of GPUs | Seconds per answer |

The most important row is "Correct answer known?". During training, after `YOUR CAT IS A LOVELY`, the text says the next token is `CAT`, so the model's probability of 0.772 for `CAT` can be turned into a loss of 0.259 ([Chapter 2](02-training.md#worked-example-chapter-1s-probabilities)). At inference, the same probabilities come out: CAT 0.772, DOG 0.211, IS 0.013, LOVELY 0.004. But there is no correct answer to compare them with. The model simply **uses** them to choose the next token. With no loss there is nothing to backpropagate, and with no gradients there is nothing for the optimiser to do.

### Why inference needs less memory

Training must keep more than the weights in memory. For every weight it also stores a **gradient** and Adam's two running averages **m** and **v** ([Chapter 2](02-training.md#adam-a-smarter-step)). That is four numbers per weight instead of one.

**Worked example** with the base model's 65 million weights, each stored as a 32-bit number of 4 bytes ([Chapter 4](04-quantization.md) explains number formats):

| What is stored | Numbers per weight | Memory |
| --- | --- | --- |
| Inference: weights only | 1 | 65,000,000 × 4 bytes = 260 MB |
| Training: weights + gradients + m + v | 4 | 4 × 260 MB = 1,040 MB |

Training needs four times as much memory before counting anything else. (Training also keeps the intermediate results of the forward pass for the backward pass, which adds more.) Inference has one extra cost of its own, the KV cache, explained later in this chapter.

### Same input, same output?

With frozen weights and dropout switched off, the forward pass is **deterministic**: the same prompt always gives exactly the same probabilities. Any variety in a model's answers comes only from step 3 of the loop, the decoding rule. If the rule always picks the most likely token, the answer is the same every time; if it picks randomly according to the probabilities, answers can differ. The decoding rules are explained at the end of this chapter.

## Generation step by step

A model writes its answer one token at a time: each new token needs its own forward pass, and each pass can only start once the previous token is chosen. This section follows that loop on the translation example from [Chapter 2](02-training.md#the-translation-example), then shows that chat models such as Claude work the same way.

### Encoder–decoder: translating "I love you very much"

The task is the one used to train the model in Chapter 2: translate "I love you very much" into Italian. As there, each word counts as one token to keep the tables short.

**The encoder runs once.** The source sentence `<SOS>` I love you very much `<EOS>` (7 tokens) goes through the encoder in a single pass. Its output, one 512-number vector per source token (a 7 × 512 matrix), is kept and reused, unchanged, at every decoder step. The decoder reads it through cross-attention ([Chapter 1](01-transformers.md#self-attention-vs-cross-attention)): the encoder output supplies the K and V, the decoder supplies the Q.

**The decoder runs once per output token.** It starts with only `<SOS>` and grows its input by one token per step:

| Step | Decoder input | Positions in the input | Prediction used | Chosen token |
| --- | --- | --- | --- | --- |
| 1 | `<SOS>` | 1 | Position 0 | Ti |
| 2 | `<SOS>` Ti | 2 | Position 1 | amo |
| 3 | `<SOS>` Ti amo | 3 | Position 2 | molto |
| 4 | `<SOS>` Ti amo molto | 4 | Position 3 | `<EOS>`: stop |

The final decoder input `<SOS>` Ti amo molto is exactly the decoder input used in training. The difference is how it was built: in training it came from the data and went through the model in one pass; here the model built it itself, one token per pass. [Chapter 2's comparison table](02-training.md#training-vs-generation-for-the-same-sentence) sums this up: 1 forward pass in training, 4 in generation.

**One step in detail.** At step 2 the decoder reads `<SOS>` Ti and the last position outputs a probability for every Italian token. With made-up numbers:

| Candidate | Probability |
| --- | --- |
| amo | 0.81 |
| adoro | 0.12 |
| voglio | 0.04 |
| all other tokens together | 0.03 |

The simplest decoding rule, taking the most likely token, picks `amo`. It is appended, and step 3 starts from `<SOS>` Ti amo. Other decoding rules are explained at the end of this chapter.

**Why only the last position is used.** At step 4 the decoder outputs predictions for all four positions, but positions 0 to 2 only repeat the predictions already made at steps 1 to 3. Only the last position says something new. Recomputing the earlier positions is wasted work: across the four steps the decoder processes 1 + 2 + 3 + 4 = 10 positions to produce 4 tokens. The KV cache, later in this chapter, removes most of that waste.

### Decoder-only: how chat models generate

GPT-style models and Claude have no encoder ([Chapter 1](01-transformers.md#three-ways-the-block-is-used-today)). The prompt and the answer live in **one sequence**: the model reads the prompt, then keeps appending its own tokens to the end of that same sequence.

The same translation, asked as a prompt, looks like the supervised fine-tuning example in [Chapter 2](02-training.md#stage-2-supervised-fine-tuning-sft):

| Step | Model input (prompt + answer so far) | Chosen token |
| --- | --- | --- |
| 1 | Translate to Italian: I love you very much | Ti |
| 2 | Translate to Italian: I love you very much Ti | amo |
| 3 | Translate to Italian: I love you very much Ti amo | molto |
| 4 | Translate to Italian: I love you very much Ti amo molto | `<EOS>`: stop |

The prompt plays the role the encoder played above: it is the information the answer depends on. The causal mask ([Chapter 1](01-transformers.md#what-the-mask-looks-like)) is the same as in training, so each new token sees the prompt and the answer so far, and nothing after it.

The same loop also explains [Chapter 1's logits example](01-transformers.md#from-the-last-vector-back-to-a-word). Given the prompt `YOUR CAT IS A LOVELY`, the last position outputs CAT 0.772, DOG 0.211, IS 0.013, LOVELY 0.004. Taking the most likely token appends `CAT`, and the next pass starts from `YOUR CAT IS A LOVELY CAT`.

In a chat, "the prompt" is the whole conversation so far: earlier messages, the new question, and some special tokens that mark who is speaking. The model generates until it produces its end-of-answer token.

### Encoder–decoder vs decoder-only at inference

| | Encoder–decoder (translation model) | Decoder-only (chat model) |
| --- | --- | --- |
| What reads the input | The encoder, once | The same stack that writes the answer |
| Where the answer goes | A separate decoder sequence starting with `<SOS>` | Appended to the end of the prompt |
| How the answer sees the input | Cross-attention to the encoder output | Ordinary masked self-attention over the prompt |
| Forward passes for "Ti amo molto `<EOS>`" | 1 encoder pass + 4 decoder passes | 4 passes (the first one also reads the prompt) |
| Stops at | `<EOS>` or a length limit | End-of-answer token or a length limit |

## Prefill vs decode: why the first token is slower

Generation has two phases that use the hardware in very different ways. Knowing them explains two things every chat user notices: there is a pause before the first word appears, and after that the words stream out at a steady pace.

### Definitions

- **Prefill (prompt processing):** the first forward pass, which reads the whole prompt and produces the first output token.
- **Decode (token generation):** every pass after that, each producing one more token.
- **Time to first token (TTFT):** how long the user waits before the first token appears. It is mostly the prefill time.
- **Time per output token:** how long each following token takes. Its inverse is the familiar "tokens per second".
- **FLOP (floating-point operation):** one multiplication or addition of two numbers. Running one token through a model costs about **2 × (number of weights)** FLOPs, because each weight is used in one multiplication and one addition.

### Prefill works like training

The whole prompt is known before generation starts, just as the whole target sentence is known in training. So all prompt tokens go through the model **in one pass**, in parallel, with the causal mask keeping each position from seeing later ones ([Chapter 2](02-training.md#why-training-happens-in-one-time-step)). A 1,000-token prompt does not need 1,000 passes; it needs one pass over a 1,000-row matrix.

Decode cannot do this. Token 2 of the answer depends on which token 1 was chosen, so each decode pass must wait for the one before it. With the KV cache (next section), each decode pass only has to process the **one new token**.

### Why decode is limited by memory, not arithmetic

Every forward pass has to read **all the weights** from the GPU's memory, whether it processes 1,000 tokens or 1. What changes is how much arithmetic each weight read pays for.

**Worked example with the base model** (65 million weights at 4 bytes each = 260 MB):

| | Prefill, 1,000-token prompt | Decode, 1 new token |
| --- | --- | --- |
| Weights read from memory | 260 MB | 260 MB |
| Arithmetic: 2 × 65,000,000 × tokens | 130,000,000,000 FLOPs | 130,000,000 FLOPs |
| Arithmetic per byte read | 500 FLOPs | 0.5 FLOPs |

In prefill, each weight is read once and used for 1,000 tokens, so the GPU is busy calculating: prefill is **compute-bound** (limited by arithmetic speed). In decode, each weight is read and used for a single token, so the GPU spends most of its time waiting for memory: decode is **memory-bound** (limited by how fast memory can be read). This counts only the weights; attention adds a smaller cost that grows with the length of the text.

### Worked example: a 7-billion-weight model

The effect is easier to feel on a model the size of today's small open models. Assume a model with 7 billion weights stored in 16-bit format, 2 bytes each ([Chapter 4](04-quantization.md) covers number formats), and a GPU that can read 2,000 GB of memory per second and compute 500 trillion FLOPs per second. These are round illustrative figures; real GPUs vary.

- Weights in memory: 7,000,000,000 × 2 bytes = **14 GB**.

| | Prefill, 1,000-token prompt | Decode, 1 token |
| --- | --- | --- |
| Time to read the weights: 14 GB ÷ 2,000 GB/s | 7 ms | 7 ms |
| Time for the arithmetic: 2 × 7 billion × tokens ÷ 500 trillion per second | 28 ms | 0.028 ms |
| Time taken (the slower of the two) | **28 ms**, limited by arithmetic | **7 ms**, limited by memory |

For a 1,000-token prompt and a 200-token answer:

- The first token appears after about **28 ms** (prefill).
- Each later token takes about **7 ms**, so the model streams at most 1 ÷ 0.007 ≈ **143 tokens per second**.
- The whole answer takes about 28 ms + 200 × 7 ms = **1.43 seconds**.

Prefill handled 1,000 tokens in 28 ms; decode would need 1,000 × 7 ms = 7 seconds for the same number of tokens. Per token, prefill is 250 times faster, because it reads the weights once for all of them.

### What this means in practice

- **Long prompts delay the first token.** Prefill time grows with the prompt length, so a 10,000-token document takes about ten times longer to process than a 1,000-token prompt before any answer appears.
- **Streaming speed depends on memory speed and model size.** Each decode step reads every weight, so a smaller model, or the same model stored in fewer bytes per weight, generates faster. This is a main reason for quantization ([Chapter 4](04-quantization.md)).
- **Serving many users at once is efficient.** When several users' decode steps are grouped into one batch, the weights are read once and used for all of them. The memory cost is shared, which is why providers batch requests together.

## The KV cache

The KV cache is the main trick that makes generation fast: it stores the keys and values of tokens already processed, so each decode step only computes them for the one new token. It saves a great deal of arithmetic, but it costs memory, and that memory cost shapes how long a conversation can be and how many users a GPU can serve.

### The waste it removes

In the translation example, the decoder processed 1 + 2 + 3 + 4 = 10 positions to produce 4 tokens. At step 4, for example, it recomputed the key and value vectors of `<SOS>`, Ti and amo, which it had already computed at steps 1 to 3, and got exactly the same numbers.

They are exactly the same because nothing that produces them has changed:

- The weights W\_K and W\_V are frozen.
- A token's vector at every layer depends only on itself and the tokens **before** it, because the causal mask ([Chapter 1](01-transformers.md#what-the-mask-looks-like)) hides everything after it. Appending a new token at the end cannot change the vectors of the tokens before it.

So the key and value of each earlier token, in every layer and head, can be computed once and kept.

### What the KV cache is

The **KV cache** is a store, in GPU memory, of the key vector k and value vector v of every token processed so far, for every layer and every head. Each decode step then does only this, in each layer and head:

1. Compute q, k and v for the **new token only**: one row of 512 numbers times W\_Q, W\_K and W\_V, giving three vectors of 64 numbers ([Chapter 1](01-transformers.md#where-q-k-and-v-come-from)).
2. Append the new k and v to the cache.
3. Compare the new q with **every** key in the cache to get the attention scores, apply softmax, and take the weighted sum of **every** value in the cache.

Only the new token's query is needed: earlier tokens' predictions were already made, so their queries are never used again. This is why the cache holds K and V, but not Q.

### The cache growing, step by step

For the decoder's self-attention in the translation example:

| Step | New token computed | Keys and values in the cache after the step | Positions processed |
| --- | --- | --- | --- |
| 1 | `<SOS>` | `<SOS>` | 1 |
| 2 | Ti | `<SOS>`, Ti | 1 |
| 3 | amo | `<SOS>`, Ti, amo | 1 |
| 4 | molto | `<SOS>`, Ti, amo, molto | 1 |
| **Total** | | | **4** instead of 10 |

The saving grows with length. For an answer of 1,000 tokens, the decoder without a cache would process 1 + 2 + … + 1,000 = 500,500 positions; with the cache it processes 1,000.

For cross-attention the saving is even simpler: the K and V computed from the encoder output never change during the translation, so they are computed once, before step 1, and reused at every step. In a decoder-only chat model, prefill fills the cache with the keys and values of the whole prompt in one pass, and each decode step adds one token.

### How much memory the cache needs

For every token, the cache holds one key and one value vector in every layer. Across all heads of a layer, these are d\_model numbers each (8 heads × 64 = 512 in the base model). So:

$$
\text{cache bytes per token} = 2 \times \text{layers} \times d_{model} \times \text{bytes per number}
$$

- **2** counts the key and the value.
- **layers** is the number of layers that attend to earlier tokens.
- **d\_model** is the number of key (or value) numbers per token per layer, summed over all heads.
- **bytes per number** is 4 for 32-bit numbers and 2 for 16-bit numbers.

**Base model** (6 decoder layers, d\_model = 512, 4 bytes):

| Quantity | Calculation | Result |
| --- | --- | --- |
| Per token | 2 × 6 × 512 × 4 bytes | 24,576 bytes ≈ 24.6 KB |
| 1,000 tokens | 24,576 × 1,000 bytes | ≈ 24.6 MB |

That is about 9% of the 260 MB of weights: small, because the base model is small.

**A 7-billion-weight model** with settings similar to Llama 2 7B (32 layers, d\_model = 4,096, 16-bit numbers of 2 bytes):

| Quantity | Calculation | Result |
| --- | --- | --- |
| Per token | 2 × 32 × 4,096 × 2 bytes | 524,288 bytes ≈ 0.52 MB |
| One conversation of 4,096 tokens | 524,288 × 4,096 bytes | ≈ 2.15 GB |
| 32 users at once, 4,096 tokens each | 32 × 2.15 GB | ≈ 68.7 GB |

The weights of this model take 14 GB (previous section). With 32 users the cache takes almost five times as much memory as the model itself.

### The trade-off

- **Compute saved:** each decode step processes one new token instead of recomputing every earlier token.
- **Memory spent:** the cache grows by a fixed amount with every token, for every user. On a busy server it, not the weights, often decides how many conversations fit on one GPU.
- **Attention still grows with length:** the new query must still be compared with every cached key, so each step does a little more work than the one before. This is much smaller than the work the cache saves.

This is why longer conversations cost more to serve, and why model designers try to shrink the cache. Chapter 4 describes the two main ways: storing it in fewer bytes ([quantization](04-quantization.md)) and sharing keys and values between heads (grouped-query attention).

## Choosing the next token

Every decode step ends with a probability for every token in the vocabulary. A **decoding rule** (also called a decoding strategy or sampling method) turns those probabilities into one chosen token. The rule does not change the model at all, yet it decides whether answers are repetitive or varied, safe or creative.

All the rules below use the same example from [Chapter 1](01-transformers.md#from-the-last-vector-back-to-a-word). After the prompt `YOUR CAT IS A LOVELY`, the model's last position gives these logits (raw scores) and, after softmax, these probabilities. For simplicity, the vocabulary is just these four tokens.

| Token | Logit | Probability |
| --- | --- | --- |
| CAT | 6.2 | 0.772 |
| DOG | 4.9 | 0.211 |
| IS | 2.1 | 0.013 |
| LOVELY | 1.0 | 0.004 |

### Greedy decoding

**Greedy decoding** always picks the token with the highest probability. Here it picks `CAT` (0.772), every time.

- **Deterministic:** the same prompt always gives the same answer, because the forward pass is deterministic and so is the choice.
- **Short-sighted:** it picks the best token for this one step, without checking whether that choice leads to a good sentence overall.
- **Prone to repetition:** in long answers it can fall into loops, repeating the same phrase, because once a phrase is likely, repeating it often stays the most likely choice.

### Beam search

**Beam search** looks a few steps ahead by keeping several partial answers at once instead of one. Each partial answer is called a **beam**, and the **beam width** (here 2) is how many are kept after each step. A beam's score is the probability of the whole sequence: the product of the probabilities of its tokens.

**Worked example, beam width 2.** The step-1 probabilities are the ones above. The step-2 probabilities are made up for illustration: after `CAT` the model is unsure how to continue, while after `DOG` it is confident the sentence should end.

**Step 1.** Keep the 2 most likely tokens: `CAT` (0.772) and `DOG` (0.211).

**Step 2.** Extend each beam with its 2 most likely next tokens, and score each extended sequence:

| Beam after step 1 | Next token | Its probability | Sequence score |
| --- | --- | --- | --- |
| CAT (0.772) | `<EOS>` | 0.20 | 0.772 × 0.20 = 0.154 |
| CAT (0.772) | AND | 0.18 | 0.772 × 0.18 = 0.139 |
| DOG (0.211) | `<EOS>` | 0.95 | 0.211 × 0.95 ≈ **0.200** |
| DOG (0.211) | AND | 0.02 | 0.211 × 0.02 = 0.004 |

Keep the 2 best: `DOG <EOS>` (0.200) and `CAT <EOS>` (0.154). Both have ended, so the search stops and returns the best one, `YOUR CAT IS A LOVELY DOG`.

**Greedy decoding on the same example** picks `CAT` at step 1 and `<EOS>` at step 2, for a sequence score of 0.154. Beam search found a sequence with a higher total probability, 0.200, because it kept `DOG` alive even though `DOG` lost the first step.

Three practical details:

- **Logarithms instead of products.** Multiplying many probabilities gives numbers too small for a computer to store accurately, so implementations add the logarithms instead: ln 0.211 + ln 0.95 = −1.607 is higher than ln 0.772 + ln 0.20 = −1.868, giving the same ranking.
- **Length penalty.** Every extra token multiplies the score by a number below 1, so plain beam search favours short answers. Implementations divide the score by a function of the length to correct this. The original Transformer used a beam width of 4 for translation, with a length penalty.
- **Where it is used.** Beam search suits tasks with one best answer, such as translation and speech recognition. Chat models rarely use it: it is slower (several beams per step), and its answers tend to be bland and repetitive. They use sampling instead, described next.
