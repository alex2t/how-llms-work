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
