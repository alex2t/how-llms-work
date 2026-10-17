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
