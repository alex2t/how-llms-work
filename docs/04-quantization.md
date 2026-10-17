# Quantization, Efficiency and the Modern Transformer

## The big picture

[Chapter 3](03-inference.md) showed that generating text is limited mostly by memory: every decode step reads every weight, and the KV cache grows with every token. This chapter is about making models smaller and faster without making them noticeably worse. The main tool is **quantization**: storing each number with fewer bits.

The chapter follows one question, "how many bytes does this model need, and what do we lose by using fewer?":

1. **Why size matters:** memory = number of weights × bytes per weight, and decode speed follows from it.
2. **Number formats:** what FP32, FP16, BF16, INT8 and INT4 can and cannot store.
3. **How quantization works:** turning real numbers into small integers with a scale, and the rounding error this causes.
4. **Kinds of quantization:** when it is done, what is quantized, and the common methods (GPTQ, AWQ, GGUF).
5. **Quality vs speed:** when the savings start to cost accuracy.
6. **What changed since 2017:** the improvements that turned the original Transformer of [Chapter 1](01-transformers.md) into today's models, several of them also aimed at efficiency.

## Why size matters

A model's size decides which hardware can run it and how fast it generates. Both follow from one multiplication.

### Memory = weights × bytes per weight

Every weight is stored as a number in a fixed format, and each format uses a fixed number of bytes. So:

$$
\text{memory for the weights} = \text{number of weights} \times \text{bytes per weight}
$$

- **number of weights** is the parameter count, e.g. 65 million for the base model of [Chapter 1](01-transformers.md#key-settings-of-the-base-model) or 7 billion for a small modern model.
- **bytes per weight** depends on the number format: 4 bytes for 32-bit numbers, 2 for 16-bit, 1 for 8-bit and half a byte for 4-bit. The formats are explained in the next section.

The base model at 4 bytes per weight needs 65,000,000 × 4 = 260 MB, as in [Chapter 3](03-inference.md#why-inference-needs-less-memory).

### Worked example: a 7-billion-weight model

| Format | Bytes per weight | Memory for 7 billion weights |
| --- | --- | --- |
| 32-bit (FP32) | 4 | 7,000,000,000 × 4 = 28 GB |
| 16-bit (FP16 or BF16) | 2 | 14 GB |
| 8-bit (INT8) | 1 | 7 GB |
| 4-bit (INT4) | 0.5 | 3.5 GB |

What this means in practice:

- **At 32 bits** the model does not fit on a typical 24 GB graphics card.
- **At 16 bits** it fits, with 10 GB left for the KV cache. One 4,096-token conversation needs about 2.15 GB of cache ([Chapter 3](03-inference.md#how-much-memory-the-cache-needs)), so the total is about 16.1 GB.
- **At 4 bits** it fits on a laptop with 8 GB of memory.

The same arithmetic decides which large models are practical. A 70-billion-weight model needs 140 GB at 16 bits, more than any single common GPU holds, but 35 GB at 4 bits.

### Fewer bytes also means faster generation

[Chapter 3](03-inference.md#why-decode-is-limited-by-memory-not-arithmetic) showed that each decode step must read every weight from memory, so the speed limit is roughly memory speed ÷ model size. With the same illustrative GPU that reads 2,000 GB per second:

| Format | Model size | Maximum tokens per second (2,000 ÷ size in GB) |
| --- | --- | --- |
| FP32 | 28 GB | 71 |
| FP16 / BF16 | 14 GB | 143 |
| INT8 | 7 GB | 286 |
| INT4 | 3.5 GB | 571 |

Halving the bytes per weight roughly doubles the decode speed limit. In practice the gain is a little smaller, because the small integers must be converted back to ordinary numbers before the arithmetic, but the direction is the same. This is the main reason quantization is used almost everywhere models are served.
