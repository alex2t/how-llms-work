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

## Number formats

Every number in a model, whether a weight, an activation or a value in the KV cache, is stored in a **number format**: a fixed rule for turning a pattern of bits into a value. The format decides how much memory each number uses, how large or small a value it can hold, and how precisely it holds it.

### Floating-point numbers: range and precision

Most model numbers are **floating-point** numbers. A floating-point number works like scientific notation, 1.2345 × 10³, stored in binary. Its bits are split into three parts:

- **Sign (1 bit):** positive or negative.
- **Exponent:** the power of 2 that sets the size of the number. More exponent bits give a wider **range**, from very small to very large values.
- **Fraction (also called mantissa):** the significant digits. More fraction bits give more **precision**, that is, more correct digits.

The three floating-point formats used in LLMs split their bits differently:

| Format | Total bits | Sign | Exponent | Fraction | Largest value | Correct decimal digits (about) |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 (single precision) | 32 | 1 | 8 | 23 | 3.4 × 10³⁸ | 7 |
| FP16 (half precision) | 16 | 1 | 5 | 10 | 65,504 | 3 to 4 |
| BF16 (brain float 16) | 16 | 1 | 8 | 7 | 3.4 × 10³⁸ | 2 to 3 |

FP16 and BF16 both use 16 bits but make opposite choices. FP16 keeps more fraction bits, so it is more precise, but its 5 exponent bits give a small range. BF16, designed at Google for machine learning, keeps FP32's 8 exponent bits, so it has the same huge range as FP32, but only 7 fraction bits.

### Worked example: storing the same values in each format

The values below were stored in each format and read back (computed with NumPy):

| Value to store | FP32 stores | FP16 stores | BF16 stores |
| --- | --- | --- | --- |
| 0.1234567 (a typical weight) | 0.1234567 | 0.12347 | 0.12354 |
| 70,000 (a large activation) | 70,000 | **infinity** (too large) | 70,144 |
| 0.00000001 (a tiny gradient) | 0.00000001 | **0** (too small) | 0.0000000100 |

- **A typical weight:** all three formats keep the value close. FP16's error is 0.000017 and BF16's is 0.000078, both small compared with the weight itself.
- **A large value:** FP16 cannot store anything above 65,504, so 70,000 becomes infinity and the calculation breaks. BF16 stores it with a small error.
- **A tiny value:** FP16 rounds 0.00000001 to 0, so the information is lost. BF16 keeps it.

This is why **BF16 is the usual format for training** today: running out of range breaks training, while BF16's lower precision is usually harmless. For inference, both FP16 and BF16 are common.

### Integer formats

Quantized models store weights as **integers** (whole numbers) with very few bits:

| Format | Bits | Bytes | Values it can hold |
| --- | --- | --- | --- |
| INT8 | 8 | 1 | 256 whole numbers, from −128 to 127 |
| INT4 | 4 | 0.5 | 16 whole numbers, from −8 to 7 |

An integer format cannot store 0.1234567 at all, only whole numbers such as 12 or −3. To use it for weights, every weight must be mapped onto those few whole numbers, and mapped back when it is needed. That mapping is quantization, explained in the next section.

### Summary of the formats

| Format | Bytes per number | Range | Precision | Typical use |
| --- | --- | --- | --- | --- |
| FP32 | 4 | Huge | High | Optimiser state, master copy of weights during training |
| BF16 | 2 | Huge | Low | Training and inference |
| FP16 | 2 | Small | Medium | Inference |
| INT8 | 1 | 256 steps, with a scale | Steps of a fixed size | Quantized weights, KV cache |
| INT4 | 0.5 | 16 steps, with a scale | Coarse steps | Quantized weights |

## How quantization works

**Quantization** means storing numbers in a format with fewer bits, usually by mapping real-valued weights onto a small set of whole numbers. The mapping uses one extra number, the **scale**, which says how much one integer step is worth. Understanding the scale explains both why quantization works so well and where its errors come from.

### Definitions

- **Quantize:** turn a real number into a small integer, q = round(w ÷ scale).
- **Dequantize:** turn the integer back into an approximate real number, ŵ = q × scale. This happens just before the weight is used in a calculation.
- **Scale:** the size of one integer step, chosen so the largest weight lands on the largest allowed integer.
- **Rounding error:** the difference between the original weight w and the dequantized ŵ. It can never be larger than half a step (scale ÷ 2).

$$
\text{scale} = \frac{\max |w|}{q_{\max}} \qquad q = \text{round}\!\left(\frac{w}{\text{scale}}\right) \qquad \hat{w} = q \times \text{scale}
$$

- **max |w|** is the largest weight ignoring its sign.
- **q\_max** is the largest integer used: 127 for INT8 and 7 for INT4.

This version is called **symmetric** quantization: 0 stays exactly 0, and positive and negative weights use the same step.

### Worked example: five weights to INT8

Take five weights (made-up values): 0.42, −1.30, 0.053, 0.871, −0.317.

1. **Find the scale.** The largest weight ignoring sign is 1.30, so scale = 1.30 ÷ 127 = 0.010236.
2. **Quantize.** Divide each weight by the scale and round to the nearest whole number.
3. **Store** the five integers (1 byte each) and the one scale.
4. **Dequantize** when needed: multiply each integer by the scale.

| Weight w | w ÷ scale | Stored integer q | Dequantized ŵ = q × 0.010236 | Rounding error |
| --- | --- | --- | --- | --- |
| 0.42 | 41.03 | 41 | 0.4197 | 0.0003 |
| −1.30 | −127.00 | −127 | −1.3000 | 0.0000 |
| 0.053 | 5.18 | 5 | 0.0512 | 0.0018 |
| 0.871 | 85.09 | 85 | 0.8701 | 0.0009 |
| −0.317 | −30.97 | −31 | −0.3173 | 0.0003 |

The largest error is 0.0018, below the half-step limit of 0.010236 ÷ 2 = 0.0051. Each weight is now stored in 1 byte instead of 4, and its value is still correct to about 2 decimal places.

### The same weights in INT4

INT4 allows only the integers −7 to 7 here, so the steps are much bigger: scale = 1.30 ÷ 7 = 0.1857.

| Weight w | Stored integer q | Dequantized ŵ = q × 0.1857 | Rounding error |
| --- | --- | --- | --- |
| 0.42 | 2 | 0.3714 | 0.0486 |
| −1.30 | −7 | −1.3000 | 0.0000 |
| 0.053 | 0 | 0.0000 | 0.0530 |
| 0.871 | 5 | 0.9286 | 0.0576 |
| −0.317 | −2 | −0.3714 | 0.0544 |

The largest error grows from 0.0018 to 0.0576, about 30 times larger than with INT8, and the small weight 0.053 has become exactly 0. Each weight now takes only half a byte. Whether these errors matter depends on how many weights there are and how cleverly the rounding is done, the subject of the next two sections.

### Zero-point (asymmetric) quantization

The weights above range from −1.30 to 0.871, but symmetric INT8 reserves integers up to +127 for positive values that never appear: the largest positive weight only reaches 85. **Asymmetric** quantization fixes this by mapping the actual range, minimum to maximum, onto all 256 values of an unsigned 8-bit integer (0 to 255). It needs a second stored number, the **zero point**: the integer that represents the real value 0.

- scale = (0.871 − (−1.30)) ÷ 255 = 0.008514
- zero point = round(1.30 ÷ 0.008514) = 153
- q = round(w ÷ scale) + zero point, so −1.30 → 0, 0 → 153, and 0.871 → 255.

The step is now 0.0085 instead of 0.0102, because no integer values are wasted. Asymmetric quantization is most useful for activations, which are often all positive (for example after ReLU), so half of a symmetric range would be wasted.

### The outlier problem and group-wise scales

One scale for many weights works well only if the weights have similar sizes. Add a single large weight, 10.0, to the five above and quantize all six to INT4 with one scale:

- scale = 10.0 ÷ 7 = 1.43
- 0.42, 0.053 and −0.317 all round to 0, and −1.30 becomes −1.43.

Three of the five ordinary weights are wiped out by one **outlier** (a value much larger than the rest). Real models do have such outliers, so quantizing a whole matrix with one scale gives poor results.

The standard fix is **group-wise quantization**: split each row of weights into small groups, typically 32, 64 or 128 weights, and give each group its own scale. An outlier then only spoils the scale of its own group. The cost is the extra scales. With one 16-bit scale per group:

| Group size | Bits per weight for INT4 | Bits per weight for INT8 |
| --- | --- | --- |
| 32 | 4 + 16 ÷ 32 = 4.5 | 8.5 |
| 64 | 4.25 | 8.25 |
| 128 | 4.125 | 8.125 |

So a "4-bit" 7-billion-weight model with groups of 32 really uses 4.5 bits per weight: 7,000,000,000 × 4.5 ÷ 8 ≈ 3.9 GB, not 3.5 GB.

## Kinds of quantization

"A 4-bit model" can mean several quite different things. Quantization methods differ in **when** the quantization is done, **what** is quantized, and **how** the rounding is chosen. This section sorts them out, so that names such as GPTQ, AWQ or a GGUF file can be placed at a glance.

### When: after training, or during it

| | Post-training quantization (PTQ) | Quantization-aware training (QAT) |
| --- | --- | --- |
| When | After training is finished | During training (or extra fine-tuning) |
| How | Take the trained weights and round them, as in the previous section | The forward pass simulates the rounding, so the model learns weights that still work after rounding |
| Cost | Minutes to hours; needs at most a few hundred example prompts | A training run, with all its data and compute ([Chapter 2](02-training.md)) |
| Quality | Very good at 8 bits; good at 4 bits with a smart method | Best at very low bit counts |
| Typical use | Almost all open models you can download in quantized form | Models whose makers publish an official low-bit version |

Most quantization in practice is post-training. The smart PTQ methods below improve on plain rounding by using a small set of example inputs, called **calibration data**, to see which weights matter most.

### What: weights, activations, or the KV cache

- **Weights only.** The weights are stored as INT8 or INT4 and dequantized to 16 bits just before each matrix multiplication; the arithmetic itself stays in 16 bits. This saves memory and, because decode is memory-bound ([Chapter 3](03-inference.md#why-decode-is-limited-by-memory-not-arithmetic)), it speeds up generation. It is the most common kind. Such setups are written **W4A16**: 4-bit weights, 16-bit activations.
- **Weights and activations.** The **activations** (the vectors flowing between layers while the model runs) are quantized too, so the matrix multiplication itself runs in 8-bit integers (**W8A8**). This also speeds up the arithmetic, which helps prefill, which is compute-bound. It is harder: activations change with every input, and large models have a few activation dimensions with very large values (outliers) that must be handled separately.
- **KV cache.** The keys and values stored during generation can be quantized as well. With the 7-billion-weight model of [Chapter 3](03-inference.md#how-much-memory-the-cache-needs), INT8 halves the cache from 0.52 MB to 0.26 MB per token, so 32 users with 4,096 tokens each need about 34.4 GB instead of 68.7 GB.

### How: three names you will meet

- **GPTQ** (2022) quantizes a layer's weights one column at a time. After rounding each column, it slightly adjusts the weights not yet quantized to compensate for the error just introduced, using calibration data to measure which errors affect the layer's output most. The result is much better than rounding every weight independently, especially at 4 bits.
- **AWQ** (activation-aware weight quantization, 2023) starts from the observation that about 1% of weights matter far more than the rest: those multiplied by large activations. It finds them from calibration data and scales them up before quantization, so they lose less precision, while the matching activations are scaled down to keep the result the same.
- **GGUF** is not a quantization method but a **file format**, used by the llama.cpp project and tools built on it, for running models on ordinary computers. A GGUF file holds the weights already quantized in blocks, using types named like `Q8_0` or `Q4_K_M`. For example, `Q8_0` stores blocks of 32 weights as 8-bit integers with one 16-bit scale per block: (32 × 8 + 16) ÷ 32 = 8.5 bits per weight. `Q4_0` does the same with 4-bit integers: 4.5 bits per weight. These are the group-wise scales of the previous section.

### Summary

| Choice | Options | Most common today |
| --- | --- | --- |
| When | Post-training (PTQ) or quantization-aware training (QAT) | PTQ |
| What | Weights only; weights + activations; KV cache | Weights only (W4A16 or W8A16), often plus an 8-bit KV cache |
| How | Plain rounding; GPTQ; AWQ; other methods | GPTQ or AWQ for GPU serving; GGUF files for laptops and desktops |
| Granularity | One scale per matrix, per row, or per group | Per group of 32 to 128 weights |

## Quality vs speed

Every bit removed saves memory and time, but each step down adds more rounding error to every weight. This section shows where the trade-off usually lands and how to check it for a particular model, instead of trusting the file name.

### The trade-off for a 7-billion-weight model

Sizes include one 16-bit scale per group of 32 weights; speeds are the upper limit on the same illustrative GPU (2,000 GB/s) used earlier. The quality column describes the pattern widely reported for models of this size, not a measurement.

| Format | Bits per weight | Size | Max tokens per second | Typical quality compared with BF16 |
| --- | --- | --- | --- | --- |
| BF16 | 16 | 14.0 GB | 143 | The reference |
| INT8, groups of 32 | 8.5 | 7.4 GB | 269 | Practically the same |
| INT4, groups of 32 | 4.5 | 3.9 GB | 508 | Slightly worse; usually acceptable with GPTQ, AWQ or a good GGUF type |
| 3-bit | 3.5 | 3.1 GB | 653 | Noticeably worse |
| 2-bit | 2.5 | 2.2 GB | 914 | Often badly damaged |

The pattern: going from 16 to 8 bits is almost free, 4 bits is the usual sweet spot, and below 4 bits quality falls quickly, because with 8 or 4 integer levels the rounding errors become a large fraction of every weight (compare the INT8 and INT4 tables in "How quantization works").

### When quality drops noticeably

- **Below 4 bits per weight**, as in the table.
- **Small models.** A small model has fewer weights doing each job, so each rounding error matters more. Large models tolerate low bit counts better.
- **Plain rounding instead of a smart method.** GPTQ, AWQ and the better GGUF types exist because naive rounding at 4 bits loses noticeably more quality.
- **Demanding tasks.** Long chains of reasoning, maths and code tend to suffer first, because one slightly wrong token early can derail the rest of the answer ([Chapter 3](03-inference.md#generation-step-by-step)). Short, simple answers are more robust.

### A bigger model at low precision often beats a smaller one at high precision

With a fixed memory budget, the choice is often between a small model at 16 bits and a larger model at 4 bits. A 7-billion-weight model at 16 bits needs 14 GB; a 70-billion-weight model at 4.5 bits needs 70,000,000,000 × 4.5 ÷ 8 ≈ 39 GB. For a given amount of memory, a larger model quantized to about 4 bits usually gives better answers than a smaller model kept at 16 bits. This is one reason 4-bit models are so popular for running on personal computers.

### How to check a quantized model

Do not rely on the label: measure the quantized model against the original on the same inputs.

1. **Perplexity** on a sample of text ([Chapter 2](02-training.md#perplexity-the-same-number-in-friendlier-units)): a small rise means a small loss of quality. Both models must use the same tokenizer, which is always true here, since quantization never changes the tokenizer.
2. **Task results** on examples from the real use case, such as the fraud labels of [Chapter 5](05-evaluation.md). A model that is "only slightly worse" on average can still be worse exactly where it matters.
3. **Speed and memory** on the real hardware: the integer formats only pay off if the software and GPU handle them efficiently.
