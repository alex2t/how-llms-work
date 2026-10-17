# Introduction: Foundations and History

## The big picture

These notes explain how large language models work, with real arithmetic at every step. Before [Chapter 1](01-transformers.md) opens up the Transformer, this introduction does two things:

1. **Foundations.** It explains the small amount of maths and machine learning the rest of the notes rely on: vectors, matrices, the dot product, and what a neural network, a loss and an embedding are.
2. **History.** It tells the story of how computers learned to model text, from one-hot vectors to Word2Vec to recurrent networks, and why each approach hit a wall that the Transformer removed.

Readers who already know linear algebra and basic machine learning can skip to the history sections, or go straight to [Chapter 1](01-transformers.md).

## What you need to know first

Everything a language model does is arithmetic on lists and tables of numbers. This section explains those lists and tables, and the few operations the notes use on them, each with a tiny example you can check by hand.

### Vectors

A **vector** is an ordered list of numbers. In language models, each token is represented by a vector. In Chapter 1's example, the token CAT is represented (in shortened form) by the vector \[0.81, 0.10, −0.52\]: 3 numbers, so its **length** or **dimension** is 3. In the real base model each token's vector has 512 numbers, the value called **d\_model**.

### The dot product: a measure of similarity

The **dot product** of two vectors of the same length multiplies them number by number and adds the results. It produces a single number that is large when the two vectors point in a similar direction, near 0 when they are unrelated, and negative when they point in opposite directions.

**Worked example** with three token vectors from [Chapter 1's embedding table](01-transformers.md#token-embeddings) (made-up numbers):

| Pair | Number-by-number products | Dot product |
| --- | --- | --- |
| CAT \[0.81, 0.10, −0.52\] · DOG \[0.77, 0.15, −0.48\] | 0.6237 + 0.0150 + 0.2496 | **0.888** |
| CAT \[0.81, 0.10, −0.52\] · A \[−0.05, 0.08, 0.91\] | −0.0405 + 0.0080 − 0.4732 | **−0.506** |

**How each result is computed.** Pair the numbers by position (1st with 1st, 2nd with 2nd, 3rd with 3rd), multiply each pair, then add the three products.

CAT · DOG:

| Position | CAT | DOG | Multiply | Product |
| --- | --- | --- | --- | --- |
| 1st | 0.81 | 0.77 | 0.81 × 0.77 | 0.6237 |
| 2nd | 0.10 | 0.15 | 0.10 × 0.15 | 0.0150 |
| 3rd | −0.52 | −0.48 | −0.52 × −0.48 | +0.2496 |
| **Sum** | | | 0.6237 + 0.0150 + 0.2496 | **0.8883 ≈ 0.888** |

CAT · A:

| Position | CAT | A | Multiply | Product |
| --- | --- | --- | --- | --- |
| 1st | 0.81 | −0.05 | 0.81 × −0.05 | −0.0405 |
| 2nd | 0.10 | 0.08 | 0.10 × 0.08 | +0.0080 |
| 3rd | −0.52 | 0.91 | −0.52 × 0.91 | −0.4732 |
| **Sum** | | | −0.0405 + 0.0080 − 0.4732 | **−0.5057 ≈ −0.506** |

**The sign rule decides the result.** Two numbers with the same sign multiply to a positive product; numbers with opposite signs give a negative one. In CAT · DOG, all three positions have matching signs (+ with +, + with +, − with −), so every product is positive and they add up to a large total. In CAT · A, the first and third positions have opposite signs, so two of the products are negative. The third one (−0.4732) is the largest, so the total ends up negative. Positions where both numbers are large count the most; that is why the 1st position dominates CAT · DOG and the 3rd dominates CAT · A.

CAT and DOG, words used in similar ways, have a large positive dot product; CAT and the article A have a negative one. Attention in Chapter 1 is built on exactly this operation: it uses dot products to decide how much each token should look at each other token.

### Matrices and their shapes

A **matrix** is a table of numbers with rows and columns. Its **shape** is written rows × columns. Stacking the vectors of a 6-token sentence, each with 512 numbers, gives a 6 × 512 matrix: one row per token.

### Matrix multiplication

To multiply matrix A by matrix B, take the dot product of **each row of A** with **each column of B**. The result has as many rows as A and as many columns as B. The rule for shapes:

$$
(m \times n) \cdot (n \times p) = (m \times p)
$$

The inner numbers (n) must match, because each dot product pairs a row of n numbers with a column of n numbers. **m**, **n** and **p** are just the sizes of the matrices.

**Worked example: a 2 × 3 matrix times a 3 × 2 matrix.**

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \qquad B = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 2 & 1 \end{bmatrix}
$$

| Result cell | Row of A · column of B | Value |
| --- | --- | --- |
| Row 1, column 1 | \[1, 2, 3\] · \[1, 0, 2\] = 1 + 0 + 6 | 7 |
| Row 1, column 2 | \[1, 2, 3\] · \[0, 1, 1\] = 0 + 2 + 3 | 5 |
| Row 2, column 1 | \[4, 5, 6\] · \[1, 0, 2\] = 4 + 0 + 12 | 16 |
| Row 2, column 2 | \[4, 5, 6\] · \[0, 1, 1\] = 0 + 5 + 6 | 11 |

$$
A \cdot B = \begin{bmatrix} 7 & 5 \\ 16 & 11 \end{bmatrix} \qquad (2 \times 3) \cdot (3 \times 2) = (2 \times 2)
$$

In Chapter 1 the same rule turns a sentence's 6 × 512 matrix into its queries: (6 × 512) · (512 × 64) = 6 × 64, one 64-number query per token ([Where Q, K and V come from](01-transformers.md#where-q-k-and-v-come-from)).

### The transpose

The **transpose** of a matrix, written with a superscript T, swaps its rows and columns: row 1 becomes column 1, and so on. The 2 × 3 matrix A from the example above becomes 3 × 2:

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \longrightarrow \quad A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
$$

| In A | In Aᵀ |
| --- | --- |
| Row 1: 1, 2, 3 | Column 1: 1, 2, 3 |
| Row 2: 4, 5, 6 | Column 2: 4, 5, 6 |

Each number moves from row r, column c to row c, column r. For example, the 6 sits in row 2, column 3 of A, and in row 3, column 2 of Aᵀ.

The attention formula in Chapter 1 uses QKᵀ: transposing K turns its rows (one per token) into columns, so that multiplying Q by Kᵀ computes the dot product of every query with every key in one matrix multiplication.

### What a neural network is

A **neural network** is a function built from many simple steps, each with adjustable numbers called **weights**. The basic step, one **neuron**, does three things:

1. Multiply each input by a weight and add the results (a dot product).
2. Add one more number, the **bias**.
3. Apply an **activation function**, a simple non-linear rule. A common one is **ReLU**: keep the value if it is positive, otherwise output 0.

**Worked example:** inputs \[2, 3\], weights \[0.5, −1\], bias 1:

- Weighted sum: 2 × 0.5 + 3 × (−1) = 1 − 3 = −2. Add the bias: −2 + 1 = −1.
- ReLU(−1) = 0. The neuron outputs 0.

The same neuron as a picture. The neuron has two inputs, so it has two weights, one per input: the first input (2) goes with the first weight (0.5), the second input (3) with the second weight (−1).

```mermaid
flowchart LR
  x1["input 2"] --> m1["2 × 0.5 = 1<br/>(input × its weight 0.5)"]
  x2["input 3"] --> m2["3 × (−1) = −3<br/>(input × its weight −1)"]
  b["bias 1<br/>(not multiplied)"] --> S
  m1 --> S["add everything<br/>1 + (−3) + 1 = −1"]
  m2 --> S
  S --> R["ReLU(−1) = 0<br/>(negative, so 0)"] --> O["output 0"]
```

Reading the picture from left to right:

| Step | What happens | Calculation | Result |
| --- | --- | --- | --- |
| 1 | First input times the first weight | 2 × 0.5 | 1 |
| 2 | Second input times the second weight | 3 × (−1) | −3 |
| 3 | Add the two products and the bias. The bias is not multiplied by any input: it is a fixed extra number added to the sum | 1 + (−3) + 1 | −1 |
| 4 | Apply ReLU: a negative value becomes 0 | ReLU(−1) | 0 |

With weights \[0.5, 1\] instead, the sum is 1 + 3 + 1 = 5 and the output is 5. Changing the weights changes what the neuron computes; that is all "learning" will mean.

### Where the weights and the bias come from

Nobody chooses the weights or the bias by hand. The 0.5, −1 and 1 above are made up for the example; in a real network, no person decides whether a weight should be 0.5 or 0.8. Every value is found automatically by **training**, in two phases.

**1. Start with random values.** Before training, every weight is set to a small random number, for example drawn between about −0.1 and 0.1. Biases usually start at 0. The values must be random rather than all equal: if every neuron in a layer started with the same weights, they would all compute the same thing, receive the same corrections, and stay identical forever.

**2. Correct them a little, many times.** Training shows the network an example whose correct answer is known, measures how wrong the output is (the **loss**), and moves every weight and the bias a small step in the direction that reduces that error. The size of each move depends on how much that weight contributed to the error.

**Worked example: one training step.** Take the neuron with weights \[0.5, 1\] and bias 1 (output 5), and suppose the correct answer for inputs \[2, 3\] is **3**. Measure the error as (output − target)², so the loss is (5 − 3)² = 4.

The output is 2 too high. Each value is moved in proportion to how strongly it pushed the output up:

| Value | Multiplied by input | Correction (error signal × input × step size) | New value |
| --- | --- | --- | --- |
| Weight 1 | 2 | 4 × 2 × 0.01 = 0.08 | 0.5 − 0.08 = **0.42** |
| Weight 2 | 3 | 4 × 3 × 0.01 = 0.12 | 1 − 0.12 = **0.88** |
| Bias | (always 1) | 4 × 1 × 0.01 = 0.04 | 1 − 0.04 = **0.96** |

- The **error signal** 4 is 2 × (5 − 3): it grows with the size of the error, and its sign says "the output is too high, so go down".
- Weight 2 moves the most because it multiplied the larger input (3): it was most responsible for the output being too high.
- The bias is treated like a weight whose input is always 1, so it is corrected in exactly the same way.
- The **step size** 0.01, called the **learning rate**, keeps each change small so training does not overshoot.

New output: 2 × 0.42 + 3 × 0.88 + 0.96 = 0.84 + 2.64 + 0.96 = 4.44. The loss falls from 4 to (4.44 − 3)² = 2.07. Repeating the same step:

| Step | Weights | Bias | Output | Loss |
| --- | --- | --- | --- | --- |
| 0 | 0.5, 1 | 1 | 5.00 | 4.00 |
| 1 | 0.42, 0.88 | 0.96 | 4.44 | 2.07 |
| 2 | 0.362, 0.794 | 0.931 | 4.04 | 1.08 |
| 3 | 0.321, 0.731 | 0.911 | 3.75 | 0.56 |
| 5 | 0.270, 0.654 | 0.885 | 3.39 | 0.15 |

After five steps the output is close to 3, with weights of about 0.27 and 0.65. Those final values were not chosen by anyone: they are simply where the corrections stopped. A real model does exactly this, but for millions or billions of weights at once and over billions of examples; a 7-billion-weight model has 7 billion numbers set this way. [Chapter 2](02-training.md#backpropagation-gradient-descent-and-the-optimiser) explains how the corrections are computed for every weight in a deep network (gradients and backpropagation).

A **layer** is many neurons working side by side on the same inputs, which is a matrix multiplication followed by the activation. A network stacks layers, each feeding the next. The Transformer's feed-forward network ([Chapter 1](01-transformers.md#the-feed-forward-network-in-detail)) is exactly two such layers.

Without the activation, stacking layers would be pointless: several matrix multiplications in a row equal one matrix multiplication, which can only describe straight-line relationships. The non-linear activation is what lets a deep network learn complicated patterns.

### Loss and training

- **Loss:** one number that measures how wrong the network's outputs are on some examples. Lower is better.
- **Training:** repeatedly adjusting every weight by a tiny amount in the direction that lowers the loss, over many examples.

[Chapter 2](02-training.md) explains both in detail with real numbers: the loss used for language (cross-entropy), how the direction to move each weight is found (gradients and backpropagation), and how each step is taken.

### Embeddings

An **embedding** is a learned vector that represents an item, such as a word or a token. A model keeps a table with one row per item; to represent an item, it reads that item's row. The numbers in the table start random and are adjusted by training, so items used in similar ways end up with similar vectors, as CAT and DOG did in the dot-product example. The rest of this introduction tells the story of how such vectors were invented, and [Chapter 1](01-transformers.md#token-embeddings) shows how a Transformer uses them.

## Historical evolution of text modelling

Today's chat assistants are the result of about a decade of steps, each fixing a limit of the one before. Knowing the order of those steps explains why the Transformer was such a turning point, and why the rest of this introduction looks at older methods at all.

### Timeline

| Year | Step | What changed |
| --- | --- | --- |
| 1997 | LSTM | A recurrent network that can remember information over longer stretches of text (explained later in this introduction) |
| 2013 | Word2Vec | Words get learned vectors in which similar words are close together |
| 2014 | Sequence-to-sequence with attention | Recurrent encoder–decoder networks translate whole sentences; attention lets the decoder look back at the source words |
| 2017 | The Transformer, "Attention Is All You Need" | Recurrence is removed entirely; attention alone, computed in parallel ([Chapter 1](01-transformers.md#attention-is-all-you-need-2017)) |
| 2018 | BERT and GPT | One large model is pre-trained on plain text, then adapted to many tasks |
| 2019 | GPT-2 | 1.5 billion weights; it writes surprisingly fluent paragraphs |
| 2020 | GPT-3 | 175 billion weights; it can do new tasks from a few examples in the prompt, without retraining |
| 2022 | ChatGPT | Fine-tuning and preference training ([Chapter 2](02-training.md#stages-of-an-llms-life-pre-training-fine-tuning-and-preference-training)) turn a text predictor into an assistant |
| 2023–2026 | Modern assistants and agents | Chat models such as Claude handle long documents and images, use tools, and act as coding agents that write, run and fix code over many steps |

### The task-specific era (the 2010s)

Before the Transformer, natural language processing meant building **a separate model for each task**:

- one model for **sentiment analysis** (is this review positive or negative?),
- another for **translation** (English to German),
- another for **named-entity recognition** (find the people, places and companies in a text).

Each was usually a recurrent neural network (RNN) trained from scratch on its own labelled data, often a few hundred thousand examples. Nothing learned by the sentiment model helped the translation model. Labelled data was expensive, so each model saw only a small slice of language, and every new task meant starting again.

### The turning point (2017)

The Transformer replaced recurrence with attention ([Chapter 1](01-transformers.md)). The important consequence was not only better translation: because every position is processed **in parallel** during training ([Chapter 2](02-training.md#why-training-happens-in-one-time-step)), Transformers could be trained on far more data with far more weights than recurrent networks, using the same hardware. One architecture now **scaled**: more data, more weights and more compute reliably gave a better model.

### Scaling to modern LLMs (2018–2026)

From 2018, the approach changed from "one model per task" to "**one model for all tasks**":

1. **Pre-train** one large model on huge amounts of unlabelled text, with the simple goal of predicting the next token. No human labels are needed ([Chapter 2](02-training.md#three-ideas-weights-prediction-error)).
2. **Adapt** it to each use: by fine-tuning on a small labelled set, by giving examples in the prompt, or simply by asking in plain language.

Model sizes grew very fast:

| Model | Year | Weights | Compared with the 2017 base model |
| --- | --- | --- | --- |
| Transformer (base) | 2017 | 65 million | 1× |
| GPT-2 | 2019 | 1.5 billion | about 23× |
| GPT-3 | 2020 | 175 billion | about 2,700× |

Larger models trained on more text kept getting better, and new abilities appeared along the way: following instructions, writing code, explaining their reasoning. After preference training made them helpful assistants (2022), the next steps were longer contexts, tool use and **agents**: models that plan, call tools, read the results and continue over many steps, such as coding agents that work inside a software project.

The sections below go back to the beginning of this story, to see the problems these steps solved: how words were first turned into vectors, and why the recurrent networks of the task-specific era could not scale.

## Vector embeddings: static representations

A model can only compute with numbers, so every word must first become a vector. How that vector is chosen decides what the model can know about the word. This section follows the two classic answers: one-hot vectors, which carry no meaning at all, and Word2Vec, the first widely used learned vectors, which carry meaning but still have serious limits.

### One-hot encoding

**One-hot encoding** gives each word in the vocabulary its own position. The **vocabulary** is the fixed list of words the model knows, decided before training. Every word's vector has one slot per word in that list, with a 1 in the word's own slot and 0 in all the others. So every vector has exactly as many numbers as there are words in the vocabulary, and all words have vectors of the same length: with a 5-word vocabulary each vector has 5 numbers; with a 50,000-word vocabulary each vector has 50,000 numbers, one 1 and 49,999 zeros.

**Worked example** with a 5-word vocabulary: cat, dog, bank, river, money.

| Word | One-hot vector |
| --- | --- |
| cat | \[1, 0, 0, 0, 0\] |
| dog | \[0, 1, 0, 0, 0\] |
| bank | \[0, 0, 1, 0, 0\] |
| river | \[0, 0, 0, 1, 0\] |
| money | \[0, 0, 0, 0, 1\] |

**Do these vectors know that cat and dog are similar?** A useful word vector should place words with related meanings close together: cat should be closer to dog (both animals) than to bank. The [dot product](#the-dot-product-a-measure-of-similarity) tests this: a large result means similar, 0 means unrelated. Compute it for cat and dog by lining the two vectors up slot by slot, multiplying each pair and adding:

| Slot | 1 (cat) | 2 (dog) | 3 (bank) | 4 (river) | 5 (money) | Sum |
| --- | --- | --- | --- | --- | --- | --- |
| cat | 1 | 0 | 0 | 0 | 0 | |
| dog | 0 | 1 | 0 | 0 | 0 | |
| cat × dog | 1 × 0 = 0 | 0 × 1 = 0 | 0 × 0 = 0 | 0 × 0 = 0 | 0 × 0 = 0 | **0** |

Every product is 0, because cat's 1 sits in slot 1 and dog's 1 sits in slot 2: wherever one vector has its 1, the other has a 0. The same thing happens for cat and bank, and for **every** pair of different words, since no two words share a slot. Only a word with itself scores 1 (cat · cat = 1 × 1 = 1).

| Pair | Dot product | What it says |
| --- | --- | --- |
| cat · cat | 1 | Same word |
| cat · dog | 0 | Unrelated |
| cat · bank | 0 | Unrelated, exactly like cat and dog |

The **distance** between two vectors (how far apart they are) tells the same story. It is computed by subtracting the vectors slot by slot, squaring each difference, adding, and taking the square root. For cat and dog: cat − dog = \[1, −1, 0, 0, 0\]; squares 1 + 1 + 0 + 0 + 0 = 2; distance √2 ≈ 1.414. For cat and bank: \[1, 0, −1, 0, 0\], again √2 ≈ 1.414. Every pair of different words is exactly 1.414 apart.

So with one-hot vectors, the model sees cat as no closer to dog than to bank or money. The vectors are **orthogonal** (at right angles to each other): "cat" is exactly as far from "dog" as from "bank". One-hot vectors identify words but say nothing about their meaning. They are also wasteful: with a 50,000-word vocabulary, every word needs 50,000 numbers, all but one of them 0.

### Word2Vec: learning vectors from context

**Word2Vec** (Mikolov and colleagues at Google, 2013) learns a short, dense vector for every word, typically 300 numbers, with the idea that **a word is known by the company it keeps**: words that appear in similar contexts should get similar vectors.

It learns them with a **proxy task**, also called a practice task. The model is trained to guess missing words from their neighbours. Nobody needs these guesses: they are never used for anything. The task is there for one reason: to get good at it, the model must give similar vectors to words that appear in similar places, and those vectors are what we want. It is like practising scales on a piano: nobody listens to the scales, but they build the skill.

Here is the guessing task on the sentence "the cat sat on the mat". The **context window** is the set of neighbours the model looks at; here it is 2 words on each side of the word "sat". Word2Vec comes in two versions, which guess in opposite directions:

| Proxy task | Input | Prediction target |
| --- | --- | --- |
| **CBOW** (continuous bag of words) | The context: the, cat, on, the | The middle word: sat |
| **Skip-gram** | The middle word: sat | Each context word: the, cat, on, the |

Training slides this window over billions of words of text and adjusts the vectors to make these predictions better, just as training adjusts weights to lower a loss ([Chapter 2](02-training.md#what-training-is)). Because "cat" and "dog" appear among the same neighbours (the, sat, fed, pet), they end up with similar vectors. The prediction task is then thrown away; the vectors are the product.

**The network behind it.** Word2Vec is a very small [neural network](#what-a-neural-network-is) with three parts:

| Part | What it holds | Size (50,000-word vocabulary) |
| --- | --- | --- |
| Input | The word as a one-hot vector | 50,000 numbers |
| Hidden layer | The word's vector | 300 numbers |
| Output | One score per vocabulary word: how likely each word is to be a neighbour | 50,000 numbers |

The first weight matrix, between the input and the hidden layer, has one row per vocabulary word and 300 columns. Multiplying a one-hot vector by it simply **picks out one row**, because every 0 in the one-hot vector cancels its row and the single 1 keeps its row.

**Example** with the 5-word vocabulary from above and 3 numbers per word instead of 300 (made-up weights). The one-hot vector for *dog* is \[0, 1, 0, 0, 0\]:

| Row of the weight matrix | Weights | Multiplied by dog's one-hot number | Result |
| --- | --- | --- | --- |
| 1 (cat) | \[0.2, 0.8, −0.1\] | × 0 | \[0, 0, 0\] |
| 2 (dog) | \[0.3, 0.7, 0.0\] | × 1 | \[0.3, 0.7, 0.0\] |
| 3 (bank) | \[−0.6, 0.1, 0.9\] | × 0 | \[0, 0, 0\] |
| 4 (river) | \[−0.4, 0.2, 0.5\] | × 0 | \[0, 0, 0\] |
| 5 (money) | \[−0.5, −0.3, 0.8\] | × 0 | \[0, 0, 0\] |
| **Sum** | | | **\[0.3, 0.7, 0.0\]**, row 2 |

So the hidden layer for *dog* is just row 2. **That row is dog's vector**, and the whole first weight matrix is the table of word vectors (the **embedding table**). When training adjusts these weights, it is adjusting the word vectors.

**Negative sampling: a cheaper way to train.** The output layer is expensive. To check one guess, the network would compute a score for every word in the vocabulary, 50,000 scores, and turn them into probabilities with a softmax. Doing this for billions of training examples is too slow.

**Negative sampling** replaces that one big question ("which of the 50,000 words is the neighbour?") with a few yes/no questions:

1. Take a real pair from the text, for example *cat* and its neighbour *sat*. The correct answer to "Is *sat* a neighbour of *cat*?" is **yes**.
2. Draw a few random words from the vocabulary, for example *bank* and *river*. These are the **negatives**: random words are almost never real neighbours, so the correct answer for each is **no**. Word2Vec typically uses 2 to 20 negatives.
3. For each pair, the model's answer is a probability of "yes", computed from the dot product of the two words' vectors and passed through the **sigmoid** function σ(x) = 1 / (1 + e⁻ˣ). The sigmoid turns any number into a value between 0 and 1: a large positive dot product gives a probability close to 1 (yes), a large negative one gives a probability close to 0 (no).
4. The loss measures how wrong each answer is: −ln(probability) for a "yes" pair, −ln(1 − probability) for a "no" pair. Each term is near 0 when the answer is right and grows when it is wrong.

**Worked example** (made-up dot products), centre word *cat*, one true neighbour and two negatives:

| Pair | Should be | Dot product | σ(dot product) | Loss term |
| --- | --- | --- | --- | --- |
| cat, sat | Yes | 2.0 | 0.881 | −ln 0.881 = 0.127 |
| cat, bank | No | −1.0 | 0.269 | −ln(1 − 0.269) = 0.313 |
| cat, river | No | 0.5 | 0.622 | −ln(1 − 0.622) = 0.974 |

The total loss is 0.127 + 0.313 + 0.974 = 1.414. The largest term comes from *river*, a random word the model currently thinks is a likely neighbour; training pushes the vectors of *cat* and *river* apart and those of *cat* and *sat* together. Each example now needs 3 dot products instead of 50,000.

### Vector arithmetic: Paris − France + Germany ≈ Berlin

The learned space has a striking property: **directions carry meaning**. The step from a country to its capital points in roughly the same direction for every country, so it can be computed with ordinary vector arithmetic.

**Illustrative 2-D example** (made-up vectors; real Word2Vec vectors have about 300 numbers):

| Word | Vector |
| --- | --- |
| France | (1, 1) |
| Paris | (1, 3) |
| Germany | (4, 1) |
| Berlin | (4.1, 2.9) |
| Italy | (7, 1) |
| Rome | (7, 3.1) |

1. Paris − France = (1 − 1, 3 − 1) = (0, 2): the "capital of" direction.
2. Add Germany: (0, 2) + (4, 1) = **(4, 3)**.
3. Find the word whose vector is closest to (4, 3):

| Word | Distance from (4, 3) |
| --- | --- |
| **Berlin** | **0.14** |
| Germany | 2.00 |
| Paris | 3.00 |
| Rome | 3.00 |
| France, Italy | 3.61 |

The nearest word is Berlin. Nobody told Word2Vec what a capital is: the relationship emerged because capitals and their countries appear in similar contexts across many texts.

### The limits of Word2Vec

Word2Vec vectors are **static**: each word has exactly one vector, fixed after training, used in every sentence. That causes two problems.

- **No polysemy.** **Polysemy** means one word having several meanings. "Bank" in "the river bank" and in "the bank raised its rates" gets the same vector, a blurred average of both meanings. The model cannot tell which one a sentence means.
- **No word order.** The vectors of "dog bites man" and "man bites dog" are the same three vectors. Word vectors alone say what words mean in general, not what a particular sentence says.

The Transformer keeps the idea of a learned vector per token, its embedding table ([Chapter 1](01-transformers.md#token-embeddings)), but then lets attention change each token's vector according to the other tokens in the sentence. The starting vector of "bank" is still fixed, as [Chapter 1 notes](01-transformers.md#what-is-learned-and-what-changes-after-training); attention is what turns it into "river bank" or "money bank". Word order is added through positional encoding. Before that, the first attempt to read sentences in order was the recurrent network, described next.

## Sequence architectures: RNNs and LSTMs

Word vectors say what words mean in general; to understand a sentence, a model must also read the words in order and combine them. Recurrent networks were the standard way to do that for most of the 2010s. Seeing how they work, and where they fail, explains exactly what the Transformer changed.

### The recurrent neural network (RNN)

A **recurrent neural network (RNN)** reads a sequence one token at a time. It keeps a **hidden state**: a vector that acts as its memory of everything read so far. At each step it combines the new token with the previous hidden state to make the next hidden state:

$$
h_t = \tanh\left(W_x\, x_t + W_h\, h_{t-1} + b\right)
$$

- **x\_t** is the vector of the token at step t.
- **$h_{t-1}$** is the hidden state from the previous step (all zeros before the first step), and **h\_t** the new one.
- **W\_x** and **W\_h** are weight matrices, and **b** a bias. The **same** weights are used at every step: that reuse is what "recurrent" means.
- **tanh** is an activation function that squeezes any number into the range −1 to 1.

At each step the RNN can also produce an output y\_t from h\_t, for example a prediction of the next word.

```mermaid
flowchart LR
  x1[x1] --> R1[RNN step 1] --> y1[y1]
  x2[x2] --> R2[RNN step 2] --> y2[y2]
  x3[x3] --> R3[RNN step 3] --> y3[y3]
  R1 -- "hidden state h1" --> R2
  R2 -- "hidden state h2" --> R3
```

### Worked example: three steps

To keep the arithmetic readable, each token and the hidden state are single numbers instead of vectors (made-up values): W\_x = 0.5, W\_h = 0.8, b = 0, and the start state h₀ = 0. The three tokens of `YOUR CAT IS` are represented by the numbers 1.0, 0.5 and −0.5.

| Step | Token value x\_t | 0.5 × x\_t | 0.8 × $h_{t-1}$ | Sum | h\_t = tanh(sum) |
| --- | --- | --- | --- | --- | --- |
| 1 (YOUR) | 1.0 | 0.50 | 0.8 × 0 = 0.000 | 0.500 | 0.462 |
| 2 (CAT) | 0.5 | 0.25 | 0.8 × 0.462 = 0.370 | 0.620 | 0.551 |
| 3 (IS) | −0.5 | −0.25 | 0.8 × 0.551 = 0.441 | 0.191 | 0.189 |

The final state 0.189 is the network's whole memory of the three words. Each step had to wait for the previous one, because step 2 needs h₁ and step 3 needs h₂.

### The LSTM

The **LSTM** (long short-term memory, Hochreiter and Schmidhuber, 1997) is an RNN designed to remember for longer. Besides the hidden state, it keeps a **cell state**, a second memory vector, and three **gates**. Each gate is a set of numbers between 0 and 1, computed from the current token and the previous hidden state, that controls how much information flows:

- **Forget gate:** how much of the old cell state to keep. A value of 1 keeps everything; 0 erases it.
- **Input gate:** how much of the new token's information to write into the cell state.
- **Output gate:** how much of the cell state to reveal in the hidden state at this step.

The key is that the cell state is updated mostly by multiplying by the forget gate and adding new information, rather than being squeezed through tanh at every step. If the forget gate stays near 1, information survives for many steps: with a forget gate of 0.99, about 0.99⁵⁰ ≈ 0.61 of a stored value is still there after 50 steps. LSTMs powered most translation and speech systems from about 2014 to 2017.

### Why RNNs hit a wall

**1. Vanishing and exploding gradients.** Training needs gradients ([Chapter 2](02-training.md#the-gradient-which-way-to-move-each-weight)) that tell each weight how it affected the loss. In an RNN, the effect of an early word on a late prediction passes back through every step in between, and is multiplied by W\_h (times the slope of tanh, which is at most 1) at each step:

| Steps between the word and the prediction | Factor with W\_h = 0.8 (vanishing) | Factor with W\_h = 1.2 (exploding) |
| --- | --- | --- |
| 10 | 0.8¹⁰ = 0.107 | 1.2¹⁰ = 6.2 |
| 20 | 0.8²⁰ = 0.012 | 1.2²⁰ = 38.3 |
| 50 | 0.8⁵⁰ = 0.000014 | 1.2⁵⁰ ≈ 9,100 |

With a factor below 1, the signal from 50 words back almost disappears, so the network cannot learn from it (**vanishing gradient**). With a factor above 1, it grows so large that training becomes unstable (**exploding gradient**). LSTMs reduce the problem but do not remove it.

**2. Long-range dependencies are hard.** In "The keys that the man left on the kitchen table **are** missing", the verb "are" must agree with "keys", eight words earlier. Everything the network knows about "keys" must survive eight updates of one fixed-size hidden state, shared with everything else it is trying to remember. The longer the gap, the more likely it is lost.

**3. Sequential processing blocks parallel training.** Step t cannot start until step t − 1 has finished. A 1,000-token text needs 1,000 steps one after another, even on a GPU that could do thousands of calculations at once. This makes RNNs slow to train on large datasets, which kept them small.

**4. One fixed-size summary: the context bottleneck.** For translation, the 2014 sequence-to-sequence models used two RNNs: an **encoder** read the whole source sentence, and a **decoder** wrote the translation. The only thing passed between them was the encoder's **last hidden state**, one vector of fixed size. To translate "I love you very much", the decoder had to produce "Ti amo molto" from that single vector, with no way to look back at the individual English words. A 5-word sentence fits; a 30-word sentence must be squeezed into the same 512 numbers, and details are lost. Experiments at the time showed translation quality falling quickly as sentences got longer.

The fix, also from 2014, was the first use of **attention**: instead of one summary, the decoder keeps all the encoder's hidden states, one per source word, and at each output step computes a weight for each of them, then takes their weighted sum. When writing "molto", it can put most of its weight on the states for "very much". This is the same idea as attention in Chapter 1; the Transformer kept the attention and dropped the recurrence.

### The bridge to Chapter 1

The Transformer removes all four problems by dropping recurrence and using **attention**: every token looks directly at every other token ([Chapter 1](01-transformers.md#the-problem-it-solved)).

| RNN problem | How attention removes it |
| --- | --- |
| Vanishing and exploding gradients across many steps | Any two tokens are one attention step apart, so the gradient between them does not pass through a long chain of multiplications |
| Long-range dependencies squeezed into one hidden state | "are" can attend to "keys" directly, however far back it is |
| The whole source squeezed into one fixed-size vector | Every output step attends to every source token, so nothing has to fit in a single summary ([Chapter 1](01-transformers.md#self-attention-vs-cross-attention)) |
| Sequential processing | All positions are computed at once in training ([Chapter 2](02-training.md#why-training-happens-in-one-time-step)), so GPUs are fully used |

That last point is what made today's models possible: once training could use parallel hardware fully, models and datasets could grow by thousands of times, as the timeline above shows. [Chapter 1](01-transformers.md) explains how the Transformer works, step by step.

## Cheat sheet

| Concept | One-line meaning |
| --- | --- |
| Vector | An ordered list of numbers; one per token in a language model |
| Dot product | Multiply number by number and add: large when two vectors are similar |
| Matrix and shape | A table of numbers, described as rows × columns |
| Matrix multiplication | Row-by-column dot products; (m × n) · (n × p) = (m × p) |
| Transpose | Swap rows and columns; QKᵀ compares every query with every key |
| Neuron | Weighted sum + bias, then an activation such as ReLU |
| Layer and network | Many neurons side by side; layers stacked one after another |
| Loss and training | A number measuring error; adjust the weights to lower it |
| Embedding | A learned vector per item, read from a table |
| Task-specific era | One separately trained model per task, usually an RNN |
| Pre-train, then adapt | One large model trained on plain text, then used for many tasks |
| One-hot vector | A 1 at the word's position, 0 elsewhere: no notion of similarity |
| Word2Vec | Dense word vectors learned by predicting words from their context (CBOW) or context from a word (skip-gram) |
| Vector arithmetic | Paris − France + Germany ≈ Berlin: directions in the space carry meaning |
| Static embedding | One vector per word in every sentence: no polysemy, no word order |
| RNN | Reads tokens one at a time, carrying a hidden state as memory |
| LSTM | An RNN with a cell state and gates that keep information longer |
| Vanishing / exploding gradient | Signals shrink or grow when multiplied through many steps |
| Why the Transformer won | Attention connects any two tokens directly and trains all positions in parallel |
| Context bottleneck | An RNN encoder–decoder passes the whole source as one fixed-size vector; attention removed this limit |

## Sources

- [Deep Learning (Goodfellow, Bengio and Courville, 2016)](https://www.deeplearningbook.org/): linear algebra, neural networks and recurrent networks
- [Speech and Language Processing, 3rd edition draft (Jurafsky and Martin)](https://web.stanford.edu/~jurafsky/slp3/): word vectors, RNNs and LSTMs
- [Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781): Word2Vec, CBOW and skip-gram
- [Distributed Representations of Words and Phrases and their Compositionality (Mikolov et al., 2013)](https://arxiv.org/abs/1310.4546): word-vector arithmetic
- [Long Short-Term Memory (Hochreiter and Schmidhuber, 1997)](https://doi.org/10.1162/neco.1997.9.8.1735)
- [Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)](https://arxiv.org/abs/1409.0473): attention added to recurrent translation
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf): GPT
- [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): GPT-2
- [Language Models are Few-Shot Learners (Brown et al., 2020)](https://arxiv.org/abs/2005.14165): GPT-3
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361): why larger models and datasets keep improving results
- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches (Cho et al., 2014)](https://arxiv.org/abs/1409.1259): translation quality falls on long sentences
- [CME295: Transformers & Large Language Models, Lecture 1 (Stanford University, Afshine and Shervine Amidi)](https://cme295.stanford.edu/syllabus/): context bottleneck and Word2Vec training topics
