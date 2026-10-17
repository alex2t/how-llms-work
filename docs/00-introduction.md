# Introduction: Foundations and History

## The big picture

This book explains how large language models work, with real arithmetic at every step. Before [Chapter 1](01-transformers.md) opens up the Transformer, this introduction does two things:

1. **Foundations.** It explains the small amount of maths and machine learning the rest of the book relies on: vectors, matrices, the dot product, and what a neural network, a loss and an embedding are.
2. **History.** It tells the story of how computers learned to model text, from one-hot vectors to Word2Vec to recurrent networks, and why each approach hit a wall that the Transformer removed.

Readers who already know linear algebra and basic machine learning can skip to the history sections, or go straight to [Chapter 1](01-transformers.md).

## What you need to know first

Everything a language model does is arithmetic on lists and tables of numbers. This section explains those lists and tables, and the few operations the book uses on them, each with a tiny example you can check by hand.

### Vectors

A **vector** is an ordered list of numbers. In language models, each token is represented by a vector. In Chapter 1's example, the token CAT is represented (in shortened form) by the vector \[0.81, 0.10, −0.52\]: 3 numbers, so its **length** or **dimension** is 3. In the real base model each token's vector has 512 numbers, the value called **d\_model**.

### The dot product: a measure of similarity

The **dot product** of two vectors of the same length multiplies them number by number and adds the results. It produces a single number that is large when the two vectors point in a similar direction, near 0 when they are unrelated, and negative when they point in opposite directions.

**Worked example** with three token vectors from [Chapter 1's embedding table](01-transformers.md#token-embeddings) (made-up numbers):

| Pair | Number-by-number products | Dot product |
| --- | --- | --- |
| CAT \[0.81, 0.10, −0.52\] · DOG \[0.77, 0.15, −0.48\] | 0.6237 + 0.0150 + 0.2496 | **0.888** |
| CAT \[0.81, 0.10, −0.52\] · A \[−0.05, 0.08, 0.91\] | −0.0405 + 0.0080 − 0.4732 | **−0.506** |

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

The **transpose** of a matrix, written with a superscript T, swaps its rows and columns: row 1 becomes column 1, and so on. A 2 × 3 matrix becomes 3 × 2:

$$
A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
$$

The attention formula in Chapter 1 uses QKᵀ: transposing K turns its rows (one per token) into columns, so that multiplying Q by Kᵀ computes the dot product of every query with every key in one matrix multiplication.

### What a neural network is

A **neural network** is a function built from many simple steps, each with adjustable numbers called **weights**. The basic step, one **neuron**, does three things:

1. Multiply each input by a weight and add the results (a dot product).
2. Add one more number, the **bias**.
3. Apply an **activation function**, a simple non-linear rule. A common one is **ReLU**: keep the value if it is positive, otherwise output 0.

**Worked example:** inputs \[2, 3\], weights \[0.5, −1\], bias 1:

- Weighted sum: 2 × 0.5 + 3 × (−1) = 1 − 3 = −2. Add the bias: −2 + 1 = −1.
- ReLU(−1) = 0. The neuron outputs 0.

With weights \[0.5, 1\] instead, the sum is 1 + 3 + 1 = 5 and the output is 5. Changing the weights changes what the neuron computes; that is all "learning" will mean.

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

**One-hot encoding** gives each word in the vocabulary its own position. A word's vector has a 1 at its position and 0 everywhere else, so the vector is as long as the vocabulary.

**Worked example** with a 5-word vocabulary: cat, dog, bank, river, money.

| Word | One-hot vector |
| --- | --- |
| cat | \[1, 0, 0, 0, 0\] |
| dog | \[0, 1, 0, 0, 0\] |
| bank | \[0, 0, 1, 0, 0\] |
| river | \[0, 0, 0, 1, 0\] |
| money | \[0, 0, 0, 0, 1\] |

Now measure similarity with the dot product from the foundations section:

- cat · dog = 1×0 + 0×1 + 0×0 + 0×0 + 0×0 = **0**
- cat · bank = **0**
- The same holds for every pair of different words: the dot product is always 0, and the distance between any two words is always √2 ≈ 1.414.

The vectors are **orthogonal** (at right angles to each other): "cat" is exactly as far from "dog" as from "bank". One-hot vectors identify words but say nothing about their meaning. They are also wasteful: with a 50,000-word vocabulary, every word needs 50,000 numbers, all but one of them 0.

### Word2Vec: learning vectors from context

**Word2Vec** (Mikolov and colleagues at Google, 2013) learns a short, dense vector for every word, typically 300 numbers, with the idea that **a word is known by the company it keeps**: words that appear in similar contexts should get similar vectors.

It learns them with a **proxy task**: a simple prediction task nobody actually needs solved, set up only because solving it forces useful vectors to form. Take the sentence "the cat sat on the mat" and a **context window** of 2 words on each side of the word "sat":

| Proxy task | Input | Prediction target |
| --- | --- | --- |
| **CBOW** (continuous bag of words) | The context: the, cat, on, the | The middle word: sat |
| **Skip-gram** | The middle word: sat | Each context word: the, cat, on, the |

Training slides this window over billions of words of text and adjusts the vectors to make these predictions better, just as training adjusts weights to lower a loss ([Chapter 2](02-training.md#what-training-is)). Because "cat" and "dog" appear among the same neighbours (the, sat, fed, pet), they end up with similar vectors. The prediction task is then thrown away; the vectors are the product.

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
