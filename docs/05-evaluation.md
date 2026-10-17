# Evaluation: How We Measure a Model

## The big picture

The earlier chapters explained how a model is built ([Chapter 1](01-transformers.md)), trained ([Chapter 2](02-training.md)), used ([Chapter 3](03-inference.md)) and shrunk ([Chapter 4](04-quantization.md)). Each step raised the same question: is the result any good? This chapter explains how that question is answered, from the single number that training watches to the business results that justify a project.

The chapter uses one running example from start to finish:

!!! note "The fraud example"
    A bank uses an LLM to label card transactions as **fraud** or **not fraud**. For each transaction, the prompt describes it (amount, merchant, country, time) and the model answers `FRAUD` or `OK`. The probability the model gives to the token `FRAUD` ([Chapter 1](01-transformers.md#from-the-last-vector-back-to-a-word)) serves as its **fraud score**, between 0 and 1.

    - **Evaluation sample:** 100 past transactions with known answers, deliberately chosen to contain 10 fraud cases and 90 genuine ones, so there are enough fraud cases to measure.
    - **Real traffic:** 100,000 transactions a month, of which 2% (2,000) are fraud.

    All numbers in this example are made up for illustration.

## Three levels of evaluation

A model can be measured at three levels, and each level answers a different question for a different audience. Confusing them is the most common evaluation mistake: a model can improve at one level and get worse at another.

| Level | Question it answers | Typical measures | Who uses it | Fraud example |
| --- | --- | --- | --- | --- |
| 1. Training loss | Is the model learning? | Cross-entropy loss, perplexity | The people training the model | The loss falls during fine-tuning on labelled transactions |
| 2. Model metrics | Does it do the task well? | Accuracy, precision, recall, F1, AUC; benchmarks; human ratings | Data scientists, ML engineers | 8 of the 10 fraud cases in the sample are caught |
| 3. Business outcomes | Is it worth it? | Money saved, hours saved, customer complaints, return on investment | Managers, the business owner | Fraud losses fall by half; analysts spend 311 fewer hours a month on reviews |

The levels are connected but not interchangeable:

- **A lower loss usually means better metrics, but not always.** The loss rewards predicting every token well; a metric only checks the final answer.
- **Better metrics do not guarantee better business results.** A model that catches more fraud can also block more genuine customers, and the extra complaints can cost more than the fraud saved.
- **Each level needs the one below it.** Business results cannot be estimated without model metrics, and model metrics depend on a model that has learned.

The rest of the chapter climbs the three levels in order: loss and perplexity, then model metrics for classification and for generated text, then business outcomes, and finally a checklist for choosing metrics for a project.

## Loss vs metric

Training optimises a **loss**, but people judge a model by a **metric**. The two are often confused because both are "a number that says how good the model is". Knowing why they differ explains why training never optimises accuracy directly, and why a good loss alone does not prove a model is useful.

### Definitions

- **Loss:** a number the training process makes smaller. It must change smoothly when the weights change slightly, so that a gradient can be computed ([Chapter 2](02-training.md#the-gradient-which-way-to-move-each-weight)). For LLMs it is the [cross-entropy](02-training.md#the-loss-cross-entropy): −ln of the probability given to the correct token.
- **Metric:** a number that measures what people actually care about, such as the share of correct answers. It does not need to change smoothly, because no gradient is ever computed from it.

### Worked example: four fraud cases

Four transactions in the sample are fraud. The model answers `FRAUD` when its fraud score is at least 0.5:

| Transaction | Fraud score (probability of `FRAUD`) | Loss = −ln(score) | Model's answer | Correct? |
| --- | --- | --- | --- | --- |
| 1 | 0.95 | 0.051 | FRAUD | Yes |
| 2 | 0.60 | 0.511 | FRAUD | Yes |
| 3 | 0.55 | 0.598 | FRAUD | Yes |
| 4 | 0.45 | 0.799 | OK | No |

- **The loss sees every difference.** Transaction 1 (0.95) and transaction 2 (0.60) are both correct, but the loss is ten times lower for the confident one. Raising transaction 4 from 0.45 to 0.46 lowers its loss a little, so training gets a useful signal.
- **The accuracy metric sees only right or wrong.** Transactions 1 to 3 count the same. Moving transaction 4 from 0.45 to 0.46 changes nothing; moving it from 0.49 to 0.51 suddenly flips it to correct. Such a jumpy number has no useful gradient, so it cannot be trained on directly.

### Loss and metric side by side

| | Loss | Metric |
| --- | --- | --- |
| Purpose | Guide training | Judge the result |
| Computed on | Every token, during training | Final answers, on test data |
| Must change smoothly? | Yes, to compute gradients | No |
| Example | Cross-entropy 0.383 per token | 92% of transactions labelled correctly |
| Read by | The optimiser | People |

The practical rule: watch the loss to check that training works, but decide whether a model is good with metrics measured on data the model never saw during training. The next section turns the loss into a friendlier number, perplexity; the sections after it cover the metrics.

## Perplexity

Perplexity is the standard way to report a language model's loss as a number people can picture. It is the main measure used during pre-training and when comparing quantized models with the original ([Chapter 4](04-quantization.md#how-to-check-a-quantized-model)), so it is worth knowing exactly what it means and when it can be compared.

### Definition

**Perplexity** is e raised to the average cross-entropy loss per token:

$$
\text{perplexity} = e^{\bar{L}} \qquad \bar{L} = \frac{1}{N}\sum_{i=1}^{N} -\ln p_i
$$

- **L̄** is the average loss over the N tokens of a text.
- **p\_i** is the probability the model gave to the correct token at position i.
- **e** is the base of the natural logarithm (about 2.718).

It can be read as **"how many tokens the model is choosing between, on average"**. A perplexity of 4 means the model is as uncertain as if it were picking evenly among 4 tokens at every step. A perfect model, which always gives the correct token probability 1, has perplexity 1.

### Worked examples with the running numbers

| Case | Average loss | Perplexity | Reading |
| --- | --- | --- | --- |
| Chapter 1's example: CAT given 0.772 | −ln 0.772 = 0.259 | e^0.259 = 1.30 | Almost sure: about 1.3 choices |
| [Chapter 2's translation](02-training.md#from-one-position-to-a-whole-batch): 4 positions | 0.383 | e^0.383 = 1.47 | About 1.5 choices |
| A model guessing evenly among 4 tokens (each 0.25) | −ln 0.25 = 1.386 | e^1.386 = 4.00 | Exactly 4 choices |
| A perfect model (every correct token gets 1) | 0 | e^0 = 1 | No hesitation |

The third row shows why the reading works: when every one of k choices has probability 1/k, the perplexity is exactly k.

### What a good value is

There is no universal good value: perplexity depends on the model, the text and the tokenizer. A model will have a low perplexity on simple, predictable text and a higher one on technical or unusual text. Perplexity is therefore useful for **comparisons on the same text**: two checkpoints during training, a model before and after fine-tuning, or a quantized model against the original. A lower value on the same text means the model predicts that text better.

### Only comparable with the same tokenizer

Perplexity is an average **per token**, so it depends on how the text was cut into tokens ([Chapter 1](01-transformers.md#bpe-tokenizer-from-text-to-token-ids)). Suppose two models give a sentence exactly the same total probability, so the same total loss of 3.0 (made-up numbers), but their tokenizers split it differently:

| Model | Tokens for the sentence | Average loss | Perplexity |
| --- | --- | --- | --- |
| A | 6 | 3.0 ÷ 6 = 0.5 | e^0.5 = 1.65 |
| B | 10 | 3.0 ÷ 10 = 0.3 | e^0.3 = 1.35 |

Model B looks better, but both models assign the text exactly the same probability: B only spreads the same loss over more, smaller tokens. Perplexities can be compared only between models that share a tokenizer. This is always true for a quantized model and its original, because quantization never changes the tokenizer.

### What perplexity does not tell you

Perplexity measures how well a model predicts text, not whether its answers are correct, helpful or safe. A model can have a low perplexity and still answer `OK` to an obvious fraud. Task metrics, covered next, are needed for that.
