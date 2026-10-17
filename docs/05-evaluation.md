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

## Mean squared error (MSE)

Cross-entropy suits answers that are one choice out of many, such as the next token or a fraud label. When the answer is a **number**, such as an amount of money, the usual measure is the mean squared error. This section shows how it works and why it is the wrong tool for next-token prediction.

### Definitions

- **Regression:** predicting a number on a continuous scale (an amount, a temperature, a price), as opposed to **classification**, which predicts a category (fraud / not fraud, or one token from the vocabulary).
- **Error:** predicted value minus true value.
- **Mean squared error (MSE):** the average of the squared errors.

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} \left(\hat{y}_i - y_i\right)^2
$$

- **n** is the number of examples.
- **ŷ\_i** (y-hat) is the predicted value for example i, and **y\_i** the true value.

Squaring does two things: it makes every error positive, so errors above and below cannot cancel out, and it makes large errors count much more than small ones.

### Worked example: estimating the amount at risk

For each flagged transaction, the bank also asks the model to estimate how much money is at risk, in euros. Four cases, with made-up numbers:

| Case | True amount y | Predicted ŷ | Error ŷ − y | Squared error |
| --- | --- | --- | --- | --- |
| 1 | 120 | 100 | −20 | 400 |
| 2 | 80 | 90 | +10 | 100 |
| 3 | 300 | 240 | −60 | 3,600 |
| 4 | 50 | 55 | +5 | 25 |
| | | | **Sum** | 4,125 |
| | | | **MSE (÷ 4)** | **1,031.25** |

- **Large errors dominate.** Case 3's error of 60 contributes 3,600 of the 4,125 total, 87%, although it is only one case out of four.
- **The unit is squared.** An MSE of 1,031.25 is in "squared euros", which is hard to picture. Taking the square root gives the **root mean squared error (RMSE)**: √1,031.25 = 32.11 euros, a typical error size in the original unit.
- **A gentler alternative** is the **mean absolute error (MAE)**, the average of the errors without their signs: (20 + 10 + 60 + 5) ÷ 4 = 23.75 euros. It does not give large errors extra weight.

### Why MSE does not suit next-token prediction

MSE needs a meaningful distance between the predicted and the true value: 240 is closer to 300 than 100 is. Token IDs have no such distance. In the running example, CAT has ID 6587 and DOG has ID 4012 ([Chapter 1](01-transformers.md#what-is-a-token-id)). If the correct token is CAT, predicting the ID 6,500 is not "nearly right": 6,500 is just some other token, unrelated to cats. The IDs are labels, not quantities.

So for next-token prediction, the model outputs a probability for every token and is scored with cross-entropy, which only asks how much probability went to the correct token ([Chapter 2](02-training.md#the-loss-cross-entropy)).

| | Cross-entropy | MSE |
| --- | --- | --- |
| Kind of answer | A category (token, label) | A number on a scale |
| Model outputs | A probability for every category | One number |
| Measures | −ln of the probability of the correct category | Average squared distance from the true value |
| Used for | Next-token prediction, classification | Regression: amounts, scores, forecasts |

## Classification metrics: the fraud example

When a model sorts items into categories, its answers can be checked one by one against the truth. This section builds every common classification metric from a single table of counts, using the 100-transaction evaluation sample: 10 fraud cases and 90 genuine ones. The model answers `FRAUD` when its fraud score is at least 0.5.

### The confusion matrix

A **confusion matrix** counts the four possible outcomes of a yes/no prediction. Here "positive" means "fraud", the thing being looked for.

- **True positive (TP):** fraud, and the model said FRAUD. A catch.
- **False negative (FN):** fraud, but the model said OK. A missed fraud.
- **False positive (FP):** genuine, but the model said FRAUD. A false alarm.
- **True negative (TN):** genuine, and the model said OK.

| | Model says FRAUD | Model says OK | Total |
| --- | --- | --- | --- |
| **Really fraud** | TP = 8 | FN = 2 | 10 |
| **Really genuine** | FP = 6 | TN = 84 | 90 |
| **Total** | 14 | 86 | 100 |

The model caught 8 of the 10 frauds, missed 2, and raised 6 false alarms among the 90 genuine transactions. Every metric below is computed from these four numbers.

### Accuracy, and why it misleads

**Accuracy** is the share of all answers that are correct:

$$
\text{accuracy} = \frac{TP + TN}{\text{all cases}} = \frac{8 + 84}{100} = 0.92
$$

92% sounds good. But consider a useless model that answers `OK` to every transaction:

| Model | Frauds caught | Accuracy on the sample (10% fraud) | Accuracy on real traffic (2% fraud) |
| --- | --- | --- | --- |
| The LLM at threshold 0.5 | 8 of 10 | 92% | about 93% |
| "Always OK" | 0 | 90% | **98%** |

The model that never catches a single fraud scores 98% accuracy on real traffic, higher than the useful model. When one category is much rarer than the other, data is called **imbalanced**, and accuracy is dominated by the common category. Fraud, disease screening and defect detection are all imbalanced, so accuracy alone should never be used for them.

### Precision, recall and F1

Two metrics look only at the rare category, from two different sides:

$$
\text{precision} = \frac{TP}{TP + FP} = \frac{8}{8 + 6} = 0.571 \qquad \text{recall} = \frac{TP}{TP + FN} = \frac{8}{8 + 2} = 0.80
$$

- **Precision** answers: "when the model says FRAUD, how often is it right?" Here 57% of alerts are real fraud; 43% are false alarms.
- **Recall** (also called **sensitivity** or **true positive rate**) answers: "of all real frauds, how many did the model catch?" Here 80%.

The "always OK" model has a recall of 0: precision and recall expose it immediately, where accuracy did not.

**F1** combines the two into one number, using the **harmonic mean**, which stays low if either of the two is low:

$$
F_1 = \frac{2 \times \text{precision} \times \text{recall}}{\text{precision} + \text{recall}} = \frac{2\,TP}{2\,TP + FP + FN} = \frac{16}{16 + 6 + 2} = 0.667
$$

The second form, written with the counts, gives the same value and avoids rounding: with the rounded 0.571 the first form gives 0.666.

**When to favour each:**

| Favour | When mistakes of this kind are expensive | Example |
| --- | --- | --- |
| Recall | Missing a positive costs a lot | Fraud losses, missed disease: catch as many as possible, accept more false alarms |
| Precision | A false alarm costs a lot | Blocking a genuine customer's card, sending a case to a costly investigation |
| F1 | Both matter and there is no clear cost difference | Comparing models with one number |

### Moving the threshold: the ROC curve and AUC

The threshold of 0.5 is a choice, not a law. Lowering it flags more transactions: more frauds are caught, but more genuine ones are flagged too. The **ROC curve** (receiver operating characteristic, a name from radar engineering) shows this trade-off for every threshold at once. It plots two rates:

- **True positive rate (TPR)** = recall = TP ÷ 10 real frauds.
- **False positive rate (FPR)** = FP ÷ 90 genuine transactions: the share of genuine customers wrongly flagged.

The model's results on the sample at several thresholds (made-up but consistent with the matrix above):

| Threshold (flag if fraud score ≥) | TP | FP | TPR (recall) | FPR | Precision |
| --- | --- | --- | --- | --- | --- |
| 1.0 (flag nothing) | 0 | 0 | 0.00 | 0.000 | — |
| 0.9 | 4 | 1 | 0.40 | 0.011 | 0.800 |
| 0.7 | 6 | 3 | 0.60 | 0.033 | 0.667 |
| **0.5** | **8** | **6** | **0.80** | **0.067** | **0.571** |
| 0.3 | 9 | 15 | 0.90 | 0.167 | 0.375 |
| 0.1 | 10 | 40 | 1.00 | 0.444 | 0.200 |
| 0.0 (flag everything) | 10 | 90 | 1.00 | 1.000 | 0.100 |

Plotting TPR (up) against FPR (across) for each row gives the ROC curve. It always runs from (0, 0), flag nothing, to (1, 1), flag everything. A good model climbs steeply at first: it catches most frauds while still flagging few genuine customers.

The **AUC** (area under the ROC curve) summarises the whole curve in one number between 0 and 1:

- **AUC = 1.0:** a perfect model. Some threshold catches every fraud with no false alarms.
- **AUC = 0.5:** no better than flipping a coin. The curve is the straight diagonal line from (0, 0) to (1, 1).
- **A useful reading:** the AUC is the chance that a randomly chosen fraud gets a higher fraud score than a randomly chosen genuine transaction.

Estimating the area from the seven points above, as a series of trapezoids, gives **AUC ≈ 0.94**: the model separates fraud from genuine transactions well. The AUC does not depend on any threshold, which makes it good for comparing models; but the business still has to choose one threshold to run the model, and that choice is about costs, as the business section below shows.

## Evaluating generative LLMs

A fraud label is either right or wrong, so it can be counted. Most LLM output is free text: an explanation, a summary, an answer to a question. There are many good answers and no single correct string to compare against, so other methods are needed. This section covers the three main ones and what each can and cannot tell you.

In the fraud example, the model also writes a short **explanation** for each flagged transaction, for example "Amount is 12 times this customer's average and the merchant is in a country the card has never been used in." Analysts read it before deciding. Its quality cannot be measured with a confusion matrix.

### Benchmarks

A **benchmark** is a fixed, public set of test questions with a scoring rule, used to compare models on the same tasks. Two well-known examples:

- **MMLU** (Massive Multitask Language Understanding): about 16,000 multiple-choice questions across 57 subjects, from law to medicine. The score is plain accuracy.
- **HumanEval:** 164 programming tasks. The model's code is run against unit tests; the score is the share of tasks it solves.

Benchmarks make comparison cheap and repeatable, but have real limits:

- **Contamination.** Benchmark questions are published on the internet, and models are trained on huge amounts of internet text ([Chapter 2](02-training.md#stage-1-pre-training)). If the test questions and answers were in the training data, the model may simply remember them. A high score then measures memory, not ability. This is called **data contamination**, and it is hard to rule out for any public benchmark.
- **Saturation.** When the best models all score close to 100%, the benchmark can no longer tell them apart, and a newer, harder one is needed.
- **Distance from the real task.** A model that is excellent at multiple-choice exam questions may still write poor fraud explanations. A benchmark only measures what is in it.

The practical lesson: public benchmarks help to shortlist models, but the decision should rest on an evaluation set built from the real task, kept private so it cannot leak into training data.

### Human evaluation

People read the outputs and judge them. Two common forms:

- **Rating:** a reviewer scores each answer against a **rubric** (a written list of criteria), for example 1 to 5 for correctness, clarity and usefulness.
- **Pairwise comparison:** a reviewer sees two answers to the same prompt and picks the better one. People agree with each other more easily on "which is better" than on an absolute score. Public leaderboards such as Chatbot Arena rank models from many such votes.

Human evaluation is the closest measure of real usefulness, but it is slow and expensive, and reviewers disagree. A common check is to have two reviewers rate the same sample and measure how often they agree.

### LLM-as-judge

An **LLM-as-judge** is a second model, given a rubric, that rates or compares outputs in place of a human. It is fast and cheap enough to run on every output.

**Worked example.** The bank uses a judge model to score fraud explanations as acceptable or not, with the rubric "states at least one concrete, checkable reason; contains no claim that contradicts the transaction data". To check the judge before trusting it, an analyst rates 50 explanations by hand (made-up numbers):

| | Analyst: acceptable | Analyst: not acceptable | Total |
| --- | --- | --- | --- |
| **Judge: acceptable** | 38 | 4 | 42 |
| **Judge: not acceptable** | 2 | 6 | 8 |
| **Total** | 40 | 10 | 50 |

The judge agrees with the analyst on 38 + 6 = 44 of 50 explanations, 88%. That is the same confusion matrix as before, used to evaluate the evaluator. Most disagreements are the 4 explanations the judge accepted but the analyst rejected, so the judge is a little too lenient, which is worth knowing before relying on its scores.

Known weaknesses of LLM judges, and the usual fixes:

| Weakness | What happens | Usual fix |
| --- | --- | --- |
| Position bias | In pairwise comparisons, the judge prefers the answer shown first (or second) | Judge each pair twice with the order swapped; keep only consistent verdicts |
| Length bias | Longer answers get higher scores even when they are not better | Rubric that rewards concision; compare answers of similar length |
| Self-preference | A judge rates answers from its own model family higher | Use a judge from a different model family |
| Drift from human judgement | The judge's idea of "good" differs from the users' | Regularly check the judge against a human-rated sample, as above |

### Choosing a method

| Method | Measures | Cost | Main risk |
| --- | --- | --- | --- |
| Public benchmark | General ability on standard tasks | Low | Contamination; far from the real task |
| Private task set with exact answers | Correctness on the real task (e.g. fraud labels) | Low once built | Must be kept up to date |
| Human evaluation | Real usefulness of free text | High | Slow; reviewers disagree |
| LLM-as-judge | Free-text quality at scale | Low | Biases; must be checked against humans |

In practice they are combined: benchmarks to shortlist models, a private labelled set for the measurable part of the task, an LLM judge for free text at scale, and a regular human sample to keep the judge honest.
