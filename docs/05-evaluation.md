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
