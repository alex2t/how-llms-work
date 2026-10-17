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
