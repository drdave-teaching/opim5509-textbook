# Preface

:::{admonition} ⚠️ Work in progress
:class: warning
These materials are a **living draft** — actively being written, revised, and expanded from my lecture transcripts and course notebooks. Expect rough edges, gaps, and changes between visits. This is a teaching companion, **not a final or official reference**. Spot something off? That's expected — it's a work in progress!
:::

Welcome to **Deep Learning — A Graduate Introduction**, the book edition of **OPIM 5509** at the University of Connecticut.

This is not a dry reference manual. It is the course as I actually teach it — casual in voice, serious in content, and **relentlessly code-first**. Every idea in this book is paired with code you can run, because in deep learning the gap between "I understand the concept" and "I can build the model" is exactly where careers are made or lost. We are going to close that gap.

## Who this book is for

You're a graduate student (or a practicing analyst) who already has **solid data-wrangling skills and a working knowledge of machine learning**. You can read a dataset into pandas, split it into train and test partitions, fit a `RandomForestRegressor`, and say something intelligent about the result. If that sentence made you nervous, start with {doc}`Chapter 1 <m1_refresher/index>` — it's a deliberate level-set — and consider spending a few extra weeks on the data-science fundamentals first. Deep learning without the fundamentals is like learning to fly a plane before you can ride a bike.

## How the book is organized

The book follows the arc of the course, from the simplest neural network to the architectures behind modern AI:

```{tableofcontents}
```

- **Chapter 1 — Refresher.** EDA and the machine-learning methodology in scikit-learn. Our shared baseline.
- **Chapter 2 — Dense Neural Networks.** The whole engine, by hand: forward propagation, the dot product, hot-and-cold learning, backpropagation, and gradient descent — *then* the same thing in three lines of Keras.
- **Chapter 3 — Convolutional Neural Networks.** Images are just numbers. Automated feature engineering with convolution and pooling, trainable-parameter arithmetic, transfer learning, and autoencoders.
- **Chapter 4 — Recurrent Networks for Numeric Sequences.** The window method, SimpleRNN → LSTM → GRU, bidirectional layers, 1-D convolutions, and many-to-many forecasting.
- **Chapter 5 — Recurrent Networks for Text.** Classic NLP (bag-of-words, TF-IDF), word embeddings, and RNNs that read.
- **Chapter 6 — Special Topics.** Image segmentation with U-Nets, and deep recommender systems.

## How to read it

Read with a notebook open. When you hit a code block, run it. When you hit a **math callout**, don't skip it — the math in this book is the *minimum* you need to reason about layer shapes, parameter counts, and why a model is or isn't learning. When you hit a {bdg-primary}`Key idea`, slow down.

The code in this book is drawn directly from the course notebooks (all public on the [`drdave-teaching`](https://github.com/drdave-teaching) GitHub account). I've reproduced the parts that matter inline so you can read the explanation and the implementation in one place.

Let's get to work.

— *Dave*
