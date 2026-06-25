# Chapter 4 — Recurrent Networks for Numeric Sequences

So far our data has had no order — shuffle the rows of Boston Housing and nothing changes. **Sequence data is different**: it's a time series — hourly temperature, daily stock closes, warehouse inventory — where *yesterday matters for today*. This chapter is about models that respect that order. We start with numbers (not text) on purpose: text forces you to learn embeddings *and* sequences at once, which is confusing. Master recurrent networks on numeric series first, then Chapter 5 generalizes to words.

There are two ways to feed a time series to a model, and we'll do both.

## 4.1 The window method — turn a sequence into a table

The **window method** (a.k.a. lags) is feature engineering: to predict the next value, use the previous *k* values as features. You slide a window of length `n_steps` (the **look-back**) across the series, and each window becomes a row of X with the next value as y. The beauty is that this **destroys the temporal structure on purpose** — once it's a flat table, you can throw *any* model at it: linear regression, random forest, or a dense neural net from Chapter 2.

Here's the canonical helper (straight from the course notebook):

```python
from numpy import array

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:     # stop at the end of the series
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]  # window → next value
        X.append(seq_x); y.append(seq_y)
    return array(X), array(y)

n_steps = 10                                # look-back: try 10, then 30
X, y = split_sequence(raw_seq, n_steps)
```

This is a great baseline — a window-method dense net is often shockingly competitive. But it *throws away* the sequence. The recurrent network keeps it.

## 4.2 Recurrent networks — let the model engineer the features

A **recurrent neural network (RNN)** preserves the sequence: it reads the time steps **one at a time**, carrying a **hidden state** that summarizes everything it has seen so far. At each step it combines the current input with the hidden state from the previous step — that recurrence is the memory. Just like a ConvNet learned image features for you, an RNN **learns the temporal features for you**; your job shifts from hand-crafting lags to simply getting the data into the right shape.

```{admonition} The 3-D tensor — think "deck of cards"
:class: important
Recurrent layers expect a **3-D tensor**: `(samples, look-back, features)`. If you have 25 windows, a look-back of 5, and 4 features, that's a $25\times5\times4$ tensor — a deck of 25 cards, each card a $5\times4$ slice of history. For a **univariate** series you must `reshape` to add the trailing features dimension of 1:

```python
X = X.reshape((X.shape[0], X.shape[1], 1))   # (samples, n_steps, n_features=1)
```
This shape is where 90% of RNN errors live — get it right and the rest is easy.
```

Building the model is Chapter-2 muscle memory, with a recurrent layer at the front:

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

model = Sequential()
model.add(SimpleRNN(2, input_shape=(n_steps, n_features)))  # 2 hidden units
model.add(Dense(1))                                         # regression output
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
```

```{admonition} Trainable parameters in a recurrent cell
:class: tip
Let $H$ = hidden-state size, $I$ = number of input features, and $G$ = number of little neural nets inside the cell. The cell's parameters are:

$$
\#\text{params} = G\,\big[\,H(H+I) + H\,\big],\qquad
G=\begin{cases}1 & \text{SimpleRNN}\\ 3 & \text{GRU}\\ 4 & \text{LSTM}\end{cases}
$$

The **output shape** of the recurrent layer is just $H$ — it doesn't depend on the look-back. (Tiny worked example: $H=2$, $I=3$, SimpleRNN → the cell learns $2(2+3)+2 = 12$ params; attach a 1-unit dense head and you add $2\cdot1+1=3$, for **15** total.) The same $G/H/I$ recipe covers every recurrent flavor — that's the whole point of writing it as one formula.
```

**SimpleRNN, LSTM, GRU.** A `SimpleRNN` often performs about the same as the window-method dense net — useful, but it struggles to remember long-range dependencies (the vanishing-gradient problem). The **LSTM** ($G=4$) adds gates — input, forget, output — that let it learn what to keep and what to discard over long sequences; the **GRU** ($G=3$) is a lighter cousin that's often just as good. In Keras you swap one word — `SimpleRNN` → `LSTM` → `GRU` — and everything else stays the same.

```{admonition} Always have baselines for time series
:class: warning
Before celebrating an LSTM, beat the dumb models: **mean-only** (predict the average), **persistence** (predict today = yesterday), and **linear regression on the lags**. Persistence in particular is brutally hard to beat on many series. If your fancy recurrent model can't clear these bars, it isn't learning anything useful.
```

## 4.3 Advanced recurrent architectures

Once the basic RNN works, four upgrades squeeze out more signal — and they stack:

- **1-D convolution + pooling.** A `Conv1D` slides a learned filter along the time axis to produce a richer, transformed sequence (the same convolution idea from Chapter 3, in one dimension), and `MaxPooling1D` downsamples it before the recurrent layer. Note: it's `Conv1D`, *not* `Conv2D`, and pull `input_shape` from `X_train` rather than hard-coding it.
- **Recurrent dropout.** Dropout designed for the recurrent connection, so the cell can't over-rely on any single path through time.
- **Stacking recurrent layers.** Feed one recurrent layer into another with `return_sequences=True` so the first layer passes its full sequence forward — but more layers isn't always better; it's a hyperparameter, not a virtue.
- **Bidirectional layers.** Read the sequence **forwards and backwards in parallel** with two independent cells, then **concatenate** their hidden states:

```python
from tensorflow.keras.layers import Bidirectional, LSTM
model.add(Bidirectional(LSTM(30, return_sequences=True)))
```

A `Bidirectional(LSTM(3))` produces 3 forward units and 3 backward units, concatenated to a width-6 output. Reading a series both directions can add surprising predictive power, because patterns that are hard to see going forward sometimes pop out going backward.

**Many-to-many** rounds it out: instead of predicting one value, the network predicts **multiple variables at once** (one model for several product lines) and/or **multiple steps into the future** — a single recurrent model doing a whole forecasting job.

## Wrap-up

```{admonition} Key takeaways
:class: tip
- **Sequence data has order**; the **window method** flattens it into a table (any model works), while **RNNs preserve it** and learn temporal features for you.
- Recurrent layers want a **3-D tensor** `(samples, look-back, features)` — `reshape` univariate series to a trailing 1.
- **Parameters:** $G[H(H+I)+H]$ with $G=1/3/4$ for SimpleRNN/GRU/LSTM; the recurrent layer's **output shape is $H$**, independent of look-back.
- **SimpleRNN → LSTM → GRU** is a one-word swap; LSTM/GRU gates handle long-range memory the SimpleRNN can't.
- Always beat **mean / persistence / linear** baselines.
- Advanced add-ons stack: **Conv1D+pooling, recurrent dropout, stacking (`return_sequences`), and Bidirectional**; many-to-many predicts multiple targets/horizons.
```
