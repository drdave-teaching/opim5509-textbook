# Chapter 2 — Dense Neural Networks

:::{admonition} 🔗 Notebooks for this chapter
:class: seealso dropdown
Open in Colab and **Runtime → Run all** — data loads from a stable link, nothing to upload.

- **Binary Classification Titanic structured data** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/0_BinaryClassification_Titanic_structured_data.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/0_BinaryClassification_Titanic_structured_data.ipynb)
- **Forward Propagation** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/1_ForwardPropagation.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/1_ForwardPropagation.ipynb)
- **Multiclass Classification Iris structured data** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/1_MulticlassClassification_Iris_structured_data.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/1_MulticlassClassification_Iris_structured_data.ipynb)
- **A First Look at NN with MNIST images** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/2_A_FirstLook_atNN_with_MNIST_images.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/2_A_FirstLook_atNN_with_MNIST_images.ipynb)
- **Hot And Cold** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/2_HotAndCold.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/2_HotAndCold.ipynb)
- **Back Prop and Re LU** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/3_BackProp_and_ReLU.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/3_BackProp_and_ReLU.ipynb)
- **Back Prop and Re LU new Data** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/3_BackProp_and_ReLU_newData.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/3_BackProp_and_ReLU_newData.ipynb)
- **Fashion MNIST with fashion images** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/3_Fashion_MNIST_with_fashion_images.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/3_Fashion_MNIST_with_fashion_images.ipynb)
- **IMDB Movie Reviews text data** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/4_IMDB_Movie_Reviews_text_data.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/4_IMDB_Movie_Reviews_text_data.ipynb)
- **Assignment2** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/Assignment2.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/Assignment2.ipynb)
- **Assignment3** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/Assignment3.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/Assignment3.ipynb)
- **CA Housing Regression** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/CA_Housing_Regression.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/CA_Housing_Regression.ipynb)
- **Cheat Sheet Building FFNNs** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/CheatSheet_BuildingFFNNs.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/CheatSheet_BuildingFFNNs.ipynb)
- **BatchNorm FromScratch** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/BatchNorm_FromScratch.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/BatchNorm_FromScratch.ipynb)
:::


You've probably met neural networks before, usually drawn like this: some input data on the left (say Boston Housing), a **crystal ball** in the middle, and the target on the right. The crystal ball is "really good at nonlinear relationships," there are "hidden layers" and "activation functions," and maybe you've dragged-and-dropped one in JMP. That's a fine cartoon — but in this chapter we tear the crystal ball open. By the end you'll know exactly how **one row of a spreadsheet becomes a prediction** (forward propagation), and how the network **updates its weights** from the error so the next prediction is better (backpropagation). Then we'll build the whole thing in three lines of Keras.

We'll use a small, friendly dataset throughout the theory: weather variables (temperature, humidity, wind) predicting how many people are **playing golf** on a given day — 14 rows, easy to hold in your head.

## 2.1 Forward propagation: a network is just weighted sums

Start as simple as it gets: **one input, one weight.** A prediction is the input scaled by the weight. Think of a weight as a **volume knob** — turn it up, the input counts for more; turn it down, less. Now give the row three features and three weights. The prediction is the **dot product**:

$$
\hat{y} = x_1 w_1 + x_2 w_2 + x_3 w_3 = \mathbf{x}\cdot\mathbf{w}
$$

A dot product measures a kind of **similarity** between two vectors — but the network doesn't care about any qualitative meaning; it just multiplies element-wise and sums. Stack a *matrix* of weights instead of a vector and you transform the input into an intermediate **hidden layer**; dot *that* with another weight matrix and you get the output. That stack — input → hidden → output — *is* a neural network.

The single most useful thing to internalize is the **shape bookkeeping**, because Keras will demand it of you:

```{admonition} The golden rule of dot products
:class: important
A $(1\times m)$ vector dotted with an $(m\times n)$ matrix yields a $(1\times n)$ result — **the inner dimensions must match and they cancel**, leaving the outer dimensions:

$$
(1\times 4)\cdot(4\times 5)=(1\times 5),\qquad
(1\times 113)\cdot(113\times 50)=(1\times 50)
$$

So a network layer with $m$ inputs and $n$ units carries an $(m\times n)$ weight matrix. If the inner numbers don't line up, the dot product *doesn't compute* — and that's the source of half the shape errors you'll ever hit. Read your model summary with this rule in hand and the shapes stop being mysterious.
```

So forward propagation is nothing more than: take a row, dot it with $W^{(1)}$ to make the hidden layer, (apply an activation — we'll get there), dot with $W^{(2)}$ to make the output. We "make an answer across the network."

## 2.2 How a network learns

The weights start **randomly initialized**, so the first prediction is basically a guess. Learning is the process of nudging those weights to shrink the error.

**Hot-and-cold learning** is the toy version: try the weight a little bigger — did the error go down? Try it a little smaller — better or worse? Move it in the direction that helps, and repeat. It works for one row, but a network that memorizes one row is useless; we want one whose weights have been updated **iteratively across all the rows** so it *generalizes*.

**Backpropagation** is the grown-up version of the same idea, and it's just calculus doing the bookkeeping. For a squared-error loss on one example, the steps are **P, E, and the weight delta**:

```{admonition} The gradient-descent update (one weight, by hand)
:class: tip
With prediction $\hat{y}=\mathbf{x}\cdot\mathbf{w}$ and squared error $E=(\hat{y}-y)^2$:

$$
\text{weight delta}_i = (\hat{y}-y)\,x_i,
\qquad
w_i \leftarrow w_i - \alpha\,(\hat{y}-y)\,x_i
$$

In words: the update to each weight is **(predicted − actual) × the input that flowed through that weight**, scaled by the **learning rate** $\alpha$. Inputs that didn't contribute (an $x_i$ of 0) get no update — the network only adjusts the knobs that mattered for this example. The learning rate controls step size: too big and you overshoot and *diverge*; too small and you crawl.
```

**When do we update?** Three flavors:

- **Stochastic gradient descent (SGD)** — update after *every* row. Lots of randomness ("stochastic" = random).
- **Full (batch) gradient descent** — track the error across the *entire* dataset, update once at the end of the epoch. Smooth, but the step can be clunky and slow to converge.
- **Mini-batch gradient descent** — the sweet spot: accumulate the error over *B* rows, then update. **Batch size $B$ is a hyperparameter** (5, 10, 200, 500 — depends on your data; there's no universal best value).

## 2.3 Why we need nonlinearity (ReLU)

So far the network is just stacked linear maps — and a stack of linear maps is still… linear. That can only learn straight-line correlation between inputs and output. The real world isn't linear. The fix is to insert a **nonlinear activation** between layers. The workhorse is **ReLU**:

$$
\text{ReLU}(z)=\max(0,z)
$$

It's almost embarrassingly simple — pass positives through, clamp negatives to zero — but inserting it between dense layers is what lets the network bend, fold, and approximate genuinely nonlinear relationships. (For the *output* layer we choose the activation to match the task — more on that in a moment.)

## 2.4 Your first neural network — regression in Keras

Now the payoff. Everything above becomes a handful of lines with the **Keras Sequential API**. We predict `medv` on Boston Housing — same five-step methodology from Chapter 1 (read → split → scale → fit → evaluate), but the model is now a network.

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  # hidden layer 1
model.add(Dense(64, activation='relu'))                                   # hidden layer 2
model.add(Dense(1,  activation='linear'))                                 # output (regression → linear)
model.summary()
```

Read that `input_shape=(X_train.shape[1],)` as "the number of feature columns" — it's the golden rule again, setting the inner dimension of the first weight matrix. The output layer has **one** node with a **linear** activation because we're predicting a single continuous number.

```python
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=300, batch_size=10, verbose=1)
```

Three choices worth naming: the **loss** is mean-squared error (regression), the **metric** we watch is MAE (interpretable, in the target's units), and `batch_size=10` is the mini-batch from §2.2. The call returns a **`History`** object — a dictionary of what happened every epoch — which we use to draw **learning curves**:

```python
hist = history.history
epochs = range(1, len(hist['loss']) + 1)
plt.plot(epochs, hist['loss'], 'bo', label='Training loss')
plt.plot(epochs, hist['val_loss'], 'orange', label='Validation loss')
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
plt.title('Training vs. validation loss'); plt.show()
```

```{admonition} Read the learning curves like a doctor reads an X-ray
:class: important
- Both curves falling and hugging each other → healthy learning.
- Training loss keeps dropping while **validation loss turns back up** → **overfitting**; the model is memorizing.
The two standard remedies are **early stopping** (halt when validation stops improving) and **dropout** (randomly zero a fraction of units during training so the network can't lean on any one path). Both, plus the number of layers and hidden units, are **hyperparameters** you tune.
```

**Best practices — how many layers, how many units?** The honest answer is *it's a grid search* — there's no formula that's optimal for every dataset. Reasonable starting points: one hidden layer with as many units as input features, or twice that; then add a second layer; then tune with early stopping and dropout. You might get an equally good model from one wide layer or two narrow ones — different architectures can reach similar fits because the weights are randomly initialized and there are many good solutions.

## 2.5 Classification — binary, multiclass, and beyond

The beautiful secret from Chapter 1 holds for networks too: **classification is the same workflow, the output layer and loss change.**

**Binary** (e.g., Titanic survived/not): one output node with a **sigmoid** activation, `loss='binary_crossentropy'`, and you read results off a confusion matrix (accuracy, precision, recall, F1).

**Multiclass** (e.g., Iris — three species): **one output node per class** with a **softmax** activation, and `loss='categorical_crossentropy'`. Softmax forces the outputs to be positive and **sum to 1** — a probability distribution over the classes:

$$
\text{softmax}(z)_k=\frac{e^{z_k}}{\sum_j e^{z_j}}
$$

```python
from keras.utils import to_categorical

dummy_y = to_categorical(encoded_Y)        # 0,1,2 → 3 one-hot columns

model = Sequential()
model.add(Dense(16, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(3,  activation='softmax'))  # ONE output node per class
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X, dummy_y, epochs=40, validation_split=0.2)
```

Two gotchas worth burning in: encode the target with `to_categorical` so it has **one column per class**, and at prediction time take `argmax` across those columns to recover the predicted class before you build the confusion matrix:

```python
import numpy as np
y_pred = np.argmax(model.predict(X_test), axis=1)
```

The same dense networks even work on **images** (flatten MNIST digits or Fashion-MNIST into a long vector) and **text** (IMDB movie-review sentiment) — they'll *run*, and it's a great way to see the limits of a plain dense net. But for images a network that respects spatial structure does far better, and for sequences a network with memory wins. Those are exactly the next two chapters.

## Wrap-up

```{admonition} Key takeaways
:class: tip
- **Forward propagation** is stacked dot products; the **golden rule** $(1\times m)\cdot(m\times n)=(1\times n)$ governs every layer shape.
- **Learning** = nudging randomly-initialized weights down the error gradient: $w_i \leftarrow w_i - \alpha(\hat y - y)x_i$.
- **SGD / full / mini-batch** trade randomness for stability; **batch size** and **learning rate** are hyperparameters.
- **ReLU** between layers buys you nonlinearity; without it a deep net is just a linear model.
- **Output layer + loss define the task:** linear+MSE (regression), sigmoid+binary-crossentropy (binary), softmax+categorical-crossentropy (multiclass).
- Diagnose with **learning curves**; fight overfitting with **early stopping** and **dropout**.
```


---

## 📌 Lecture key points

*Distilled takeaways from the video lectures behind this chapter — click each to expand.*


:::{admonition} ForwardProp — Part 1
:class: note dropdown
- A neural net is just **weighted sums**; a weight is a "volume knob" scaling its input.
- Build up: one input×one weight → 3 inputs·3 weights (a **dot product**) → input·weight-**matrix** = a hidden layer.
- The "crystal ball" finds **nonlinear** relationships between inputs and target.
- Goal of the series: see how **one row of data becomes a prediction**, then how weights update.
- Running example: weather features (temp/humidity/wind) → number of golfers.
:::

:::{admonition} ForwardProp — Part 2
:class: note dropdown
- Stacking dot products: `(1×3)·(3×5)=(1×5)` hidden, then `(1×5)·(5×1)=(1×1)` output.
- **Golden rule of dot products:** inner dimensions must match and cancel, leaving the outer dims.
- This rule lets you **design weight-matrix shapes** and predict every layer's output shape.
- A dot product encodes a kind of **similarity**; the network doesn't need human meaning for it.
- Mismatched shapes simply **don't compute** — the source of most shape errors.
:::

:::{admonition} HotCold — Part 1
:class: note dropdown
- **Hot-and-cold learning:** nudge a weight up/down, keep the direction that lowers error.
- Weights start **randomly initialized**; learning = iteratively improving them.
- A net that memorizes one row is useless; we want weights updated **across all rows** to generalize.
- Demonstrated on the golf data with the target recoded to 0/1 (majority played or not).
- This is the intuition that backprop will formalize.
:::

:::{admonition} HotCold — Part 2
:class: note dropdown
- Pro: hot-cold finds the optimum; **con: slow**, and a fixed step size can **skip over** the true value.
- Better than subtracting a constant: scale the update by **direction and amount**.
- "Direction and amount" = a function of the **error** and the **size of the input** into that weight.
- Step-size (learning-rate) choice is critical — too big overshoots, too small crawls.
- Motivates gradient descent as the principled version of hot-cold.
:::

:::{admonition} BackProp — Part 1
:class: note dropdown
- Gradient-descent steps: **P** (predict) → **E** (squared error) → **weight delta** → update.
- **Weight delta = (predicted − actual) × the input** that flowed through that weight.
- Start with no hidden layer, learn one row, then generalize.
- Inputs of 0 produce no update — the net only adjusts knobs that mattered.
- Weights randomly initialized; our job is to update them toward accuracy.
:::

:::{admonition} BackProp — Part 2
:class: note dropdown
- **SGD** (update every row) vs **full** (update after all rows) vs **mini-batch** (the sweet spot).
- **Batch size is a hyperparameter** (5/10/200/500…), problem-dependent, no universal best.
- Nets learn **correlation** between inputs and output; uncorrelated inputs aren't adjusted.
- Pure linear stacks can't capture **nonlinear** patterns → need activations.
- Sets up ReLU and nonlinearity as the next idea.
:::

:::{admonition} PuttingItAllTogether — NN Learning
:class: note dropdown
- Full loop: **forward pass** (dot products → 1×1 prediction) → error → **weight update**.
- **Big miss → big update; small miss → small update.**
- Overestimate → shrink positive weights / make negatives more negative; underestimate → the reverse.
- Ties the by-hand mechanics to the code and matching PPT graphics.
- This *is* training — repeated across rows and epochs.
:::

:::{admonition} Best practices for NN regression
:class: note dropdown
- "How many layers / hidden units?" → honestly, it's a **grid search** (hyperparameter tuning).
- You don't know the perfect architecture up front; random init means many good solutions exist.
- Five strategies: 1 layer = #features; 1 layer = 2×features; deeper/wider variants, etc.
- Same-quality fit can come from different architectures (1 wide vs 2 narrow layers).
- Start simple, then tune — don't over-engineer the first model.
:::

:::{admonition} Our first NN for Regression — Part 1
:class: note dropdown
- Import sklearn + keras; read data; handle **dummy variables and missing values**; train/test split.
- Same 5-step methodology as Chapter-1 ML, now with a network.
- Mind shapes throughout (rows/features).
- Scale features (fit on train).
- Set up X/y for Keras.
:::

:::{admonition} Our first NN for Regression — Part 2
:class: note dropdown
- Build with the **Keras Sequential API**: `Dense(64, relu)` layers.
- **`input_shape=(X_train.shape[1],)`** = number of feature columns (the golden rule again).
- Output layer = **1 node, linear** activation for regression.
- `model.summary()` to read parameter counts/shapes.
- Architecture is a stack: hidden → hidden → output.
:::

:::{admonition} Our first NN for Regression — Part 3
:class: note dropdown
- **Compile**: optimizer (rmsprop/Adam), `loss='mse'`, `metrics=['mae']`.
- **Fit** with `validation_data`, `epochs`, `batch_size` (mini-batch GD).
- Use an **early-stopping callback** to halt when validation stops improving.
- The choices (loss/metric/batch) map directly to the theory.
- `fit` returns a **History** object for diagnostics.
:::

:::{admonition} Our first NN for Regression — Part 4
:class: note dropdown
- Evaluate via **learning curves** (training vs validation loss by epoch) **and** traditional metrics.
- Both curves falling together = healthy; validation turning up = **overfitting**.
- Plot from `history.history` dict (`loss`, `val_loss`).
- Complement curves with R²/MAE on test.
- Diagnose before trusting a model.
:::

:::{admonition} Our first NN for Regression — Part 5
:class: note dropdown
- **Dropout** randomly zeros units during training to fight overfitting.
- Tune hyperparameters: dropout rate, early-stopping **patience**, # layers, hidden units.
- These are exactly the knobs you grid-search.
- Regularization buys generalization at a small training-fit cost.
- Iterate: architecture + regularization + early stopping.
:::

:::{admonition} Binary classification with NNs: Titanic — Part 1 & 2
:class: note dropdown
- Output layer = **1 sigmoid node**, `loss='binary_crossentropy'`.
- Prep structured data: numeric-only, **LabelEncoder**/string replacement, handle missing.
- Evaluate with **confusion matrix + classification report** (with/without early stopping).
- **Recall reads along the row** of the true class (handy mnemonic).
- Same Sequential workflow as regression, output+loss changed.
:::

:::{admonition} Multiclass with NNs: Iris — Part 1 & 2
:class: note dropdown
- Output = **one softmax node per class**; `loss='categorical_crossentropy'`.
- Encode target with **`to_categorical`** (one-hot, e.g. 3 columns for Iris).
- **Softmax** outputs a probability distribution summing to 1.
- Take **`argmax`** of predictions to recover the class for the confusion matrix.
- Add proper train/val/test (`validation_split`, holdout test).
:::

:::{admonition} DNNs for MNIST images
:class: note dropdown
- MNIST = first big DL win (**LeNet**, Bell Labs/USPS, reading handwritten ZIP codes).
- Flatten each 28×28 image to a 784-vector and **divide by 255** to scale.
- A dense net *works* on images but ignores spatial structure → **ConvNets are better** (Module 3).
- Multiclass output: 10 softmax nodes.
- Great bridge from structured to **unstructured** data.
:::

:::{admonition} Fashion MNIST and IMDB Movie Reviews
:class: note dropdown
- **Fashion-MNIST**: 10 clothing categories, same pipeline as MNIST (centered images, ÷255).
- **IMDB**: text sentiment — a first taste of unstructured **text** as a dense-net input.
- Both show the limits of plain dense nets → motivates CNNs (images) and RNNs (text).
- Same architecture pattern, different data prep.
- Closes out binary + multiclass classification.
:::
