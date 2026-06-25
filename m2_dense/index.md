# Chapter 2 — Dense Neural Networks

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
