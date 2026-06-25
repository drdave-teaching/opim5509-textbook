# Chapter 6 — Special Topics

This is the cherry on top — two applications that recombine everything you've built. **Image segmentation** takes the autoencoder idea and turns it into picture-to-picture prediction with **U-Nets**. **Deep recommender systems** take the embedding idea from text and use it to learn who you are. Both also introduce the **Functional API**, the more flexible way to wire up Keras models when a simple stack won't do.

## 6.1 Image segmentation with U-Nets

Every ConvNet so far used the **Sequential API**: read the image, convolve (light downsampling), pool (aggressive downsampling), stack those blocks while *expanding* the number of filters to learn richer textures, then flatten into a couple of dense layers and predict **one label** for the whole image ("cat" or "dog").

**Segmentation asks a harder question:** not *"is there a cat?"* but *"which pixels are the cat?"* — a label for **every pixel**. The trick is to take the ConvNet's funnel and, at the bottom, **reverse it**:

```{admonition} Key idea — go down, then back up
:class: important
A ConvNet distills an image *down, down, down*. An **autoencoder** (Chapter 3) went down to a tiny latent vector and back *up* to reconstruct the **same** image. A **U-Net** does the same down-then-up shape, but the decoder is trained to produce a **different** image — a pixel-wise **mask**. The "U" comes from the architecture's shape: a contracting **encoder** path, an expanding **decoder** path, and **skip connections** that hand high-resolution detail straight across from encoder to decoder so the output isn't blurry.
```

You can't build that with the Sequential API — the skip connections mean a layer needs inputs from *two* places. Enter the **Functional API**, where you call layers like functions and explicitly wire the graph (more on the pattern in §6.2).

The course example is the **Oxford-IIIT Pets** dataset: cat and dog photos (`images/`) paired with **trimap** masks (`annotations/trimaps/`). "Tri" = three labels per pixel — background, border, and "definitely a pet." So segmentation is really **pixel-wise classification**: for each pixel, softmax over the three classes. Run it on a GPU runtime (you'll want it), feed image→mask pairs, and the U-Net learns to paint the pet. The same machinery underlies medical imaging, self-driving perception, and background removal.

## 6.2 Deep recommender systems

Remember **embeddings** from Chapter 5: a learned vector that captures *meaning* — a word placed in n-dimensional space so similar words sit nearby (cat/dog, young/old). The big idea of this section: **embed *anything*, not just words.** Embed **users** and **movies**, and you get a recommender — how Netflix guesses what you'll like.

Picture each user as a learned latent vector (how much do you like action? comedy? old films?) and each movie as a latent vector in the *same* space (how much action? comedy? how old?). If a user's vector and a movie's vector **point the same way**, that's a high predicted rating. "Point the same way" is exactly a **dot product**:

$$
\hat{r}_{u,m} = \mathbf{p}_u \cdot \mathbf{q}_m = \sum_{f=1}^{F} p_{u,f}\,q_{m,f}
$$

where $\mathbf{p}_u$ is the user's $F$-dimensional embedding and $\mathbf{q}_m$ the movie's. (If that looks like **matrix factorization**, it is — expressed as a neural network whose embeddings are learned by gradient descent.)

This model has **two inputs** (a user id and a movie id), so we need the **Functional API**:

```python
from keras.layers import Input, Reshape, Dot, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

def RecommenderV1(n_users, n_movies, n_factors):
    user  = Input(shape=(1,))
    u = Embedding(n_users,  n_factors, embeddings_regularizer=l2(1e-6))(user)   # user → F-vector
    u = Reshape((n_factors,))(u)

    movie = Input(shape=(1,))
    m = Embedding(n_movies, n_factors, embeddings_regularizer=l2(1e-6))(movie)  # movie → F-vector
    m = Reshape((n_factors,))(m)

    x = Dot(axes=1)([u, m])                       # predicted rating = u · m
    model = Model(inputs=[user, movie], outputs=x)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
    return model

model = RecommenderV1(n_users, n_movies, n_factors)
model.summary()
```

```{admonition} The Functional API in one breath
:class: tip
Instead of `model.add(...)` in a line, you **call each layer on a tensor** — `u = Embedding(...)(user)` — and assemble the pieces with `Model(inputs=[...], outputs=...)`. This lets you have **multiple inputs/outputs** and **non-linear graphs** (two embedding towers meeting at a `Dot`; encoder/decoder skip connections in a U-Net). Sequential is the easy 90%; Functional is how you build the other 10% that makes deep learning powerful.
```

The "advanced" version simply enriches the towers — concatenate the user and movie embeddings and feed them through **dense layers** (`Embedding → Concatenate → Dense → Dense → rating`) so the model learns nonlinear interactions instead of a plain dot product. Same idea, more expressive head — and you fit it with early stopping, learning curves, and error metrics, exactly like every other model in this book.

## 6.3 Where to go next

You now have the whole toolkit: dense networks (Chapter 2), convolution for images (Chapter 3), recurrence for sequences (Chapters 4–5), and the encoder–decoder/embedding patterns that power segmentation and recommenders (Chapter 6). The architectures behind today's frontier models — transformers, diffusion models, large language models — are recombinations and scalings of these same primitives: embeddings, attention as a learned dot-product similarity, residual/skip connections, and gradient descent over a lot of data. You've seen every ingredient. Go build something.

## Wrap-up

```{admonition} Key takeaways
:class: tip
- **Segmentation = pixel-wise classification.** A **U-Net** is an encoder (down) + decoder (up) with **skip connections**, generalizing the autoencoder from "reconstruct" to "predict a mask."
- The **Functional API** (call layers like functions, wire with `Model(inputs, outputs)`) is required for skip connections and multi-input models.
- **Recommenders embed users and items** into a shared latent space; a **dot product** of the two vectors predicts the rating — matrix factorization as a neural net.
- The "advanced" recommender swaps the dot product for **concatenate → dense layers** to learn nonlinear interactions.
- Frontier models reuse these exact primitives — **embeddings, dot-product similarity, skip connections, gradient descent at scale.**
```
