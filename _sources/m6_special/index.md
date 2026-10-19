# Chapter 6 — Special Topics

:::{admonition} 🔗 Notebooks for this chapter
:class: seealso dropdown
Open in Colab and **Runtime → Run all** — data loads from a stable link, nothing to upload.

- **Deep Recommendations** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module6/Deep_Recommendations.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module6/Deep_Recommendations.ipynb)
- **oxford pets image segmentation** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module6/oxford_pets_image_segmentation.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module6/oxford_pets_image_segmentation.ipynb)
:::


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


---

## 📌 Lecture key points

*Distilled takeaways from the video lectures behind this chapter — click each to expand.*


:::{admonition} What is image segmentation
:class: note dropdown
- ConvNets used the **Sequential API** to distill an image down to one label.
- **Segmentation = label every pixel** ("which pixels are the cat?").
- Reverse the funnel: at the bottom, **upsample** to recreate an image (like an autoencoder).
- Goes from **picture to picture** (image → mask).
- The seed idea is the autoencoder, scaled up.
:::

:::{admonition} Introduction to U-Nets and the Functional API
:class: note dropdown
- A **U-Net** = contracting encoder + expanding decoder + **skip connections**.
- Skip connections hand high-res detail across so the mask isn't blurry — needs the **Functional API**.
- Example: **Oxford Pets** images + **trimap** masks (background/border/pet).
- Run on a **GPU** runtime; download/unpack with curl/tar.
- Segmentation = **pixel-wise classification** (softmax per pixel).
:::

:::{admonition} How did our U-Net do
:class: note dropdown
- Evaluate predicted masks against ground-truth trimaps.
- Visualize input image, true mask, predicted mask side by side.
- Qualitative + quantitative assessment.
- Foundation for medical imaging, self-driving, background removal.
- Wraps the segmentation thread.
:::

:::{admonition} Introduction to Deep Recommender Systems
:class: note dropdown
- Reuse **embeddings**: embed **users** and **items** into a shared latent space.
- A learned vector encodes attributes (how much action/comedy; old/young).
- Aligned user/item vectors ⇒ high predicted rating (a **dot product** = matrix factorization).
- How Netflix predicts what you'll like.
- Embed *anything*, not just words.
:::

:::{admonition} Beginner and Advanced Recommender Systems
:class: note dropdown
- **Basic:** `Embedding(user)·Embedding(movie)` via `Dot` (Functional API, two inputs).
- **Advanced:** **concatenate** embeddings → **dense layers** for nonlinear interactions.
- Regularize embeddings (`l2`); fit with early stopping + MSE.
- The Functional API enables multi-input graphs.
- Same evaluate-with-curves-and-metrics discipline.
:::

:::{admonition} Using a deep approach to embeddings (dense layers)
:class: note dropdown
- Replace the plain dot product with a learned **dense** head over concatenated embeddings.
- More expressive: captures nonlinear user–item interactions.
- Demonstrates the latent-vector idea generalizing.
- Closes the loop: embeddings power text *and* recommenders.
- Frontier models reuse these primitives (embeddings, dot-product similarity, skip connections).
:::
