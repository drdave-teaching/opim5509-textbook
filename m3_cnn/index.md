# Chapter 3 — Convolutional Neural Networks

Convolutional neural networks (ConvNets) are one of the coolest places to start in deep learning, and here's the good news up front: once you understand **convolution**, ConvNets are mostly straightforward, and **everything from Chapter 2 — dense layers, ReLU, softmax, backprop — just attaches onto the end.** The new idea is a smarter front end that does **automated feature engineering** on images.

A note on intuition before the math: a fantastic free resource for *seeing* convolution is the image-kernels explainer at [setosa.io/ev/image-kernels](https://setosa.io/ev/image-kernels). Play with it for five minutes; it makes this whole chapter click.

## 3.1 Images are just numbers

Treat an image as a **tensor**. A black-and-white image is a 2-D grid of pixel intensities (0–255). A color image is a 3-D tensor — height × width × **3 channels** (red, green, blue), each 0–255. That's it: pixels are numbers, and a model can do math on numbers.

A **kernel** (or **filter**) is a small matrix — say 3×3 — that we slide across the image. At each position we multiply the kernel by the patch of pixels under it and sum. That single operation, repeated across the whole image, is **convolution**, and it produces a **feature map**. Different kernels detect different things — edges, blurs, corners — and the magic of a ConvNet is that it **learns the kernels** instead of you hand-designing them.

```{admonition} Key idea — convolution learns the features for you
:class: important
In Chapter 1 *you* engineered features. In a ConvNet, the convolution layers **learn** the useful filters during training. Stacked convolutions build a hierarchy: early layers learn edges and textures, later layers learn parts and objects. The feature map a convolution produces is always **lower resolution** than its input, because the kernel can only sit fully inside the image at a limited number of positions.
```

A **pooling** layer (usually **max pooling**, 2×2) then downsamples each feature map by keeping only the strongest activation in each little window. Pooling shrinks the spatial size, keeps the salient signal, and — crucially — buys **spatial invariance**: the network recognizes a cat whether it's centered or off in the corner. That's why ConvNets free you from the perfectly-centered, MNIST-style data and let you bring on messy, real-world images.

## 3.2 Counting parameters and output shapes

Reading a ConvNet's `model.summary()` is a skill, and it comes down to two formulas.

```{admonition} Conv layer arithmetic
:class: tip
For a `Conv2D` layer with $F$ filters of size $M\times M$ acting on an input with $C$ channels:

$$
\#\text{params} = (M \cdot M \cdot C + 1)\cdot F
$$

(the $+1$ is the bias per filter). And a "valid" (no-padding) convolution turns an input of side $L$ into a feature map of side:

$$
L_{\text{out}} = L - (M-1)
$$

— note it's $L-(M-1)$, **not** $L-M-1$, a place students lose a pixel and then lose an hour. Max pooling with a $2\times2$ window halves each spatial dimension and has **zero** trainable parameters.
```

Work one example by hand (Conv2D → maxpool → Conv2D → pool → flatten → dense) and the model summary stops being a wall of numbers and becomes something you can predict. Use a cheat sheet first, then do it by hand until you don't need the sheet.

## 3.3 Building a ConvNet — cats vs. dogs

The "hello world" of computer vision is binary cats-vs-dogs classification. The architecture is a tower of **Conv2D + MaxPooling** pairs (extracting ever-richer features at ever-smaller resolution), then **Flatten** into a vector, then the **dense** classifier from Chapter 2:

```python
from tensorflow.keras import models, layers, optimizers

model = models.Sequential()
model.add(layers.Conv2D(32,  (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64,  (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1,   activation='sigmoid'))   # binary → one sigmoid node

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
```

Notice the output layer is exactly what Chapter 2 prescribed for a binary problem — one node, sigmoid, binary-crossentropy. The ConvNet only changed the *front end*.

Images rarely fit in memory, so we stream them from folders with an **`ImageDataGenerator`**. Two non-negotiables: **rescale** the pixels from 0–255 to 0–1 (`rescale=1./255`), and let the generator resize everything to a fixed shape:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
```

```{admonition} Data augmentation — free training data
:class: tip
With only a few thousand images, a ConvNet overfits fast. **Augmentation** generates endless plausible variants on the fly — random zoom, shift, shear, rotation, flips — by adding arguments to the *training* generator (e.g. `rotation_range=40, zoom_range=0.2, horizontal_flip=True`). The model sees a slightly different cat every epoch and generalizes better. Augment **train only** — never the validation/test stream.
```

You then `fit` against the generators (with `steps_per_epoch`), draw the same learning curves as Chapter 2, and evaluate with a confusion matrix and classification report. Same methodology, image front end.

## 3.4 Transfer learning — standing on giants

Here's the practical reality: serious computer vision wants **a lot** of data, and you usually don't have it. Suppose you work for an insurer classifying school equipment — gas-fired vs. oil-fired furnaces — and you have maybe **300 labeled images per class**. A from-scratch ConvNet might do okay, but you can do far better by **stealing the filters from a network trained on millions of images.**

That's **transfer learning**. A model like **VGG16**, trained on ImageNet's 1,000 categories and millions of images, learned convolutional filters that are *general* — edges, textures, shapes useful for almost any vision task. We keep that pre-trained **convolutional base**, throw away its classifier head, and bolt on our own:

```python
from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights='imagenet',   # the pre-trained filters
                  include_top=False,     # drop ImageNet's 1000-way classifier
                  input_shape=(150, 150, 3))
conv_base.summary()
```

Two ways to use it. **Feature extraction:** freeze `conv_base`, run your images through it once to get rich feature vectors, and train a small dense classifier on those — fast, great when data is tiny. **Fine-tuning:** unfreeze the *top* few convolutional layers and train them at a very low learning rate so they adapt to your domain without destroying what they already know. Either way, you get a strong vision model from a few hundred images.

## 3.5 Autoencoders — learning to compress (and a bridge to segmentation)

An **autoencoder** is a network that takes an input and is trained to **recreate that same input**. Why predict yourself? Because the network is forced to funnel the image through a narrow **latent** layer in the middle and rebuild it from that compact code.

Take a 28×28 MNIST digit (784 numbers), squeeze it through a `Dense(32)` bottleneck, and reconstruct the 784 pixels: you've done ~95% compression and learned a tiny representation that still captures the digit. That **encoder → latent space → decoder** structure generalizes far beyond compression:

- **Denoising** — feed noisy/corrupted images in, clean images out.
- **Colorization** — feed black-and-white in, color out (train on B&W/color pairs, then colorize century-old photos).
- **Image segmentation** — the encoder–decoder idea, scaled up with skip connections, becomes the **U-Net** we'll build in Chapter 6.

So the humble "predict yourself" model is the conceptual seed of generative and segmentation models.

## Wrap-up

```{admonition} Key takeaways
:class: tip
- Images are **tensors** (H×W for grayscale, H×W×3 for RGB); **convolution** slides learned **kernels** to build **feature maps**; **pooling** downsamples and grants **spatial invariance**.
- Master the two formulas: conv params $=(M^2C+1)F$ and valid output side $=L-(M-1)$; pooling has no parameters.
- A ConvNet is a **Conv2D+MaxPooling tower → Flatten → dense head**; the head is just Chapter 2 (sigmoid/softmax + matching loss).
- Stream images with **`ImageDataGenerator`** (`rescale=1./255`); **augment train only** to fight overfitting.
- With little data, **transfer learning** (VGG16, `include_top=False`) wins — feature-extract or fine-tune.
- **Autoencoders** compress to a latent space and reconstruct — the seed of denoising, colorization, and U-Net segmentation.
```
