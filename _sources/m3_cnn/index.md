# Chapter 3 — Convolutional Neural Networks

:::{admonition} 🔗 Notebooks for this chapter
:class: seealso dropdown
Open in Colab and **Runtime → Run all** — data loads from a stable link, nothing to upload. The cats-vs-dogs, interpretability, and transfer-learning notebooks want a **GPU runtime** (Runtime → Change runtime type → T4); the autoencoder notebooks are fine on CPU.

- **Simple Size and Param** *(M3.1 — the parameter math by hand)* &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module3/Simple_Size_and_Param.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module3/Simple_Size_and_Param.ipynb)
- **ConvNets on Small Datasets — Cats vs. Dogs** *(M3.2)* &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module3/ConvNets_on_Small_Datasets_Cats_vs_Dogs.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module3/ConvNets_on_Small_Datasets_Cats_vs_Dogs.ipynb)
- **Inside the ConvNet — Activations & Grad-CAM** *(M3.2)* &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module3/Inside_the_ConvNet_Activations_and_GradCAM.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module3/Inside_the_ConvNet_Activations_and_GradCAM.ipynb)
- **Transfer Learning with a Pretrained ConvNet** *(M3.3)* &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module3/Transfer_Learning_with_a_Pretrained_ConvNet.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module3/Transfer_Learning_with_a_Pretrained_ConvNet.ipynb)
- **Autoencoders for Images** *(M3.3 — the mechanics, plus Sequential and conv-autoencoder appendices)* &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module3/Autoencoders_for_Images.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module3/Autoencoders_for_Images.ipynb)
- **What Autoencoders Can Actually Do** *(M3.3 — outliers, features, denoising, clustering)* &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module3/What_Autoencoders_Can_Actually_Do.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module3/What_Autoencoders_Can_Actually_Do.ipynb)
:::


Convolutional neural networks (ConvNets) are one of the coolest places to start in deep learning, and here's the good news up front: once you understand **convolution**, ConvNets are mostly straightforward, and **everything from Chapter 2 — dense layers, ReLU, sigmoid/softmax, backprop — just attaches onto the end.** The new idea is a smarter front end that does **automated feature engineering** on images.

A note on intuition before the math: a fantastic free resource for *seeing* convolution is the image-kernels explainer at [setosa.io/ev/image-kernels](https://setosa.io/ev/image-kernels). Play with it for five minutes; it makes this whole chapter click.

## 3.1 Images are just numbers

Treat an image as a **tensor**. A black-and-white image is a 2-D grid of pixel intensities (0 = black, 255 = white). A color image is a 3-D tensor — height × width × **3 channels** (red, green, blue), each 0–255 — a stack of three pictures, like pages in a book. And nothing stops you from stacking ten: satellite bands, census layers. Pixels are numbers, and a model can do math on numbers.

Why not just flatten the pixels into the dense network from Chapter 2? Because that network learned *which pixel positions* correlate with the answer. It works on MNIST only because every digit is centered and curated; slide a digit into the corner and the pixel-to-output correlations shift and the model falls apart. We need a technique that doesn't care *where* the thing is.

A **kernel** (or **filter**) is a small matrix — say 3×3 — that we slide across the image with a **stride** of 1. At each position we multiply the kernel by the patch of pixels under it and sum, recoding nine pixels into one. That single operation, repeated across the whole image, is **convolution**, and it produces a **feature map**. On setosa.io a *Sharpen* kernel makes the eyes pop and a *left Sobel* lights up the vertical edges — different kernels detect different things, and the magic of a ConvNet is that it **learns the kernel values by backpropagation** instead of you hand-designing them.

```{admonition} Key idea — convolution learns the features for you
:class: important
In Chapter 1 *you* engineered features. In a ConvNet the convolution layers **learn** the useful filters during training. Think of a pasta maker: one dog goes in, and a conv layer with 32 feature maps makes 32 differently-recoded copies of it. Stacked convolutions build a hierarchy — early layers learn edges and textures, later layers learn eyes, noses, and fur. A feature map is always **lower resolution** than its input (the kernel can only sit fully inside the image at a limited number of positions) — that shrink is *light downsampling*.
```

A **pooling** layer (usually **max pooling**, 2×2) then downsamples each feature map far more aggressively: picture a domino flipping across the map, keeping only the strongest activation in each window. It moves by its own size, so 24×24 becomes 12×12, and it learns **nothing** — zero trainable parameters. The payoff is **spatial invariance**: kernels light up wherever the feature is, so the network recognizes a cat whether it's centered or off in the corner. That's why ConvNets free you from perfectly-centered MNIST-style data and let you bring on messy, real-world photos — and why a ConvNet can need a *tenth* of the parameters a dense net needed on the same images.

## 3.2 Counting parameters and output shapes

Reading a ConvNet's `model.summary()` is a skill, and it comes down to two formulas. Dave's notation: the kernel is **M × N**, **L** is the number of channels coming *in*, **B** is the bias (one per filter), and **F** is the number of feature maps the layer creates.

```{admonition} Conv layer arithmetic
:class: tip
$$
\#\text{params} = \big((M \times N \times L) + B\big) \times F
$$

and a "valid" (no-padding) convolution turns an input of side $L_{\text{in}}$ into a feature map of side

$$
L_{\text{out}} = L_{\text{in}} - (M-1)
$$

— note it's $L-(M-1)$, **not** $L-M-1$, a place students lose a pixel and then lose an hour. Max pooling with a 2×2 window halves each spatial dimension and has **zero** trainable parameters, and so does `Flatten`.

**The term that bites is L.** At the first layer it's 1 (grayscale) or 3 (RGB). Halfway through the network it's *whatever the previous conv layer produced* — a filter's depth is **inherited** from the maps coming in, while the count F is **chosen** by you.
```

Work the whole chain by hand once, on the class example: a 28×28×1 image → `Conv2D(3 maps, 5×5)` gives 24×24×3 and $((5\times5\times1)+1)\times3 = 78$ parameters → pool to 12×12×3 → `Conv2D(5 maps, 3×3)`, where each filter is now a 3×3×**3** brick, gives 10×10×5 and $((3\times3\times3)+1)\times5 = 140$ → pool to 5×5×5 → flatten to 125 → `Dense(256)` = 32,256 → output 514. Total **32,988** — versus roughly 400,000 for the dense MNIST net. Notice where the parameters live: the conv layers are cheap, the dense head at the end is where the bill comes due. Use the cheat sheet first, then do it unaided until the summary is predictable — and then try it with *ugly* numbers (29 maps, 87 channels), because textbooks that use 3 everywhere hide which 3 is the kernel and which is the channel count.

## 3.3 Building a ConvNet — cats vs. dogs

The "hello world" of computer vision is binary cats-vs-dogs classification: 2,000 training images, 1,000 validation, 1,000 test, color, and — unlike MNIST — the cat can be anywhere in the frame.

**Get the folders right first.** Keras reads your labels from the directory structure: `train/`, `validation/`, `test/`, each holding one lowercase subfolder per class (`cats/`, `dogs/`). The `ImageDataGenerator` takes the class names straight from the folder names, alphabetically — so **cat = 0, dog = 1**. Copy this structure exactly for your own data (mask vs. no-mask, anything), and the same notebook runs.

The architecture is a tower of **Conv2D + MaxPooling** pairs, extracting ever-richer features at ever-smaller resolution (150×150 in, 7×7×128 out), then **Flatten** into one long vector, then the **dense** classifier from Chapter 2:

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
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])
```

Notice the output layer is exactly what Chapter 2 prescribed for a binary problem — one node, sigmoid, binary-crossentropy. The ConvNet only changed the *front end*, and the feature engineering is handled for you: the conv/pool layers train their kernels so the right things light up in the feature maps.

Images rarely fit in memory, so we stream them from the folders with an **`ImageDataGenerator`**. Two non-negotiables: **rescale** the pixels from 0–255 to 0–1 (`rescale=1./255` — min-max scaling, without which the weight updates diverge), and let the generator resize everything to a fixed 150×150. And there are **three** generators, one per partition, because you always want a held-out set the model has never seen:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
eval_datagen  = ImageDataGenerator(rescale=1./255)

train_generator      = train_datagen.flow_from_directory(train_dir,      target_size=(150, 150), batch_size=20, class_mode='binary')
validation_generator = eval_datagen.flow_from_directory(validation_dir,  target_size=(150, 150), batch_size=20, class_mode='binary')
test_generator       = eval_datagen.flow_from_directory(test_dir,        target_size=(150, 150), batch_size=20, class_mode='binary', shuffle=False)
```

Two details in the fit call are new. **`steps_per_epoch`** is how many batches make one pass over the data — images ÷ batch size (100 images at batch 20 = 5 steps) — and you should compute it from the partition rather than hard-code it. And you train with **early stopping on validation loss** (with `restore_best_weights=True`), let the epoch count be a ceiling, and save the model in the modern **`.keras`** format:

```python
from tensorflow.keras.callbacks import EarlyStopping

train_steps = len(train_generator.filepaths) // train_generator.batch_size
val_steps   = len(validation_generator.filepaths) // validation_generator.batch_size

history = model.fit(train_generator, steps_per_epoch=train_steps, epochs=100,
                    validation_data=validation_generator, validation_steps=val_steps,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)])
model.save('/content/cats_and_dogs_small_1.keras')
```

Then draw the same learning curves as Chapter 2. Training accuracy climbs forever — the model is memorizing — and the validation curve tells you where to freeze it. Once your **loss curve plateaus, the model is done**; if it's still heading down the ramp, there's headroom left.

```{admonition} Data augmentation — free training data
:class: tip
With only a couple thousand images a ConvNet overfits fast. **Augmentation** generates endless plausible variants on the fly — random zoom, shift, rotation up to 40°, flips — by adding arguments to the *training* generator (e.g. `rotation_range=40, zoom_range=0.2, horizontal_flip=True`). Now the model has to work to classify ten altered versions of the same cat instead of memorizing one, and the fit comes out smoother and more robust. Augment **train only** — never the validation/test stream (same logic as SMOTE: alter the training data, evaluate on the real thing). Dropout helps too, but it slows learning, so give it a longer epoch runway.
```

**Evaluate like you mean it.** Score *every* partition — they should land within a few points of each other. Then go past "80% accuracy," which is where the textbook stops: predict on the (unshuffled) test generator, line the predictions up against the true labels in a DataFrame, and produce a **classification report and confusion matrix with real class names** (`target_names=list(train_generator.class_indices)`). Now you can talk precision and recall per class — on the class run the model was a little better at cats than dogs, and you can compute the recall by hand off the matrix. That's what you bring to your boss, not a single accuracy number.

```{admonition} Open the black box — activations and Grad-CAM
:class: important
A ConvNet can score well for the wrong reason. The textbook's cautionary tale: a polar-bear-vs-grizzly model that never looks at the bears — it learned *snow* means polar and *trees and mud* mean grizzly. So look at how the model decides. Load the saved `.keras` model and (1) visualize the **feature-map activations** layer by layer — the early layers still look like the dog, the deep layers dissolve into abstract boxes, and some maps never fire — then (2) run **Grad-CAM**: backpropagate the class score to the last conv layer to get a heat map of which pixels drove the decision. On a confident correct hit the dog's face lights up. On a confident *miss* — two cats sitting on a doghouse, called "dog" — the heat sits on the doghouse: the model learned that doghouses correlate with dogs. Pull your misses and study them; it's never been easier to spin up an assistant and do this, so your ConvNet doesn't have to be a black box.
```

## 3.4 Transfer learning — standing on giants

Here's the practical reality: training from scratch is slow (fifteen or twenty minutes for 2,000 images — imagine 200,000), and small data is the norm. Suppose you have a couple hundred images per class. You could squeeze the data with augmentation — or you could **steal the filters from a network trained on millions of images.**

That works because ConvNets are fundamentally **texture and feature detectors** — straight lines, curves, scales, bumpy things, smooth things — and only the dense head at the end says "cat" or "boat" or "furnace." A model like **VGG16**, trained on ImageNet's 1,000 categories, learned convolutional filters that are *general*. We keep its pre-trained **convolutional base**, throw away the classifier head, and bolt on our own:

```python
from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights='imagenet',      # the pre-trained filters
                  include_top=False,        # drop ImageNet's 1000-way classifier
                  input_shape=(150, 150, 3))
conv_base.summary()                         # 13 conv layers; squishes a 150x150 image to 4x4x512
```

Two ways to use it.

**Feature extraction.** Freeze the base and just run your images through it — it's a Plinko board, nothing trains — collecting the 4×4×512 features for every train/validation/test image, flatten them (4·4·512 = 8,192 — soft-code that shape in real life), and train a small dense classifier with dropout and early stopping on top. Validation accuracy in the 90s in about three minutes, then a clear **plateau** after eight or ten epochs: you've extracted everything this architecture has.

**Fine-tuning.** Build the cleaner version — `conv_base` → `Flatten` → dense layers — and note the base has 13 conv layers, so with a kernel and a bias each plus your two new layers that's **30 trainable weight tensors**. Then the single most important line:

```python
conv_base.trainable = False        # THE key line
```

Skip it and `model.fit` treats the pretrained base as random weights — backprop starts destroying the feature detectors you just borrowed. With the base frozen, train the head; then, for the last bit, unfreeze **only the last convolutional block** (`block5_conv1` onward) so the fine-resolution detectors adapt to *your* problem while blocks 1–4 stay general, and retrain the head alongside. That typically lifts the model into the mid-90s. The layer names differ by architecture, so adjust the block name if you swap in another pretrained model.

```{admonition} The story to tell your manager
:class: tip
Train from scratch on your own data and get a baseline (maybe the 60s or 70s). Then feature-extract with a pretrained base and retrain the head. Then fine-tune the last conv block too. Present the whole arc — that's a complete, defensible modeling story, not a single number.
```

## 3.5 Autoencoders — learning to compress, and what that buys you

An **autoencoder** is a network that takes an input and is trained to **recreate that same input**. Why predict yourself? Because the network is forced to funnel the image through a narrow **latent vector** in the middle and rebuild it from that compact code — a bow tie. Take a 28×28 MNIST digit (784 numbers), scale it by 1/255, reshape to a row of 784, squeeze it through a 32-unit encoding, and reconstruct the 784 pixels. This is where we step from the Sequential API to the **Functional API**, because we'll want to snap the encoder out on its own afterward:

```python
from keras.layers import Input, Dense
from keras.models import Model

input_img = Input(shape=(784,))
encoded   = Dense(32,  activation='relu')(input_img)     # the 32-number latent code
decoded   = Dense(784, activation='sigmoid')(encoded)    # rebuild the pixels

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True,   # X and y are the same thing
                validation_data=(x_test, x_test), callbacks=[early_stop])

encoder = Model(input_img, encoded)                      # same weights, stops at the code
```

Why **sigmoid**, not linear, on the output? Because every pixel lives in [0, 1], so a sigmoid output plus binary crossentropy is the activation that makes the math land in the right range — a modeling trick worth remembering. And the crazy part: the reconstructions from a 1×32 code come back strikingly close to the originals (a little fuzzy, but the digit is unmistakably there). Roughly 50,000 parameters, ~95% compression. *(The mechanics notebook also shows the same model the Sequential way, and a **convolutional autoencoder** — strided `Conv2D` down, `Conv2DTranspose` up — that reconstructs sharper with far fewer parameters because the layers respect the 2-D structure.)*

That **encoder → latent space → decoder** structure is the seed of a lot more than compression. Four things it buys you, all off one little autoencoder:

1. **Catch outliers (anomaly detection).** Train on *normal* data only. **Reconstruction error is an anomaly score** — small for things the model has seen, big for things it hasn't. Pick a threshold from the normal data (the 95th percentile of digit error — that 5% is your false-alarm budget, not a property of the anomalies), and anything above it gets flagged. Feed the MNIST-trained model Fashion-MNIST sneakers and purses and most of them clear the bar; a 7 rebuilds as a 7, a shoe comes out garbled. The trap: **don't add the anomalies to training** — the model would learn to rebuild them and they'd stop looking anomalous. Model normal really well and let the weird stuff out itself.
2. **Free features.** The 32-number code is a compressed, nonlinear summary of the whole image. Feed *it* to a small softmax classifier instead of 784 raw pixels — 24× fewer inputs, still around 95% on the digits — and it's a lighter, more private representation to hand a downstream team.
3. **Denoising.** Swap the target: train **noisy in → clean out**. Make the dirt yourself, and be deliberate about **leakage** — noise the train and test sets independently, fit only on the training pairs. The model learns to extract the signal from the noise.
4. **Find structure.** Squash the codes to 2-D with **t-SNE** (a nonlinear cousin of PCA) and color by the true digit, which was never used in training: the digits have sorted themselves into groups. Big separated clumps mean a classifier will do great; overlap is where it'll be confused.

The same encoder–decoder idea, scaled up with skip connections, becomes the **U-Net** we build in Chapter 6 — so the humble "predict yourself" model is the conceptual seed of segmentation and generative models too.

## Wrap-up

```{admonition} Key takeaways
:class: tip
- Images are **tensors** (H×W for grayscale, H×W×3 for RGB); **convolution** slides learned **kernels** to build **feature maps**; **pooling** downsamples and grants **spatial invariance**.
- Master the two formulas: conv params $=((M\times N\times L)+B)\times F$ with **L inherited, F chosen**, and valid output side $=L-(M-1)$; pooling and flatten have no parameters. The class example totals **32,988**.
- A ConvNet is a **Conv2D+MaxPooling tower → Flatten → dense head**; the head is just Chapter 2. **Folder names are your labels** (cat = 0, dog = 1).
- Stream images with **`ImageDataGenerator`** (`rescale=1./255`), **three** generators, soft-coded `steps_per_epoch`, **early stopping**, `.keras` saves; **augment train only**.
- Evaluate **every** partition and finish with a **classification report + confusion matrix with class names**; then **look inside** — activations and **Grad-CAM** — to make sure the model is using the image for the right reasons.
- With little data, **transfer learning** wins: VGG16 with `include_top=False`, **freeze the base** (the key line), feature-extract, then fine-tune the last conv block.
- **Autoencoders** compress to a latent code and reconstruct (Functional API, sigmoid + binary crossentropy) — and that buys **anomaly detection, free features, denoising, and clustering**, plus the seed of U-Net segmentation.
```


---

## 📌 Lecture key points

*Seventeen videos, 1:50:06. Transcripts, polished scripts, per-video skill sheets and per-section skills checklists: [opim5509-transcripts / fall2026_idl / module3](https://github.com/drdave-teaching/opim5509-transcripts/tree/main/fall2026_idl/module3).*

### M3.1 — ConvNet theory & the math

:::{admonition} Introduction to ConvNets with Setosa and Kernels (8:26)
:class: note dropdown
- The MNIST dense network falls apart the moment a digit isn't centered — the pixel-to-output correlations shift — so we need a technique that doesn't care *where* the thing is.
- An image is a **matrix of pixel values**, 0 (black) to 255 (white), proven live by mousing over a blurry LinkedIn photo on setosa.io.
- A 3×3 kernel (Sharpen, then left Sobel) slides across the image with a stride of 1, doing element-wise multiplication and summation to **recode nine pixels into one**.
- The output feature map is **smaller** than the input — that shrink is called **convolution**.
- Kernels already look like feature detectors: eyes pop, edges light up.
:::

:::{admonition} Convolutional layers / light downsampling (7:56)
:class: note dropdown
- The pasta-maker mental model: one dog goes in, multiple recoded copies come out — a conv layer with 32 feature maps makes 32 differently-recoded images.
- The key upgrade over the setosa demo: **the kernel values are themselves trainable parameters**, learned by backpropagation.
- Channels: grayscale is 1, color is an RGB stack of 3, and nothing stops you stacking 10 (satellite bands, census layers) — the kernel slides over the whole brick.
- A 5×5 image convolved by a 3×3 kernel gives 3×3; 28×28 with a 5×5 kernel gives 24×24 — **light downsampling**.
:::

:::{admonition} Pooling (no trainable parameters) (6:26)
:class: note dropdown
- Convolution's feature maps are still big, so pooling compresses them: a 2×2 window flips across the map like a **domino**, keeping only the max.
- Nothing is learned — **zero trainable parameters** — and it moves by its own size, so 24×24 becomes 12×12: *aggressive* downsampling.
- The payoff is **spatial invariance**: kernels light up wherever the feature is, no perfect centering required.
- That's why a ConvNet needs ~20,000 parameters where the dense MNIST net needed hundreds of thousands.
- AlexNet closes it out: conv/pool stacks are automated feature engineering, and the dense layers at the end are the part you already know.
:::

:::{admonition} Where we are going with ConvNets (7:50)
:class: note dropdown
- A real cat photo is 320×400×3 — pages in a book — so kernels become **M×N×3 bricks**, replicated across channels.
- Feature maps from successive conv layers emphasize eyes, fur texture, nose — learned by backpropagation, not designed.
- First outing of the formula $((M\times N\times L)+B)\times F$, with the warning that **L is the one that bites**: halfway through the network it's whatever the previous conv layer produced.
- Transfer-learning teaser: take AlexNet's free weights, chop off the dense head, keep the feature detectors, train your own head.
- The gallery of learned kernels (fish scales, car paint) emerged on their own.
:::

:::{admonition} Math for a ConvNet (Pt 1, updated) (8:34)
:class: note dropdown
- The whole chain by hand on the **5-map architecture**: 28×28×1 → Conv2D(3 maps, 5×5): size 28−(5−1)=24, params $((5\times5\times1)+1)\times3 = 78$ → pool to 12×12×3.
- Conv2D(5 maps, 3×3): each filter is a 3×3×**3** brick, $((3\times3\times3)+1)\times5 = 140$ → pool to 5×5×5 → flatten to 125.
- Dense 256 = **32,256** → output **514**. Total **32,988** — a tenth of the dense MNIST net, and the dense layers are where the parameters live.
- A filter's depth (L=3) is **inherited** from the maps coming in; the count (F=5) is **chosen**.
- Ends with the "ugly numbers" preview (29 and 87) and the rant against textbooks where every number is 3.
:::

### M3.2 — Cats vs. dogs + interpretability

:::{admonition} Welcome to cats vs. dogs, Pt 1: intro & prepping your data (5:59)
:class: note dropdown
- Use a **GPU** — conv and dense layers use the T4 automatically; just set the Colab runtime (a TPU is for bigger jobs).
- Pull `cats-dogs-sep.zip` from the OPIM5509Files repo and unzip to `cat_dog` with train/validation/test paths.
- **Folder names ARE the labels** — the ImageDataGenerator takes class names from the `cats/` and `dogs/` subfolders, so copy the structure exactly (lowercase, plural).
- 2,000 / 1,000 / 1,000 three-channel color images, not perfectly curated like MNIST.
- The cat can be anywhere: convolutional layers slide across the image and light up the feature map wherever it is.
:::

:::{admonition} Cats and Dogs, Pt 2: data generators & flow from directory (7:56)
:class: note dropdown
- Images arrive in different sizes and get resized to **150×150×3**; assume they're all color.
- Low-level to specific: early layers hunt edges and textures, later ones eyes and noses; 3×3 kernels make 32 maps and pooling halves each one, down to 7×7×128.
- The stack is flattened into one row, feeding a ~3M-parameter dense layer and a single **sigmoid** output.
- `model.fit` alone isn't enough — `flow_from_directory` feeds batches and divides pixels by 255 (min-max scaling that keeps training from diverging); **three** generators this year, including a held-out test set.
- Labels are alphabetical: **cat = 0, dog = 1**; each batch of 20 carries images plus labels.
:::

:::{admonition} Cats and Dogs, Pt 3: steps per epoch, model fit & results (4:45)
:class: note dropdown
- **Steps per epoch** = images ÷ batch size — 100 images at batch 20 means five steps; compute it from the partition instead of hard-coding.
- **Early stopping on `val_loss`**; binary crossentropy pushes predictions toward 0/1, so loss and accuracy needn't move in lockstep.
- The fit call: train generator, steps per epoch, validation generator and steps, plus the early-stopping callback — stops at epoch 28 and restores epoch 13.
- Save as **`.keras`**, the modern Keras format.
- Read the curves: training memorizes past the sweet spot; validation tells you where to freeze it.
:::

:::{admonition} Cats and Dogs, Pt 4: data augmentation & a better, more stable model (5:37)
:class: note dropdown
- Small-data problems are the business norm — **data augmentation** makes zoomed, rotated, stretched versions of each training image to stretch the utility of a small partition.
- The knobs: flips, rotations up to 40°, zooms and crops, applied randomly as the generator runs.
- **Training partition only** — like SMOTE, augment train and evaluate on the real validation and test data.
- Validation takes off: validation loss keeps falling (0.6 → 0.57 → 0.52 → 0.4) and the accuracy curve climbs gently.
- Dropout would help more, but it slows learning — give the model a longer epoch runway.
:::

:::{admonition} Wrapping up cats and dogs: classification report & confusion matrix (5:33)
:class: note dropdown
- Give yourself credit: 2,000 training images is a small-data problem; ~95% is possible but hard.
- Read the plateau — this loss curve hasn't flattened yet, so there's still room to improve.
- **Evaluate every partition** with `flow_from_directory` — all within about 10% of each other.
- Go beyond "80% accuracy": build a **classification report and confusion matrix** with named classes so you can talk precision and recall per class.
- Recall by hand: dog ≈ 346/(154+346) = 69%, cat ≈ 400/500 = 80% — a little better at cats.
:::

:::{admonition} Interpreting Cats and Dogs with Grad-CAM (6:11)
:class: note dropdown
- The polar-bear trap: a model can score well while ignoring the bears and reading snow vs. trees — you have to look at how it decides.
- Load the saved model from GitHub with `load_model`, apply it to the test partition, and confirm the ~3M trainable parameters.
- **Feature maps light up** layer by layer: early conv layers still look like the dog, deep layers turn into abstract boxes, some maps never fire.
- **Grad-CAM** backprops the class score to see which pixels drove it — the dog's face lights up on a P(dog)=1 hit.
- A confident **miss**: cats on a doghouse called "dog" because the heat sits on the doghouse — pull the misses and study them.
:::

### M3.3 — Transfer learning & autoencoders

:::{admonition} Introduction to transfer learning: feature extraction vs. fine-tuning (4:30)
:class: note dropdown
- Why not from scratch: 2,000 images takes 15–20 minutes; 100k+ would take forever, and small data is the norm.
- ConvNets are **texture detectors** — lines, curves, scales, bumps — and only the dense head names the cat, dog, car or boat.
- Borrow the giants: open-source ImageNet-class models trained on thousands of classes and millions of images.
- **Feature extraction**: run images through the frozen conv base and keep the features just before flatten.
- **Fine-tuning**: also unfreeze the last conv block so the high-resolution detectors adapt to your problem — accuracy in the 90s ahead.
:::

:::{admonition} Feature extraction: freeze the ConvNet, then train the classifier (6:05)
:class: note dropdown
- A model trained on 10,000 categories is reused by retraining only its dense layers — everything before them stays frozen.
- Import **VGG16**: ImageNet weights, 150×150 input, `include_top=False` so it doesn't predict 1,000 classes; the base squishes to 4×4×512.
- Plinko, not training: the frozen base just transforms images; `extract_features` runs all 2,000 train plus validation and test in batches of 20.
- Flatten to 8,192 — hard-coded here for VGG16; soft-code the shape in the real world.
- The plateau: validation accuracy in the 90s after ~8–10 epochs, then it flattens — you've squeezed out what this architecture has.
:::

:::{admonition} Fine-tuning the conv layer AND the dense layer (6:27)
:class: note dropdown
- VGG16's 13 conv layers → **30 trainable weight tensors** (a kernel and a bias each, plus the two new layers); ~16M parameters.
- **THE key line**: freeze the convolutional base, or `model.fit` treats it as random weights and destroys the feature detectors.
- Unfreeze **block5**: blocks 1–4 stay frozen feature detectors; the fifth block's kernels get customized to the small problem.
- Adjust the block name for other pretrained models — set trainable only where it matches.
- About 95% — the fine-tuned model does better; the jumpy training is something dropout would smooth. *(Saved as `.h5` on camera; the notebook now saves `.keras`.)*
:::

:::{admonition} Wrapping up transfer learning (1:59)
:class: note dropdown
- Exponential smoothing or dropout would tame the jumpy curves without changing the pattern.
- A labeled confusion matrix on the balanced test data: both classes predicted well, most of the data on the diagonal.
- Start with your own data and get a 60s–70s baseline before reaching for a pretrained model.
- Then transfer: feature-extract on the dense head, then fine-tune the last conv block too.
- Present the whole progression to your manager as one arc.
:::

:::{admonition} Introduction to autoencoders (7:55)
:class: note dropdown
- Squish, then rebuild: 784 pixels compressed to a nonlinear 32-number code that recreates the image — compression, anomaly detection and clustering follow.
- Prep MNIST: `imshow` a digit, divide by 255 for min-max scaling, reshape to 784 (the `-1` counts the rows).
- The **Functional API**, a step up from Sequential: a 784 input, a 32-unit encoding, a decode back to 784 — a bow tie.
- Why **sigmoid**: pixel values live in [0, 1], so a sigmoid output plus binary crossentropy is the right modeling trick.
- Ten random draws: originals on top, reconstructions below — a little fuzzy, but the 1×32 code carries the digit.
:::

:::{admonition} Applications of autoencoders (7:57)
:class: note dropdown
- A **deep autoencoder** with extra layers on both sides of a latent "code" you can pull out and reuse; trained on clean digits with patience 5.
- **Reconstruction error as a score**: take the 95th percentile of per-image error on MNIST — anything above it is anomalous.
- Fashion-MNIST can't rebuild: a 7 recreates a 7, a shoe can't; most of the sneakers and purses are flagged as never-seen.
- **The embedding as features**: feed the 32-dim code to a softmax classifier for ~95% on 0–9 — lighter and more private than 784 pixels.
- **Denoise and cluster**: noisy in, clean out extracts the signal; **t-SNE** (a nonlinear PCA) shows the digits grouping themselves.
:::
