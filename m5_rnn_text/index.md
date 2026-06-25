# Chapter 5 — Recurrent Networks for Text

We arrived at text on purpose *last*. Everything from Chapter 4 — sequences fed one step at a time, a hidden state carried forward, LSTM/GRU/bidirectional layers — applies directly to words. The one genuinely new problem is that **a computer can't read words**; it can only do math on numbers. So most of this chapter is about turning language into numbers three different ways, from naive to powerful: **bag-of-words → TF-IDF → embeddings**.

Our running examples are real and a little fun: National Weather Service **storm narratives** (hail and flash-flood event descriptions), **Trump vs. Obama tweets** (who wrote it?), and **IMDB** movie-review sentiment.

## 5.1 Getting text into modeling shape

Before any representation, you **preprocess**. A quick vocabulary note first: the **corpus** is the entire collection of text; a **document** is one sample (one narrative, one tweet). The standard cleaning steps:

- **Lowercase everything** — "Hail", "HAIL", and "hail" should map to the same token. Overwrite the column in place with `.lower()`.
- **Strip funky characters** — punctuation, `@`, `$`, `%`, anything that isn't `a–z` becomes a space (so "media," and "media" aren't different words).
- **Remove stop words** — drop high-frequency, low-signal words ("the", "on", "a") using a stop list.
- **(Optionally) stem** — collapse "running/ran/runs" toward a common root. Stemming helps most when you have **too many words and not enough rows**.

```python
import re
df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(lambda s: re.sub(r'[^a-z]+', ' ', s))   # keep letters only
```

Then a little EDA — most common words, a bar plot, a word cloud — and you're ready to vectorize.

## 5.2 Bag-of-words and TF-IDF

The simplest numeric representation is **bag-of-words (BoW)**: for "the cat sat on the mat," ask of each word *does it appear, and how often?* sklearn's **`CountVectorizer`** builds a matrix with one column per vocabulary word and counts in the cells. It throws away order entirely — a "bag" — but it's a strong, fast baseline.

The refinement is **TF-IDF** (term frequency–inverse document frequency), which down-weights words that appear in *every* document (uninformative) and up-weights words that are frequent in a document but rare across the corpus (distinctive):

$$
\text{tfidf}(t,d) = \underbrace{\text{tf}(t,d)}_{\text{count in doc }d}\;\times\;\underbrace{\log\frac{N}{1+\text{df}(t)}}_{\text{rarer}\,\Rightarrow\,\text{bigger}}
$$

where $N$ is the number of documents and $\text{df}(t)$ is how many documents contain term $t$.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_tfidf = tfidf.fit_transform(train_text)     # fit on TRAIN only (Chapter 1 rule!)
```

```{admonition} n-grams buy a little word order — at a price
:class: warning
A bag-of-words loses sequence, but **n-grams** smuggle some back: **unigrams** are single words; **bigrams** are adjacent pairs ("the cat", "cat sat"); **trigrams** are triples. Set `ngram_range=(1,3)` and you capture short phrases. The cost is **dimensionality**: the larger the window, the more columns explode, and you can easily end up with **more columns than rows** — the curse of dimensionality. Cap it with `max_features` and watch your shapes.
```

Feed the TF-IDF matrix into any classifier from Chapter 1 — or a dense net from Chapter 2 — and you have a working text model. But notice what's still missing: **sequence**. BoW and n-grams approximate it; embeddings + RNNs *use* it.

## 5.3 Tokenizing — words to integers

To preserve sequence, first map each unique word to an integer index with Keras's **`Tokenizer`**. If there are 13,512 unique words, each gets a number from 1 to 13,512; a document becomes a *list of integers* in its original order.

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

t = Tokenizer(num_words=10000)            # keep the 10k most frequent words
t.fit_on_texts(X_train)                   # learn the vocabulary on TRAIN
seqs = t.texts_to_sequences(X_train)      # "hail damage reported" -> [42, 187, 9]
X_pad = pad_sequences(seqs, maxlen=100)   # pad/truncate to a fixed length
```

Two knobs that matter: `num_words` caps the vocabulary (rarer words get dropped — try 1,000 / 10,000), and `maxlen` fixes the sequence length so every document is the same shape (short ones get **padded**, long ones **truncated**).

## 5.4 Word embeddings — the real win

Integer indices are arbitrary — word #42 isn't "twice" word #21. We want word #42 to carry *meaning*. An **embedding** is a learned, dense vector for each word; words used in similar contexts end up with similar vectors. The `Embedding` layer is simply a **lookup table** of shape `(vocab_size, embedding_dim)` that the network **trains along with everything else** (or you can load pre-trained vectors like GloVe):

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=32, input_length=100))  # word -> 32-d vector
model.add(LSTM(32))                 # read the sequence of vectors, carry a hidden state
model.add(Dense(1, activation='sigmoid'))   # binary sentiment / authorship
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
```

```{admonition} Two ways to consume an embedding
:class: tip
- **Flatten into a dense layer** — `Embedding → Flatten → Dense`. Fast, but it treats the document as a *bag of vectors* and loses order again.
- **Feed a recurrent layer** — `Embedding → LSTM/GRU` (optionally bidirectional, optionally with `Conv1D` in front). This *uses* the sequence: the LSTM reads word-vectors one at a time, carrying context — exactly the machinery of Chapter 4, now over words instead of temperatures.
```

With this stack you can do real tasks in a few lines: classify **who tweeted** (Trump vs. Obama), score **IMDB** sentiment, or label a **storm narrative**. And every advanced trick from Chapter 4 — stacking, bidirectional, `Conv1D` + pooling, recurrent dropout — transfers directly, because under the hood it's the same recurrent engine reading a sequence of vectors.

## Wrap-up

```{admonition} Key takeaways
:class: tip
- Text must become numbers; **preprocess** first (lowercase, strip non-letters, stop words, optional stemming). Mind **corpus vs. document**.
- **Bag-of-words / TF-IDF** are strong, order-free baselines; **n-grams** add local order but blow up dimensionality (watch for more columns than rows).
- **Tokenize** to integer sequences and **`pad_sequences`** to a fixed length.
- **Embeddings** are learned dense word-vectors (a trainable lookup table); they give words *meaning*.
- **`Embedding → Flatten → Dense`** ignores order; **`Embedding → LSTM`** uses it — the Chapter-4 recurrent engine, now reading words.
- Always **fit vectorizers/tokenizers on train only** — the same anti-leakage discipline from Chapter 1.
```


---

## 📌 Lecture key points

*Distilled takeaways from the video lectures behind this chapter — click each to expand.*


:::{admonition} Basic NLP processing techniques
:class: note dropdown
- **Corpus** = whole sample; **document** = individual sample.
- Lowercase everything so "Hail"/"hail" are one token.
- Strip non-letters (punctuation, @, $, %) → replace with spaces.
- Goal: turn text into something a computer can model.
- Done on the storm (hail) narratives as the running example.
:::

:::{admonition} CountVectorizer and Bag of Words
:class: note dropdown
- **Bag-of-words** counts word occurrences; `CountVectorizer` builds the matrix.
- Remove **stop words**; optionally **stem** (best when too many words, too few rows).
- **Tokenize** into indexed words (e.g., 13,512 unique → indices 1–13,512).
- Reusable script if you name things generically (`df`).
- A little EDA: common words, bar plot, word cloud.
:::

:::{admonition} Build a model with TF-IDF and the keras tokenizer
:class: note dropdown
- **TF-IDF** up-weights distinctive words, down-weights ubiquitous ones.
- Keras **`Tokenizer(num_words=...)`** learns the vocabulary and indexes words.
- `fit_on_texts` on train; inspect `word_index`, `word_counts`, `document_count`.
- `num_words` caps vocabulary (try 1k/10k).
- Feed vectors into a model for classification.
:::

:::{admonition} Building and evaluating models with BoW and TF-IDF
:class: note dropdown
- Compare BoW vs TF-IDF representations on the same task.
- Feed into standard classifiers; evaluate with the usual metrics.
- Strong, fast **baselines** before deep models.
- Watch dimensionality with n-grams.
- Fit vectorizers on **train only**.
:::

:::{admonition} Intro to NLP topics with ML
:class: note dropdown
- Overview of the classic NLP pipeline (preprocess → vectorize → model).
- Naive representations: presence (one-hot) and frequency (counts).
- **n-grams** (unigram/bigram/trigram) smuggle back some word order.
- Larger n → bigger feature space → dimensionality risk.
- Sets up why embeddings are better.
:::

:::{admonition} Taking a DL approach to structured NLP data
:class: note dropdown
- Move from sparse BoW/TF-IDF to **dense** learned representations.
- Prep tokenized, padded sequences for a network.
- Bridge from classic ML to deep text models.
- Same preprocessing, different downstream model.
- Motivates embeddings next.
:::

:::{admonition} Exploring embeddings (flattened) in dense layers
:class: note dropdown
- An **`Embedding`** layer = trainable lookup table (word → dense vector).
- **Flatten → Dense** consumes embeddings but **loses word order** (bag of vectors).
- Fast and simple; a stepping stone.
- Embedding dimension is a hyperparameter.
- Words gain *meaning* vs arbitrary indices.
:::

:::{admonition} Introduction to embeddings for text sequences
:class: note dropdown
- A computer must convert words to numbers; naive one-hot/counts lack **sequence**.
- Embeddings place words in n-D space where similar words sit nearby.
- Numeric sequences taught first **on purpose** — text is the harder generalization.
- Recall the recurrent machinery: read one token at a time, update hidden state.
- Can **import (GloVe)** or **learn** embeddings for your task.
:::

:::{admonition} Using embeddings in recurrent layers
:class: note dropdown
- **`Embedding → LSTM/GRU`** *uses* order (vs flatten→dense which ignores it).
- The Chapter-4 recurrent engine, now reading **word-vectors**.
- All advanced tricks transfer (stacking, bidirectional, Conv1D).
- The strongest classic text model in the course.
- Applies to sentiment/authorship tasks.
:::

:::{admonition} Monster! Text analytics + the kitchen sink
:class: note dropdown
- A capstone notebook combining preprocessing + embeddings + recurrent layers.
- Throw the full toolkit at a real text problem.
- Shows how the pieces assemble end-to-end.
- Manage complexity with clean, reusable code.
- Evaluate honestly (train-only fits, proper test).
:::

:::{admonition} Put it all together! Download data, process it, model it
:class: note dropdown
- Full pipeline: **scrape/download → process → model** (e.g., Trump vs Obama tweets, GetOldTweets).
- Authorship/sentiment classification from raw text.
- Reinforces reproducible, end-to-end workflow.
- Real, messy data.
- Capstone for text RNNs.
:::
