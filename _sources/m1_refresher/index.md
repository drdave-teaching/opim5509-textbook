# Chapter 1 — Refresher: EDA & the ML Methodology

:::{admonition} 🔗 Notebooks for this chapter
:class: seealso dropdown
- **Welcome & Setup** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/0_Welcome_and_Setup.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/0_Welcome_and_Setup.ipynb)
- **California Housing EDA** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/1_CaliforniaHousing_EDA.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/1_CaliforniaHousing_EDA.ipynb)
- **All The Models — Regression** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/2_AllTheModels_Regression.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/2_AllTheModels_Regression.ipynb)
- **All The Models — Classification** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/3_AllTheModels_Classification.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/3_AllTheModels_Classification.ipynb)
- **General EDA Template** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/4_General_EDA_Template.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/4_General_EDA_Template.ipynb)
- **Appendix — ROC, AUC & Thresholds** *(optional)* &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/5_Appendix_ROC_AUC_and_Thresholds.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/5_Appendix_ROC_AUC_and_Thresholds.ipynb)
:::

Let me level-set. This course assumes you arrive with **strong data-wrangling skills** and a working background in **machine-learning concepts**. So this first chapter isn't new material — it's a deliberate refresher to make sure we *all* share the same foundation before we transition into deep learning. If this chapter feels overwhelming, that's a useful signal: shore up the data-science fundamentals (or take OPIM 5512) before going further, because deep learning sits *on top* of everything here. Building deep models when you can't summarize a dataset or describe what's happening in a business is like flying a plane before you can ride a bike.

So what *is* deep learning, and why do you need it? Traditional methods like random forests rely on data in a very structured format. Neural networks work with structured data **and** unstructured data — text, time series, images, video, audio. They seem like magic, but at a high level they are just a nonlinear weighted sum of information that gets transformed and creates an output. Over the semester we open the hood on all of it.

Here's the whole methodology in one breath — and it's the same five steps whether the model is a random forest or a 20-layer neural network:

```{admonition} The five-step ML methodology
:class: tip
1. **Read & clean** the data (mind the *shape*, the *dtypes*, and the missing values).
2. **Split** into train / test partitions.
3. **Scale** the features (fit on train, apply to test — never the other way around).
4. **Fit** a baseline model and a few stronger models.
5. **Evaluate** quantitatively (metrics) *and* visually (plots) — they catch different problems.
```

We'll walk it end to end on the **California Housing** dataset, first as a regression problem, then as a classification problem.

```{admonition} Why California Housing, and not Boston?
:class: note
Boston Housing was the classic teaching set for decades, and scikit-learn **removed it in version 1.2** because one of its columns was an explicit racial proxy. A current install can't fetch it at all. California Housing loads in one line, has 20,640 rows, is mostly clean with a few excellent gotchas, and has the bonus of being **spatial** — you can plot it by latitude and longitude and color it by value.
```

## 1.1 Exploratory data analysis

The first step when analyzing any dataset is an **exploratory data analysis (EDA)**. Set up your modules, read the data, and immediately interrogate its **shape**, **columns**, and **data types**:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame

print("Shape:\n", df.shape, "\n")
print("Columns:\n", df.columns, "\n")
df.info()   # shape, dtypes, AND a missing-values report in one call
```

**Before anything else: what is a row?** In this dataset a row is a **census block**, not a house. Median income, house age, average rooms, average bedrooms — these are summary statistics describing a geographic unit. Students who forget this write nonsense in their write-ups. And the target, median house value, is in a standardized form: you are not going to buy a house for \$4.

Why harp on **shape**? Because tracking it keeps you honest through the whole pipeline. Commit 20,640 rows and 9 columns to memory, and as you move through transformations and aggregations the totals should keep making sense. You never want to silently drop rows, or explode them because of a bad join.

And why **dtypes**? Because a single stray character will flip a column from `float64` to `object`, and now you can't do math on it. `df.info()` reports dtypes, the row count, *and* missing values per column in one shot.

Then `.describe()` for summary statistics. Here is the statistical point worth pausing on: **the mean and standard deviation are most meaningful for normally distributed data.** When the standard deviation is bigger than the mean, you probably have skew. For non-normal data, lean on the percentiles — min and max give you the range, the 25th/50th/75th give you the IQR and median. Coming from an engineering background I also like the **1st and 99th percentiles**, because I'm interested in the extremes.

### The two gotchas in this dataset

**The target is right-censored.** Run `value_counts()` on the target and the most common value is 5 — that means \$500,000 *or more* — with **965 rows** sitting on that cap. Anything more expensive got smushed into the ceiling. You'd find it by plotting: the top bin looks wrong. Your model will have a lot of trouble there, and no model can predict past it.

**Average rooms reaches 141 and average occupancy reaches 1,200.** No household has 141 rooms and no house has 1,200 people in it. It happens because those blocks have tiny populations, so the ratio blows up. Not real data problems in the sense of typos — a mechanism you need to understand before deciding what to do.

### Looking at distributions and relationships

A **histogram matrix** shows every feature at once — some columns are roughly normal, others are right-tailed or bimodal. **Boxplots** surface the outliers.

```{admonition} Be careful deleting outliers
:class: warning
A lot of people say "I'm going to make a model that fits really well, so I'll get rid of all the outliers." But **the outliers might be the thing that's most important to the business.** Be aware of them, understand where they came from, and only then decide.
```

**Flag variables** are the simplest feature engineering there is. Compute the median house value, then `np.where` a 1 for above and a 0 for below, and compare the groups. Expensive blocks have higher incomes, slightly older and bigger houses, the same population, and fewer people living in them — which agrees with intuition. That's the **reasonable person test**: try to tell the story of the data without forcing a narrative, and if something comes out backwards, that's exactly what you raise in the meeting.

For **bivariate** work, the correlation matrix (Pearson or Spearman) is symmetric about the diagonal — focus on the target column and tell the story. Income is the strongest driver of house value at about 0.69. Average rooms and average bedrooms are **multicollinear**, which makes sense. And plotting income against house value shows a real linear agreement with plenty of scatter — plus that flat stripe along the top where the data is censored.

### Geography for free

Because the data has latitude and longitude, you have spatial coordinates. Plot one against the other and **the data assembles itself into the shape of California** — no base map required. Color the points by median house value and the coast lights up expensive, the interior goes cheap, with an interesting pocket up near Lake Tahoe.

```{admonition} On your own
:class: seealso
Size the bubbles by population as well as coloring by value — and sort so the **biggest bubbles draw underneath**, or the small ones disappear behind them.
```

Finally, the cleaning decision we can now justify: drop blocks with occupancy above 10 or more than 20 rooms. That removes about 100 rows and the remaining values look sensible.

## 1.2 Regression with scikit-learn

Now the five steps in code. The target is `MedHouseVal`; everything else is a feature.

```python
# 1. target and features
Y = df['MedHouseVal']
X = df.drop('MedHouseVal', axis=1)

# 2. split (80/20) — shuffle, and set a seed for reproducibility
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, shuffle=True, random_state=42)

# keep the names BEFORE scaling destroys them
feature_names = list(X_train.columns)

# 3. scale — fit on TRAIN, transform both
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

Three things worth calling out.

**Nothing mutates `df`.** There is no `inplace=True` anywhere — `df` still has all its rows, so you can re-run cells without surprises.

**Shuffle and seed.** If the rows carry structure — newest to oldest, say — then chopping off the last 20% gives you a biased sample. Shuffle when there's no time-series component, and set a random seed so your draw is reproducible.

**Save your feature names before you scale.** `MinMaxScaler` hands back a NumPy array and the column names are gone. The *positions* survive, so if you saved the names you can reattach them later for permutation importance. Otherwise it's X1, X2, X3 — and that's hard to explain in a meeting.

```{admonition} Data leakage — the mistake I see every semester
:class: warning
Fit the scaler on **train**, then `transform` the test set. Do **not** `fit_transform` both, and do **not** `fit_transform` the whole `X` before splitting — that leaks the distribution, outliers included, into the model. You want to learn patterns on a representative sample and apply them to the held-out set. And note: the test partition will **not** span the full 0–1 range. That's correct, not a bug.
```

Then fit. Instantiate, fit, predict — three lines, and only line one changes between models:

```python
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR = LR.fit(X_train, y_train)
train_preds = LR.predict(X_train)
test_preds  = LR.predict(X_test)
```

Linear regression gives an MAE around 0.5 and R² around 0.63 — a moderate fit. But a single metric never tells the full story, so write an `evaluate` function and loop a handful of models: linear regression, decision tree, random forest, gradient boosting, k-nearest neighbors.

The **decision tree overfits like crazy** on the training partition, because scikit-learn will happily let a single observation sit in a leaf. `min_samples_leaf` is the dial — the more samples you force into a terminal node, the less it overfits, and the hope is it generalizes better.

**Random forest wins out of the gate**, and there's a reason random forests do so well on tabular data: they fit many small trees, each on a random subset of the rows *and* a random subset of the columns, then average the output of those 100 or 200 trees into a more stable estimate. Where a neural network wins instead is on **unstructured** data, because it does the feature engineering for you.

Then look at the picture, not just the table. Predicted-vs-actual shows the random forest hugging the 45° line while linear regression is more of a cloud — and **both do poorly on the censored data**, which is your EDA finding coming back as residuals.

### Interrogating the model

Built-in tree importance sums Gini or entropy down the tree and reports a number. That's fine, but the method **emphasizes high-cardinality features** — a column with many unique values simply offers the forest more places to split.

**Permutation importance** is the better tool. It takes a copy of `X_test`, shuffles one column in place, and asks the model to predict on that corrupted data. If R² barely moves, the model wasn't using that column. If R² collapses from 0.9 to 0.2, the model was leaning on it hard. It relies on nothing internal to the model, so it's **model-agnostic** — you can use it on a decision tree *and* on a neural network and compare fairly.

```{admonition} The standard for this course
:class: tip
No longer are you allowed to say "I fit a model, I have no idea how it fit." You have to **interrogate and investigate** the model and show how it did.
```

## 1.3 Classification with scikit-learn

I like to take the same problem and recycle it with a twist, so the **technique** stands out and you already understand the data. California Housing is a regression dataset, so recode the target into a balanced binary label — `1` if a block is above the **median**, else `0`. Why the median? Because it splits the data roughly 50/50, giving **balanced classes**:

```python
median_value = df['MedHouseVal'].median()
df['EXPENSIVE'] = np.where(df['MedHouseVal'] > median_value, 1, 0)

y = df['EXPENSIVE']
X = df.drop(['EXPENSIVE', 'MedHouseVal'], axis=1)   # drop BOTH
```

```{admonition} Drop the raw value too
:class: warning
`EXPENSIVE` was built *from* `MedHouseVal`. Leave the raw column in `X` and you have handed the model the answer — a perfect score and a worthless model. That's **target leakage**.
```

The rest is the same skeleton with different algorithms — logistic regression instead of linear, classifier versions of the trees, KNN — and different metrics, because mean absolute error means nothing here.

A classifier gives you two things. `predict()` returns the class, and **`predict_proba()` returns the raw probability of being class 1** — and that score is a confidence reading. A model saying 0.99 is confident; a model that predicts 0.53 for everything is not. You can manipulate those probabilities by moving the threshold to get a better fit.

**Accuracy only applies when your classes are balanced.** Better tools are **precision** and **recall**. A mnemonic: recall starts with R, so it's the **row** — true positives divided by (true positives + false negatives). Precision is the **column** — true positives divided by (true positives + false positives). A confusion matrix in seaborn makes both readable; you want the mass on the diagonal.

Which error matters more is a **business** question. In IoT analytics work, of the things that were truly failing, I needed to catch them all — so I ran a model that was a little bit chatty, with some false alarms, because missing a real event was worse. Other problems want the opposite.

```{admonition} On your own
:class: seealso
Try manipulating the threshold that turns a predicted probability into a 0 or a 1, and see whether you can get a better fit. The optional appendix notebook builds an ROC curve **by hand** from twelve loan applicants and shows exactly how the false positive rate moves as you slide it.
```

## Wrap-up

Everything in this chapter — read, split, **scale on train only**, fit a baseline plus stronger models, evaluate with numbers *and* pictures — is the scaffold we will hang every deep-learning model on. The architectures get fancier; the methodology does not change. If you can do this fluently on California Housing, you're ready to build your first neural network.

## 📌 Lecture key points

*Nine videos, 51:33. Transcripts and polished scripts: [opim5509-transcripts / fall2026_idl / module1](https://github.com/drdave-teaching/opim5509-transcripts/tree/main/fall2026_idl/module1).*

:::{admonition} Welcome to IDL from Dr. Dave! (2:55)
:class: note dropdown
- Traditional methods like **random forests need structured tabular data**; neural networks handle structured *and* unstructured — text, time series, images, video, audio.
- Networks "seem kind of like magic," but they are a **nonlinear weighted sum** of information that gets transformed into an output.
- Semester arc: ML refresher → neural network bootcamp → ConvNets → RNNs → text sequences.
- **Time series comes before text** on purpose — going to text first means learning embeddings *and* sequences of embeddings at once.
- ConvNet fine-tuning is about reusing weights trained on millions of images for your own small-data problem.
:::

:::{admonition} Google Colaboratory and the UConn Library (2:42)
:class: note dropdown
- Attach Colab from Drive: **New → More → Connect more apps → search "Colab" → Install**, then refresh.
- Consider a **dedicated Gmail for class** if you want a clean 20 GB of Drive.
- The **Chollet** textbook (2021 edition) is free through library.uconn.edu — search, click "full text available," log in with your NetID.
- Dave hosts materials on GitHub and runs them in Colab; you can use another environment, but Colab is what class demos assume.
:::

:::{admonition} Welcome, Colab and GitHub (7:03)
:class: note dropdown
- **`File → Save a Copy in Drive`** puts your copy in the *Colab Notebooks* folder, where you can rename and reorganize it.
- Colab is your computer: **Python 3.12**, package manager behind the scenes, libraries already at stable versions.
- **Selecting a GPU is not enough** — Runtime → Change runtime type gets you the hardware, but your code has to be written to use it.
- The Colab runtime is **wiped when you close the browser** — anything you dragged in has to be dragged in again.
- Python sanity check: lists index from **zero** and `nums[0:3]` stops *before* 3; dictionaries hold network configs cleanly; write a function once instead of pasting code ten times.
:::

:::{admonition} Introduction to EDA on CA Housing (5:11)
:class: note dropdown
- **A row is a census block, not a house** — every "average" column is a summary statistic over a geographic unit.
- 20,640 rows and 9 columns; **commit the shape to memory** so you notice when a join explodes or drops rows.
- `df.info()` gives dtypes, row count, and missing values in one call.
- **Mean and standard deviation are most meaningful for normal data.** When the sd exceeds the mean, suspect skew and read the percentiles instead.
- Add the **1st and 99th percentiles** to `.describe()` — the extremes are where the interesting engineering lives.
:::

:::{admonition} Outliers, censored data, univariate and bivariate plots (7:00)
:class: note dropdown
- The target is **right-censored at 5** (= \$500,000 or more) with **965 rows** on the cap. You find it by plotting, and your model will struggle in that bin.
- **Average rooms hits 141, average occupancy hits 1,200** — an artifact of tiny block populations, not real houses.
- **Don't reflexively delete outliers** — they are often the most important thing to the business.
- Flag variables via `np.where`, then group-by: expensive blocks have higher income, slightly older and bigger houses, fewer occupants.
- The **reasonable person test** — tell the story of the data without forcing a narrative; if something is backwards, raise it in the meeting.
- Income drives house value (~0.69); average rooms and bedrooms are **multicollinear**.
:::

:::{admonition} Geographic EDA and final data cleaning (2:33)
:class: note dropdown
- Latitude and longitude are spatial coordinates, so **the data draws California by itself** — no base map needed.
- Color by median house value: expensive along the coast, cheaper inland, a nice pocket near Lake Tahoe.
- Bonus: size bubbles by population — and **order biggest-underneath** so small bubbles aren't hidden.
- Cleaning call: drop occupancy > 10 and rooms > 20, roughly 100 rows.
:::

:::{admonition} Intro to end-to-end ML for regression (5:50)
:class: note dropdown
- Split Y from X **without mutating `df`** — no `inplace=True` anywhere.
- `train_test_split` with **shuffle** (rows may carry order) and a **random seed** (reproducibility).
- **Save your feature names before scaling** — the array survives, the names don't, and permutation importance needs them.
- **Fit the scaler on train, transform test.** Never `fit_transform` the whole `X` before splitting — that leaks the distribution.
- The test partition **won't span the full 0–1 range**, and that's fine.
:::

:::{admonition} Fitting and evaluating ML models for regression (9:40)
:class: note dropdown
- Instantiate → fit → predict. Linear regression lands near **MAE 0.5, R² 0.63**.
- **Decision trees overfit like crazy** in scikit-learn — a single observation can sit in a leaf. `min_samples_leaf` is the dial.
- **Random forests win on tabular data** because many small trees on random subsets of rows *and* columns average into a stable estimate.
- Neural networks win instead on **unstructured** data, where they do the feature engineering for you.
- Predicted-vs-actual: the forest hugs the 45° line; **both models fail on the censored rows.**
- **Permutation importance beats Gini importance** — the built-in kind favors high-cardinality features, while permutation is model-agnostic, so a tree and a neural network can be compared fairly.
:::

:::{admonition} End-to-end ML for classification (8:39)
:class: note dropdown
- Same dataset, recycled with a twist, so the **technique** stands out instead of the data.
- Recode at the median → ~50/50 balance, and **drop both the flag and the raw house value** from `X`.
- Regressors become classifiers: logistic regression, decision tree / random forest / gradient boosting classifiers, KNN.
- **`predict_proba` is a confidence reading** — 0.99 is confident, 0.53 is not — and you can move the threshold to improve the fit.
- **Accuracy only applies when classes are balanced**; precision and recall are the better tools.
- Recall starts with R → the **row**: TP / (TP + FN). Precision is the **column**: TP / (TP + FP).
- Which error costs more is a business call — an IoT model that was "a little bit chatty" was right, because catching every true event mattered.
:::
