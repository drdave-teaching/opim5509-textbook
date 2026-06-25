# Chapter 1 — Refresher: EDA & the ML Methodology

:::{admonition} 🔗 Notebooks for this chapter
:class: seealso dropdown
Open in Colab and **Runtime → Run all** — data loads from a stable link, nothing to upload.

- **General EDA Template** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/0_General_EDA_Template.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/0_General_EDA_Template.ipynb)
- **Boston EDA** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/1_Boston_EDA.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/1_Boston_EDA.ipynb)
- **All The Models Boston Housing Regression** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/2_AllTheModels_BostonHousing_Regression.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/2_AllTheModels_BostonHousing_Regression.ipynb)
- **All The Models Boston Housing Classification** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/3_AllTheModels_BostonHousing_Classification.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/3_AllTheModels_BostonHousing_Classification.ipynb)
- **Assignment1 OPIM5509** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/Assignment1_OPIM5509.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/Assignment1_OPIM5509.ipynb)
- **M1 California Housing EDA Regression** &nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/M1_CaliforniaHousing_EDA_Regression.ipynb) &nbsp; [GitHub](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/M1_CaliforniaHousing_EDA_Regression.ipynb)
:::


Let me level-set. This course assumes you arrive with **strong data-wrangling skills** and a working background in **machine-learning concepts**. So this first chapter isn't new material — it's a deliberate refresher to make sure we *all* share the same foundation before we transition into deep learning. If this chapter feels overwhelming, that's a useful signal: shore up the data-science fundamentals (or take OPIM 5512) before going further, because deep learning sits *on top* of everything here. Building deep models when you can't summarize a dataset or describe what's happening in a business is like flying a plane before you can ride a bike.

Here's the whole methodology in one breath — and it's the same five steps whether the model is a random forest or a 20-layer neural network:

```{admonition} The five-step ML methodology
:class: tip
1. **Read & clean** the data (mind the *shape*, the *dtypes*, and the missing values).
2. **Split** into train / test partitions.
3. **Scale** the features (fit on train, apply to test — never the other way around).
4. **Fit** a baseline model and a few stronger models.
5. **Evaluate** quantitatively (metrics) *and* visually (plots) — they catch different problems.
```

We'll walk it end to end on the **Boston Housing** dataset, first as a regression problem, then as a classification problem.

## 1.1 Exploratory data analysis

The first step when analyzing any dataset is an **exploratory data analysis (EDA)**. Set up your modules, read the data, and immediately interrogate its **shape**, **columns**, and **data types**:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/drdave-teaching/OPIM5509Files/refs/heads/main/OPIM5509_Module1_Files/data/BostonHousing.csv"
df = pd.read_csv(url)

print("Shape:\n", df.shape, "\n")
print("Columns:\n", df.columns, "\n")
print("Dtypes:\n", df.dtypes)
df.info()   # shape, dtypes, AND a missing-values report in one call
```

Why do I harp on **shape**? Because tracking it keeps you honest through the whole pipeline. If you read in 1,000 rows and later your model trains on 843, you should be able to say *why* — maybe 17 rows were dropped for missing values, the rest went to the test partition. Losing track of rows is how silent bugs creep in.

And why **dtypes**? Because a single stray character — a stray letter in a numeric column, an erroneous date string — will flip a column from `float64` to `object`, and now you "can't do math on it." `df.info()` is your friend here: it reports dtypes, the row count, *and* the missing values per column in one shot. If you find a misbehaving column, stop and fix it *now*, before modeling — coerce the errors and convert back to numeric.

Beyond the basics, a good EDA computes summary statistics (mean, median, percentiles), and makes plots and tables — histograms, boxplots, scatterplots with trend lines, a correlation heatmap. The goal is to *understand* the data before you let a model loose on it.

## 1.2 Regression with scikit-learn

Now the five steps in code. The target in Boston Housing is `medv` (median home value); everything else is a feature.

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. target and features
Y = df['medv']
X = df.drop('medv', axis=1)

# 2. split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# 3. scale — fit on TRAIN, transform both
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

```{admonition} Key idea — fit the scaler on TRAIN only
:class: important
**Min-max scaling** squeezes each feature to $[0,1]$:

$$ x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} $$

The `min` and `max` must come from the **training data only**. If you `fit_transform` on the full dataset, information from the test set leaks into training and your evaluation is no longer honest. So: `fit_transform(X_train)`, then `transform(X_test)`. This single discipline — *learn the transformation on train, apply it to test* — will reappear in every chapter of this book.
```

With curated, scaled data, fitting a model is famously **three lines** — instantiate, fit, predict — so we may as well fit several and compare them against a **baseline** (for regression, a plain linear regression):

```python
models = {
    "LinearRegression":          LinearRegression(),
    "DecisionTreeRegressor":     DecisionTreeRegressor(random_state=42),
    "RandomForestRegressor":     RandomForestRegressor(random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"{name:28s}  R2={r2_score(y_test, preds):.3f}  "
          f"MAE={mean_absolute_error(y_test, preds):.3f}  "
          f"RMSE={mean_squared_error(y_test, preds, squared=False):.3f}")
```

Notice where things live, because fluency means knowing the library geography cold: `MinMaxScaler` is in `sklearn.preprocessing`, `train_test_split` in `model_selection`, `RandomForestRegressor` in `ensemble`, but `DecisionTreeRegressor` in `tree`. Everyone Googles and copy-pastes — the world runs on Stack Overflow — but if you're Googling the *basics* every time, that's amateur hour. Practice ten minutes a day and this comes out of your fingertips.

**Evaluate two ways.** The metrics give you a number to compare on:

- **$R^2$** — fraction of variance explained (1.0 is perfect… suspiciously perfect).
- **MAE** $= \frac{1}{n}\sum|y_i-\hat y_i|$ — average error in the target's units.
- **RMSE** $= \sqrt{\frac{1}{n}\sum(y_i-\hat y_i)^2}$ — like MAE but punishes large misses.

The plot catches what the number hides. Plot predicted vs. actual; a good model hugs the 45-degree line:

```python
import matplotlib.pyplot as plt
preds = models["RandomForestRegressor"].predict(X_test)
plt.scatter(y_test, preds, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual medv"); plt.ylabel("Predicted medv"); plt.title("Random Forest — fit")
plt.show()
```

If the training $R^2$ is 1.0 and every point sits exactly on the line *on the training set* but the test set is a mess, you're looking at **overfitting** — a textbook case you can *see* before you can prove it numerically. That's why we always look at both, and always compare train against test.

## 1.3 Classification with scikit-learn

Here's the big secret of data science: **classification and regression are almost the same workflow.** Swap `RandomForestRegressor` for `RandomForestClassifier`, instantiate–fit–predict exactly as before, and the *only* real difference is the **evaluation**.

Boston Housing is a regression dataset, so to demo classification we recode the target into a balanced binary label — `1` if a home is priced above the **median**, else `0`. Why the median? Because it splits the data roughly 50/50, giving us **balanced classes**:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# recode to a balanced binary target
df['target'] = (df['medv'] > df['medv'].median()).astype(int)
print(df['target'].value_counts())   # ~256 vs ~250 — nicely balanced

Y = df['target']
X = df.drop(['medv', 'target'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)

clf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
preds = clf.predict(X_test)
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))
```

```{admonition} Why balanced data matters
:class: warning
Picture predicting weather in Southern California: it's sunny ~90% of the time, so a model that *always* says "sunny" scores 90% accuracy while being completely useless — it can't beat **persistence**. That's why **accuracy alone lies** on imbalanced data, and why we look at the **confusion matrix** and the metrics derived from it.
```

For classification, the baseline isn't linear regression — it's a **logistic regression** — and the metrics change from $R^2$/MAE/MSE to **accuracy, precision, recall, and F1**. All of them are just rearrangements of the four cells of the confusion matrix — true positives (TP), true negatives (TN), false positives (FP), false negatives (FN):

$$
\text{Precision}=\frac{TP}{TP+FP}, \qquad
\text{Recall}=\frac{TP}{TP+FN}, \qquad
F_1 = 2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
$$

The `classification_report` hands you all of these per class. (A small mnemonic that saves headaches when you read sklearn's confusion matrix: **recall reads along the row** of the true class.)

## Wrap-up

Everything in this chapter — read, split, **scale on train only**, fit a baseline plus stronger models, evaluate with numbers *and* pictures — is the scaffold we will hang every deep-learning model on. The architectures get fancier; the methodology does not change. If you can do this fluently on Boston Housing, you're ready to build your first neural network.

```{admonition} Key takeaways
:class: tip
- The **five-step methodology** is model-agnostic: read → split → scale → fit → evaluate.
- **Scale on train, transform test** — your first and most important defense against data leakage.
- **Regression vs. classification** differ mainly in the *evaluation* (R²/MAE/RMSE vs. confusion-matrix metrics).
- Always evaluate **quantitatively and visually**, and always compare **train vs. test** to catch overfitting.
- Know your **library geography** (`preprocessing`, `model_selection`, `ensemble`, `tree`) — fluency beats Googling the basics.
```


---

## 📌 Lecture key points

*Distilled takeaways from the video lectures behind this chapter — click each to expand.*


:::{admonition} Setting up Colaboratory on your Google Drive (M1.1)
:class: note dropdown
- Use **Google Colab** — cloud-based so there are no local hardware/install headaches; "if you can get online and have a Gmail, you'll succeed."
- Make a **dedicated class Gmail/Drive** (e.g., `davesdeeplearning@gmail.com`) for 15 GB of free, shareable storage; academic Gmail can block easy sharing.
- Organize a class folder with **per-module subfolders**; connect Colab and use **Chrome**.
- Colab gives a free Python 3 runtime (~12 GB RAM, ~100 GB disk) — no setup cost to start modeling.
- The whole course is built so everything **runs from Drive** — reproducibility and shareability first.
:::

:::{admonition} Accessing books from the UConn library and notebooks from GitHub (M1.2)
:class: note dropdown
- **Don't buy textbooks** — get them free via `lib.uconn.edu` with your NetID (Manning/O'Reilly access).
- Core books: **Trask, *Grokking Deep Learning*** (best for the intuition-building start) and **Chollet, *Deep Learning with Python***.
- Spend the saved money on a **second monitor** instead — biggest productivity upgrade.
- Pull reference **notebooks from GitHub** straight into Colab.
- Build the habit of reading source books + running their notebooks alongside lecture.
:::

:::{admonition} Performing an EDA on Boston Housing — Part 1 (M1.3)
:class: note dropdown
- **Level-set:** you need strong wrangling + ML fundamentals before deep learning ("fly a plane only after you can ride a bike").
- Always interrogate **shape, columns, and dtypes** first; `df.info()` gives shape + dtypes + missing values in one call.
- A stray character flips a numeric column to **object** ("can't do math on it") — fix dtypes *before* modeling.
- **Track row counts** through the pipeline (split, missing-value drops) and be able to explain every change.
- EDA = summary stats + plots/tables to *understand* the data before modeling.
:::

:::{admonition} Performing an EDA on Boston Housing — Part 2 (M1.4)
:class: note dropdown
- Be **brave**: "Runtime → Run all"; you won't break anything, and restart-and-run-all always recovers.
- Standard imports every time: `pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`.
- Compute **statistics** (mean, median, percentiles) and make **plots** (boxplots, histograms, KDE, scatter).
- Read data from a **gdown shareable link** rather than mounting Drive (cleaner, no PII exposure).
- Reusable **EDA template** — same skeleton applied to any dataset.
:::

:::{admonition} Fitting Regression Models with Sci-kit Learn (Boston Housing) (M1.5)
:class: note dropdown
- The **5 steps**: read/clean → split → **min-max scale (fit_transform on train, transform on test)** → fit → evaluate.
- Fit a **baseline (LinearRegression)** plus tree models (DecisionTree, RandomForest, GradientBoosting); fitting is ~3 lines.
- Evaluate **quantitatively** (R², MSE, MAE) **and visually** (predicted-vs-actual on the 45° line).
- **R² of 1.0 + perfect line = overfitting**, especially if train ≠ test.
- Know your **library geography** (`preprocessing`, `model_selection`, `ensemble`, `tree`) — fluency beats Googling basics.
:::

:::{admonition} Fitting Classification Models with Sci-kit Learn (Boston Housing) (M1.6)
:class: note dropdown
- Classification ≈ regression workflow; swap `RandomForestRegressor` → `Classifier` — the **evaluation** is what differs.
- **Recode** `medv` to a balanced 0/1 at the **median** (~50/50) so accuracy is meaningful.
- Baseline shifts from linear → **logistic regression**.
- Metrics come from the **confusion matrix**: accuracy, **precision, recall, F1** (all rearrangements of TP/TN/FP/FN).
- **Balanced data matters for honest metrics** (the "always sunny in SoCal = 90% accuracy" trap).
:::
