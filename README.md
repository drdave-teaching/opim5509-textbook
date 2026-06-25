# OPIM 5509 — *Deep Learning: A Graduate Introduction* (textbook)

A casual-but-rigorous, **code-first deep-learning textbook** built from Dr. Wanik's OPIM 5509
lecture transcripts and course notebooks. Six cohesive chapters that weave the **narrative**
(from the transcripts) with **real code** (from the `OPIM5509Files` notebooks) and added **math**.

## View it (local, nothing public)
Open in your browser:
```
C:\Users\dww05002\kaltura\OPIM5509-textbook\_build\html\index.html
```

## Rebuild after edits
```bash
python -c "import sys; from jupyter_book.cli.main import main; sys.argv=['jupyter-book','build','.']; main()"
```
(or `jupyter-book build .` if the CLI is on your PATH)

## Chapters
1. **Refresher** — EDA & the ML methodology (sklearn)
2. **Dense Neural Networks** — forward/backprop by hand → Keras; regression & classification
3. **Convolutional Neural Networks** — convolution, pooling, parameter math, transfer learning, autoencoders
4. **RNNs for Numeric Sequences** — window method, SimpleRNN/LSTM/GRU, bidirectional, Conv1D
5. **RNNs for Text** — NLP, BoW/TF-IDF, tokenizing, embeddings, text RNNs
6. **Special Topics** — U-Net image segmentation & deep recommender systems (Functional API)

## Publish (when ready)
A GitHub Actions workflow (`.github/workflows/deploy.yml`) is included. Push this folder to a
**public** repo (free GitHub Pages) and it auto-deploys. *Currently kept LOCAL only, per request.*

## Notes / next steps
- Content is **synthesized** — each chapter distills that module's lectures into one flowing read
  (the goal was "cohesive narrative, not a hot take"). Any chapter can be expanded into finer
  sub-sections if you want more lecture-level granularity.
- **Phase 2 idea:** embed each lecture's companion notebook as a *runnable* page (read the
  explanation, then run the code) using `jupyter-book`'s notebook support.
- Source transcripts: `opim5509-transcripts/scripts` · source code: `OPIM5509Files`.
