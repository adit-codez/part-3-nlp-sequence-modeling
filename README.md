# Part 3: NLP and Sequence Modeling Mini Project
## Customer Support Ticket Sentiment Analysis

---

## Problem Statement

Build a complete NLP pipeline to classify customer support ticket messages
into three sentiment categories: **positive**, **neutral**, and **negative**.
Compare traditional text vectorization approaches against sequence-based deep
learning architectures.

---

## Dataset

- **Source**: Customer support ticket messages
- **Records**: 1,500 tickets (100-record sample embedded in notebook)
- **Features**: `ticket_id`, `channel`, `customer_message`, `sentiment_label`, `word_count`, `urgent_flag`
- **Channels**: chat, phone, email, social, app
- **Classes**: negative (~35%), neutral (~36%), positive (~29%)

---

## Repository Structure

```
part-3-nlp-sequence-modeling/
├── README.md                        ← This file
├── notebook.ipynb                   ← Complete NLP pipeline notebook
├── requirements.txt                 ← Python dependencies
└── results/
    ├── task1_dataset_overview.png   ← Class dist, word counts, channel split
    ├── task3_tfidf_top_terms.png    ← Top TF-IDF terms per sentiment class
    ├── task4_confusion_matrices.png ← Confusion matrices for both baselines
    ├── model_evaluation.csv         ← Precision / Recall / F1 per class
    └── sample_predictions.txt       ← 20 sample model predictions
```

---

## Tasks Summary

| Task | Description | Key Output |
|------|-------------|------------|
| **Task 1** | Dataset understanding – EDA, class distribution | 3 visualisation charts |
| **Task 2** | Text preprocessing pipeline | Clean tokens, padded sequences |
| **Task 3** | Text vectorization – BoW, TF-IDF, Tokenizer | Feature matrices + TF-IDF chart |
| **Task 4** | Baseline models – Naive Bayes & Logistic Regression | Confusion matrices, metrics CSV |
| **Task 5** | Bidirectional LSTM architecture + numpy forward-pass | Architecture table + Keras code |
| **Task 6** | Reflection on RNNs, LSTMs, Attention, Transformers | Written analysis in notebook |

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Run all cells in order (Cell > Run All).

---

## Key Results

Both baselines achieve high accuracy on this dataset:

| Model | Accuracy | Notes |
|-------|----------|-------|
| Naive Bayes + BoW | ~1.00 | Dataset has very consistent phrasing |
| Logistic Regression + TF-IDF | ~1.00 | Best baseline overall |

> **Note**: The dataset contains many near-duplicate messages (e.g. same template with different ticket numbers), which makes classification easy for both models. On a more diverse real-world dataset, Logistic Regression + TF-IDF typically outperforms Naive Bayes.

---

## Architecture: Bidirectional LSTM

```
Input (batch, 20)
    ↓
Embedding (batch, 20, 64)
    ↓
Bidirectional LSTM (batch, 20, 128)
    ↓
Global Max Pooling (batch, 128)
    ↓
Dropout (0.3)
    ↓
Dense 64, ReLU (batch, 64)
    ↓
Dense 3, Softmax (batch, 3)
    ↓
Output: P(negative), P(neutral), P(positive)

Loss: Sparse Categorical Cross-Entropy
Optimizer: Adam (lr=0.001)
```

---

## Dependencies

- Python 3.8+
- pandas, numpy, matplotlib
- scikit-learn
- (Optional) tensorflow >= 2.10 for full LSTM training
