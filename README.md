# Amazon Review Sentiment Analysis: BERT vs BiLSTM

Fine-tuning a pre-trained BERT transformer on Amazon product reviews for binary sentiment classification, benchmarked against a BiLSTM baseline, with an honest look at transfer-learning efficiency and real-world deployment constraints.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/Transformers-4.40%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project compares two approaches to sentiment classification on Amazon product reviews:

1. **Fine-tuned BERT** (`bert-base-uncased`) — a 110M-parameter transformer pre-trained on English text, adapted to the sentiment task with a two-epoch fine-tuning schedule.
2. **BiLSTM baseline** — a bidirectional LSTM with trainable embeddings, trained from scratch on the same data.

The comparison is deliberately **multi-axis**: accuracy and F1 alone don't tell the full story. Training time, per-sample inference latency, GPU memory footprint, and model size are tracked alongside so the trade-off between "state-of-the-art accuracy" and "production-realistic cost" is explicit rather than hidden.

---

## Why compare BERT to an LSTM?

The transformer-vs-recurrent comparison remains one of the cleanest illustrations of **transfer learning** in NLP:

- **BERT** arrives with billions of tokens of pre-training; fine-tuning merely nudges it toward a specific task.
- **BiLSTM** starts from random weights; every pattern must be learned from the labelled corpus alone.

The gap between the two models on the same data is a direct measure of how much "free knowledge" the pre-trained checkpoint provides. Beyond accuracy, BERT is also ~10× larger and substantially slower per prediction — which is exactly why **distillation** and **quantisation** exist, and why a smaller model is sometimes the correct production choice despite lower benchmark accuracy.

---

## Dataset

The project uses **Amazon Polarity** (`fancyzhx/amazon_polarity`) — a binary sentiment benchmark of ~3.6M training and 400K test reviews drawn from real Amazon product reviews spanning 18 years.

| Field | Description |
|---|---|
| `content` | Review body |
| `title` | Review title |
| `label` | `0` = negative, `1` = positive |

Labels are derived from star ratings: 1–2 stars → negative, 4–5 stars → positive. Three-star reviews are excluded in the original dataset construction, so the task is **cleanly binary**.

The pipeline uses a `train[:20000]` / `test[:2000]` slice by default so a full run fits comfortably inside a single GPU session. The `TRAIN_SIZE` and `TEST_SIZE` constants at the top of the script make it trivial to scale up — a full-dataset run typically reaches 0.95+ accuracy.

> The dataset streams directly from the Hugging Face Hub via `datasets.load_dataset`. No manual download required.

---

## Methodology
Amazon Polarity (Hugging Face Hub)
│
▼
[1] Load subset: train[:20000], test[:2000]
│
▼
[2] BERT pipeline
│ · tokenize (max_length=256, truncation)
│ · bert-base-uncased + 2-class head
│ · 2 epochs, LR=2e-5, batch=32, fp16=True
│ · early eval per epoch (accuracy, precision, recall, F1)
│
▼
[3] BiLSTM pipeline
│ · Keras Tokenizer (20k vocab) → padded to 200 tokens
│ · Embedding(100-d, trainable) → BiLSTM(64) → Dense(1, sigmoid)
│ · 5 epochs, batch=128, early stopping
│
▼
[4] Comparison
│ · accuracy / precision / recall / F1
│ · wall-clock training time per model
│ · per-sample inference latency (500 test samples)
│
▼
model_comparison_gpu.csv

text

---

## Project Structure
.
├── sentiment.py # main training + comparison script
├── notebooks/
│ └── exploration.ipynb # EDA, token-length analysis, error inspection
├── outputs/
│ ├── model_comparison_gpu.csv # side-by-side metrics table
│ ├── bert_confusion_matrix.png
│ ├── lstm_confusion_matrix.png
│ └── latency_breakdown.png
├── requirements.txt
└── README.md

text

---

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
requirements.txt

txt
torch>=2.0
transformers>=4.40
datasets>=2.16
tensorflow>=2.15
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
accelerate>=0.24
GPU strongly recommended. On a T4 (Colab / Kaggle free tier), the full script runs in ~18–22 minutes. On CPU-only hardware, drop TRAIN_SIZE to 4000 and BERT_EPOCHS to 1, or switch to distilbert-base-uncased.

Usage
Basic run
bash
python sentiment.py --train-size 20000 --test-size 2000 --model bert-base-uncased
CPU-friendly run (no GPU required, ~25–35 min)
bash
python sentiment.py --train-size 4000 --epochs 1 --model distilbert-base-uncased
Programmatic
python
from sentiment import run_comparison

results = run_comparison(
    train_size=20_000,
    test_size=2_000,
    model_name="bert-base-uncased",
    epochs=2,
    fp16=True,
)
print(results)
The script will:

Load Amazon Polarity from the Hugging Face Hub.

Fine-tune BERT and print per-epoch validation metrics.

Train the BiLSTM baseline and print per-epoch metrics.

Print a side-by-side comparison table (accuracy, F1, training time).

Benchmark per-sample inference latency on 500 test examples.

Save model_comparison_gpu.csv.

Outputs
model_comparison_gpu.csv
Model	Accuracy	Precision	Recall	F1	Train_min
BERT-base (fp16)	0.9389	0.9371	0.9410	0.9390	13.4
BiLSTM (GPU)	0.8821	0.8790	0.8863	0.8826	4.6
Inference speed (T4, batch size 1)
Model	Per-sample latency	Relative speed
BERT-base	~45 ms	1×
BiLSTM	~8 ms	~5.5× faster
Interpretation
Accuracy gap: ~5–6 points. The BERT advantage is entirely attributable to pre-training — the BiLSTM sees the same 20 000 labelled examples.

Training-time gap: BERT takes ~3× longer per run, but needs only 2 epochs vs the LSTM's 5. Fine-tuning reuses most of what the model already knows.

Inference gap: BERT is ~5× slower per prediction. This is the deployment-relevant number, not the training-time number.

Model size: BERT-base ≈ 440 MB (weights only); BiLSTM ≈ 8 MB. A 55× difference that matters on edge devices.

Design decisions worth noting
fp16 training is enabled by default on GPU. On a T4 this yields ~1.8× wall-clock speedup with negligible accuracy cost. If loss reports nan at step 1 (occasional with bert-base + fp16 + LR=2e-5), set FP16 = False in config — 2× slower but rock-solid.

Truncation at 256 tokens. Amazon reviews can run 500+ words. Truncating from the right preserves the review's opening — which usually carries the strongest sentiment signal — but long-tailed reviews may lose concluding context. MAX_LEN = 384 is a reasonable middle ground if GPU memory allows.

Frequency-vs-Monetary analog for sentiment: The BiLSTM uses trainable embeddings rather than frozen GloVe by default. This makes the baseline a true from-scratch model, isolating the contribution of pre-training alone. Swap to frozen GloVe (see USE_GLOVE flag) if the goal is a stronger baseline rather than a cleaner ablation.

Version-compatibility layer. The script auto-detects transformers >= 4.46 and uses processing_class instead of the deprecated tokenizer argument to Trainer, and eval_strategy instead of evaluation_strategy on TrainingArguments. The same script runs unchanged on transformers 4.30 through the current release.

torch.cuda.synchronize() before inference timing. GPU kernels are asynchronous; without explicit synchronisation, the wall-clock measurement returns before the work actually completes. This single line changes the reported BERT latency from "3 ms" (wrong) to "45 ms" (correct).

No fine-tuned-checkpoint persistence. The script deliberately avoids saving BERT checkpoints (save_strategy="no") so a single run does not consume multiple GB of Colab disk. The final metrics are saved to CSV; the trained model is discarded. Add save_strategy="epoch" if the model itself is the deliverable.

What this comparison reveals about deployment
The script's comparison table is useful for the report, but the real answer to "should this ship to production?" is nuanced:

Scenario	Recommendation
Offline batch scoring (nightly review triage)	BERT — accuracy matters, latency doesn't
High-throughput API (> 100 req/s)	DistilBERT + INT8 — ~40 % smaller, ~2× faster, accuracy within 1 pt
Edge / mobile	BiLSTM or a distilled mini-transformer — 8 MB weights fit in memory; BERT's 440 MB does not
Low-latency single-request	BiLSTM — 5× faster per call, and at 88 % accuracy, often "good enough"
No GPU available at inference	DistilBERT or BiLSTM — bert-base on CPU runs ~150 ms/sample
The comparison is not "BERT wins" — it is "the correct model depends on where and how it runs," and this repo produces the numbers needed to make that decision per deployment.

Future work
Three-class extension. Amazon Polarity excludes 3-star reviews. Bucketing them in as neutral — either from an alternative dataset such as IberaSoft/ecommerce-reviews-sentiment, or by engineering a synthetic neutral class — turns this into a multi-class problem and changes the metrics story considerably (macro-F1 becomes the right target, not accuracy).

DistilBERT / ALBERT / MiniLM sweep. Adding a third and fourth transformer to the comparison quantifies the accuracy-vs-size frontier rather than picking a single point on it.

Quantisation benchmark. INT8 post-training quantisation via optimum gives a per-model "size / latency / accuracy" triple that reflects what actually ships.

Cross-dataset generalisation. Fine-tune on Amazon, evaluate on Yelp or IMDb without further training. Off-domain accuracy is where the BERT-vs-LSTM gap usually widens dramatically.

Error analysis on the tail. Sort misclassified examples by abs(pred_prob - 0.5) and by review length. The pattern that emerges — usually sarcasm, negation, and long-tailed multi-issue reviews — is far more actionable than the aggregate F1.

Acknowledgements
The comparison setup follows the long-standing convention in NLP benchmarking of contrasting a pre-trained transformer against a from-scratch recurrent baseline on the same data. Amazon Polarity is distributed via the Hugging Face Datasets library. Fine-tuning leverages Hugging Face Transformers and the Trainer API; the recurrent baseline uses TensorFlow / Keras. Evaluation uses scikit-learn. Visualisations use matplotlib and seaborn.

License
MIT — see LICENSE for details.
