"""
evaluate.py - measures how accurate the classifier is.

Run with:   python evaluate.py          (accurate model)
            python evaluate.py fast     (fast model)
Needs:      eval_data.csv with columns: sentence,true_label
"""

import sys
import time

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from analyzer import MODELS, load_classifier, predict_labels

fast = len(sys.argv) > 1 and sys.argv[1].lower() == "fast"
model_label = list(MODELS.keys())[1 if fast else 0]

df = pd.read_csv("eval_data.csv")
print(f"Loaded {len(df)} labeled sentences.\nLoading model: {model_label}")

classifier = load_classifier(MODELS[model_label])
start = time.time()
df["pred_label"] = predict_labels(classifier, df["sentence"].tolist())
elapsed = time.time() - start
print(f"Classified {len(df)} sentences in {elapsed:.1f} s "
      f"({elapsed / len(df):.2f} s per sentence)")

# ---- 1. Per-category report ----
print("\n=== Per-category results ===")
print(classification_report(df["true_label"], df["pred_label"], zero_division=0))

labels = sorted(df["true_label"].unique())
print("=== Confusion matrix (rows = true, columns = predicted) ===")
print(pd.DataFrame(
    confusion_matrix(df["true_label"], df["pred_label"], labels=labels),
    index=labels, columns=labels,
))

# ---- 2. Risky vs Safe (the most important number for this tool) ----
y_true = df["true_label"] != "Safe"
y_pred = df["pred_label"] != "Safe"
print("\n=== Risky vs Safe ===")
print(f"Precision: {precision_score(y_true, y_pred):.2f}")
print(f"Recall:    {recall_score(y_true, y_pred):.2f}")
print(f"F1 score:  {f1_score(y_true, y_pred):.2f}")

df.to_csv("eval_results.csv", index=False)
print("\nSaved predictions to eval_results.csv (check the wrong ones!)")
