"""
evaluate.py - measures how accurate the classifier is.

Run with:   python evaluate.py
Needs:      eval_data.csv with columns: sentence,true_label
"""

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from analyzer import load_classifier, predict_labels

df = pd.read_csv("eval_data.csv")
print(f"Loaded {len(df)} labeled sentences.\nLoading model...")

classifier = load_classifier()
df["pred_label"] = predict_labels(classifier, df["sentence"].tolist())

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
