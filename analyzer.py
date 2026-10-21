"""
analyzer.py - the NLP "brain" of the project.

Pipeline:
    raw text -> split into sentences -> zero-shot classify each sentence
             -> risk score -> extractive summary
"""

import re

import nltk
from transformers import pipeline

# ---------------------------------------------------------------------------
# 1. Download the NLTK sentence tokenizer data (only the first time)
# ---------------------------------------------------------------------------
for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.tokenize import sent_tokenize  # noqa: E402

# ---------------------------------------------------------------------------
# 2. Settings
# ---------------------------------------------------------------------------
# Pretrained NLI models (no training needed). "Fast" is smaller, about 2-3x faster.
MODELS = {
    "Accurate (bart-large-mnli)": "facebook/bart-large-mnli",
    "Fast (distilbart-mnli-12-3)": "valhalla/distilbart-mnli-12-3",
}
MODEL_NAME = MODELS["Accurate (bart-large-mnli)"]
THRESHOLD = 0.80                           # min score to flag a sentence as risky
HYPOTHESIS = "This clause is about {}."    # sentence the model checks against
BATCH_SIZE = 8                             # sentences sent to the model at once

# Descriptive label (what the model sees) -> short category name (what the user sees)
LABELS = {
    "sharing or selling user data to third parties": "Data Sharing",
    "automatic renewal of subscription or payment": "Auto-Renewal",
    "arbitration or waiving the right to sue": "Arbitration",
    "terminating or suspending user accounts": "Account Termination",
    "storing or retaining user data": "Data Retention",
}

# How serious each risk is
WEIGHTS = {
    "Data Sharing": 3,
    "Arbitration": 3,
    "Account Termination": 2,
    "Auto-Renewal": 2,
    "Data Retention": 1,
}
MAX_SCORE = sum(WEIGHTS.values())  # 11

# Plain-English explanation for each risk
EXPLAIN = {
    "Data Sharing": "Your data may be shared with or sold to other companies.",
    "Auto-Renewal": "Your subscription may renew and charge you automatically.",
    "Arbitration": "You may lose the right to sue or join a class action.",
    "Account Termination": "Your account can be closed or suspended, possibly without notice.",
    "Data Retention": "Your data may be kept even after you stop using the service.",
}


# ---------------------------------------------------------------------------
# 3. Load the model
# ---------------------------------------------------------------------------
def load_classifier(model_name=MODEL_NAME):
    """Download (first time) and load the zero-shot classifier."""
    return pipeline("zero-shot-classification", model=model_name)


# ---------------------------------------------------------------------------
# 4. Sentence segmentation
# ---------------------------------------------------------------------------
def split_sentences(text, min_words=6):
    """Clean the text and split it into sentences, dropping headings/fragments."""
    # Short lines with no ending punctuation are headings ("4. Sharing Your Data").
    # End them with a full stop so they don't get glued to the next sentence.
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line and len(line.split()) <= 8 and not re.search(r"[.!?:;,]$", line):
            line += "."
        lines.append(line)
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()  # PDFs break lines mid-sentence
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        # NLTK data could not be downloaded (no internet / proxy):
        # fall back to splitting after . ! or ? followed by a space
        sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.split()) >= min_words]


# ---------------------------------------------------------------------------
# 5. Zero-shot classification
# ---------------------------------------------------------------------------
def score_sentences(classifier, sentences, batch_size=BATCH_SIZE, progress=None):
    """
    Runs the model ONCE and keeps every label's score, so the threshold can be
    changed later without running the model again.

    Returns one dict per sentence:
        {"sentence": str, "scores": {category: score, ...}}
    'progress' is an optional function called as progress(done, total).
    """
    scored = []
    total = len(sentences)
    for start in range(0, total, batch_size):
        chunk = sentences[start:start + batch_size]
        results = classifier(
            chunk,
            candidate_labels=list(LABELS.keys()),
            hypothesis_template=HYPOTHESIS,
            multi_label=True,               # score each label independently
            batch_size=batch_size,          # process several pairs at once
        )
        if isinstance(results, dict):       # a single sentence returns a dict
            results = [results]
        for sentence, res in zip(chunk, results):
            scores = {
                LABELS[label]: round(score, 3)
                for label, score in zip(res["labels"], res["scores"])
            }
            scored.append({"sentence": sentence, "scores": scores})
        if progress:
            progress(min(start + batch_size, total), total)
    return scored


def apply_threshold(scored, threshold=THRESHOLD):
    """
    Turns raw scores into flags (no model needed, so this is instant).
    Returns one dict per sentence:
        {"sentence": str, "categories": [(category, score), ...]}
    'categories' is sorted by score and is empty when the sentence is safe.
    """
    output = []
    for item in scored:
        flagged = sorted(
            [(cat, s) for cat, s in item["scores"].items() if s >= threshold],
            key=lambda pair: pair[1],
            reverse=True,
        )
        output.append({"sentence": item["sentence"], "categories": flagged})
    return output


def classify_sentences(classifier, sentences, threshold=THRESHOLD):
    """Score and flag in one step."""
    return apply_threshold(score_sentences(classifier, sentences), threshold)


# ---------------------------------------------------------------------------
# 6. Risk score
# ---------------------------------------------------------------------------
def risk_score(classified):
    """Weighted count of the distinct risk categories found."""
    found = {cat for item in classified for cat, _ in item["categories"]}
    score = sum(WEIGHTS[cat] for cat in found)

    if score <= 3:
        level = "Low"
    elif score <= 7:
        level = "Medium"
    else:
        level = "High"
    return score, level, found


# ---------------------------------------------------------------------------
# 7. Extractive summary
# ---------------------------------------------------------------------------
def summarize(classified):
    """For each category, pick the single most confident sentence as evidence."""
    best = {}
    for item in classified:
        for cat, score in item["categories"]:
            if cat not in best or score > best[cat][1]:
                best[cat] = (item["sentence"], score)

    # order by seriousness (highest weight first)
    ordered = sorted(best, key=lambda c: WEIGHTS[c], reverse=True)
    return [
        {
            "category": cat,
            "explanation": EXPLAIN[cat],
            "evidence": best[cat][0],
            "confidence": best[cat][1],
        }
        for cat in ordered
    ]


# ---------------------------------------------------------------------------
# 8. Full analysis (used by app.py)
# ---------------------------------------------------------------------------
def build_result(scored, threshold=THRESHOLD):
    """Everything the app displays, built from saved scores (instant)."""
    classified = apply_threshold(scored, threshold)
    score, level, found = risk_score(classified)
    return {
        "classified": classified,
        "score": score,
        "max_score": MAX_SCORE,
        "level": level,
        "summary": summarize(classified),
        "num_sentences": len(scored),
        "num_risky": sum(1 for c in classified if c["categories"]),
    }


def analyze(text, classifier, threshold=THRESHOLD):
    """Split, score and build the result in one call."""
    return build_result(score_sentences(classifier, split_sentences(text)), threshold)


# ---------------------------------------------------------------------------
# 9. Single-label prediction (used by evaluate.py)
# ---------------------------------------------------------------------------
def predict_labels(classifier, sentences, threshold=THRESHOLD):
    """Top category per sentence, or 'Safe' if nothing passes the threshold."""
    classified = classify_sentences(classifier, sentences, threshold)
    return [c["categories"][0][0] if c["categories"] else "Safe" for c in classified]
