# Terms and Privacy Policy Risk Analyzer

An NLP tool that reads a Terms of Service or Privacy Policy, flags risky clauses, and explains them in plain English with an overall risk level.

Most people accept these documents without reading them. This tool checks every sentence and points out the clauses that matter, so a user can read those few sentences instead of the whole policy.

## Features

- Accepts a PDF, a TXT file, or pasted text
- Checks every sentence for five risk types
- Gives an overall risk level (Low, Medium or High) and a risk score
- Shows a summary: one plain-English explanation and one evidence sentence per risk found
- Highlights risky sentences in the full text
- Adjustable confidence threshold that updates results instantly, without re-running the model
- Two models: Accurate (`bart-large-mnli`) and Fast (`distilbart-mnli-12-3`, about 2 to 3 times faster)
- Progress bar showing how many sentences have been analyzed
- Evaluation script that reports precision, recall and F1 score

## Risk Categories

| Category | What it means for the user | Weight |
|---|---|---|
| Data Sharing | Your data may be shared with or sold to other companies | 3 |
| Arbitration | You may lose the right to sue or join a class action | 3 |
| Account Termination | Your account can be closed or suspended, possibly without notice | 2 |
| Auto-Renewal | Your subscription may renew and charge you automatically | 2 |
| Data Retention | Your data may be kept after you stop using the service | 1 |

Risk score = sum of the weights of the categories found (maximum 11).
Risk level: 0 to 3 is Low, 4 to 7 is Medium, 8 to 11 is High.

## How It Works

```
Policy text (PDF / TXT / pasted)
        |
        v
1. Sentence segmentation (NLTK)
        |
        v
2. Zero-shot classification of each sentence, in batches of 8
   (facebook/bart-large-mnli or valhalla/distilbart-mnli-12-3)
        |
        v
3. Scores are saved, then labels above the threshold are kept (default 0.80)
        |
        v
4. Risk score and risk level
        |
        v
5. Extractive summary (most confident sentence per category)
        |
        v
6. Results shown in the Streamlit web app
```

### Why zero-shot classification?

`facebook/bart-large-mnli` is trained on Natural Language Inference (NLI): given a premise and a hypothesis, it predicts whether the premise supports the hypothesis.

Each sentence of the policy is used as the premise, and each risk category is turned into a hypothesis such as "This clause is about sharing or selling user data to third parties." The model's confidence that the sentence supports the hypothesis becomes the score for that category.

This means the tool needs no labeled training data and no model training. New categories can be added by writing a new label.

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Programming language |
| Hugging Face Transformers | Loads the pretrained model |
| PyTorch | Runs the model |
| NLTK | Splits text into sentences |
| Streamlit | Web interface |
| pypdf | Reads text from PDF files |
| scikit-learn | Precision, recall and F1 score |
| pandas | Handles evaluation data |

## Project Structure

```
policy-risk-analyzer/
|-- analyzer.py           NLP logic: segmentation, classification, risk score, summary
|-- app.py                Streamlit web interface
|-- evaluate.py           Measures accuracy on labeled data
|-- eval_data.csv         40 labeled sentences for evaluation
|-- sample_policy.txt     Sample policy for testing the app
|-- requirements.txt      Python libraries
|-- favicon.png           Browser tab icon
|-- .streamlit/
    |-- config.toml       Theme and interface settings
```

## Setup

1. Install Python 3.10 or later.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment:

   Windows:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   macOS / Linux:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install the libraries:
   ```
   pip install -r requirements.txt
   ```

## Running the App

```
streamlit run app.py
```

The app opens at `http://localhost:8501`.

1. Upload a PDF or TXT file, or paste the policy text.
2. Click **Analyze**.

The first analysis downloads the selected model (about 1.6 GB for Accurate, about 0.9 GB for Fast). After that, it is loaded from the local cache. Analysis runs on the CPU, so a long policy can take a minute or two.

### Speed

- The model runs once per document. Changing the threshold afterwards only re-applies it to the saved scores, so results update instantly.
- Clicking Analyze again on the same text with the same model reuses the saved scores.
- Sentences are sent to the model in batches of 8.
- Choose **Fast** in the sidebar for a smaller model that is about 2 to 3 times quicker, with slightly lower accuracy.

To try the app quickly, paste the contents of `sample_policy.txt`.

## Evaluation

```
python evaluate.py          # Accurate model
python evaluate.py fast     # Fast model
```

The script runs the classifier on `eval_data.csv` and prints:

- Time taken per sentence

- Precision, recall and F1 score for each category
- A confusion matrix
- Overall precision, recall and F1 score for Risky vs Safe

Predictions are saved to `eval_results.csv` so incorrect cases can be reviewed.

`eval_data.csv` has two columns, `sentence` and `true_label`. Valid labels are `Data Sharing`, `Auto-Renewal`, `Arbitration`, `Account Termination`, `Data Retention` and `Safe`. More rows can be added to make the evaluation stronger.

For this tool, recall matters more than precision: missing a risky clause is worse than showing an extra warning.

### Results

| Metric (Risky vs Safe) | Accurate | Fast |
|---|---|---|
| Precision | fill in | fill in |
| Recall | fill in | fill in |
| F1 score | fill in | fill in |
| Seconds per sentence | fill in | fill in |

## Configuration

These settings are at the top of `analyzer.py`:

| Setting | Default | Description |
|---|---|---|
| `MODELS` | Accurate and Fast | Models shown in the sidebar |
| `MODEL_NAME` | `facebook/bart-large-mnli` | Default model (used by `evaluate.py`) |
| `BATCH_SIZE` | `8` | Sentences sent to the model at once |
| `THRESHOLD` | `0.80` | Minimum score for a sentence to be flagged |
| `HYPOTHESIS` | `This clause is about {}.` | Template the model checks each label against |
| `LABELS` | 5 categories | Descriptive label shown to the model and the short name shown to the user |
| `WEIGHTS` | see table above | Seriousness of each category |

The threshold can also be changed from the sidebar in the app. A lower value catches more risky clauses but gives more false alarms. A higher value gives fewer, more certain flags.

## Limitations

- Scanned PDFs (images of text) are not supported, since they need OCR.
- Analysis is slow on CPU for long documents.
- Zero-shot accuracy depends on how the labels are worded.
- Legal language can be ambiguous, so some clauses may be missed or wrongly flagged.
- Only English text is supported.

## Related Work

- **OPP-115**: a corpus of 115 website privacy policies annotated with data practice categories (Wilson et al., 2016).
- **CLAUDETTE**: a system and dataset for detecting potentially unfair clauses in online Terms of Service (Lippi et al., 2019).

## Disclaimer

This tool gives an automated first check of a policy. It is not legal advice. Always read the flagged clauses yourself before agreeing to them.
