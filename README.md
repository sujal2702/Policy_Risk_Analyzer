# 🛡️ Terms & Privacy Policy Risk Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/Backend-PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent Natural Language Processing (NLP) tool that analyzes Terms of Service (ToS) and Privacy Policies, flags potentially hazardous clauses, and delivers plain-English risk assessments with an overall risk rating.

Most users click **"I Agree"** without reading dozens of pages of dense legal jargon. This analyzer parses every sentence in the document, isolates critical liabilities, and presents an evidence-backed summary so users can review the clauses that actually matter in seconds.

---

## 🌟 Key Features

- 📄 **Multi-Format Ingestion**: Supports `.pdf` documents (via `pypdf`), `.txt` policy files, or direct raw text pasting.
- ⚡ **Dual-Model Zero-Shot Inference**:
  - **Accurate Model**: `facebook/bart-large-mnli` (406M parameters) for maximum semantic precision and nuance.
  - **Fast Model**: `valhalla/distilbart-mnli-12-3` (~2-3x speedup) for rapid checks on lower-spec hardware.
- 🚀 **Decoupled Two-Stage Scoring (Instant Sliders)**:
  - Heavy model inference runs **only once** per document; raw category probabilities are cached in session memory.
  - Modifying the confidence threshold slider re-filters clauses **instantly (<10ms)** without triggering re-inference.
- 📊 **Real-Time Mini-Batch Progress**: Displays a dynamic Streamlit progress bar processing sentences in optimized mini-batches (`BATCH_SIZE = 8`).
- ⚖️ **Weighted Risk Index**: Transparent scoring algorithm (0 to 11) categorizing policies into **Low**, **Medium**, or **High** risk levels.
- 🔍 **Extractive Summarization**: Identifies the single most confident sentence per flagged risk category and pairs it with a plain-English explanation.
- 🖍️ **In-Situ Clause Highlighting**: Renders the complete policy text with highlighted warning blocks and categorical metadata tags.
- 🧪 **Comprehensive Evaluation Suite**: Includes `evaluate.py` to benchmark per-class precision, recall, F1 scores, latency, and confusion matrices against `eval_data.csv`.

---

## ⚖️ Risk Categories & Scoring Model

Each detected risk category carries a weight reflecting its potential legal and privacy impact on the user:

| Category | What it Means for the User | Weight | Default Threshold |
|---|---|:---:|:---:|
| **Data Sharing** | Personal data may be sold, monetized, or shared with third-party brokers and marketing partners. | **3** | `0.80` |
| **Arbitration** | You waive your constitutional right to court litigation, trial by jury, or participating in class-action lawsuits. | **3** | `0.80` |
| **Account Termination** | The service reserves the unilateral right to suspend, terminate, or delete your account without prior notice. | **2** | `0.80` |
| **Auto-Renewal** | Subscriptions automatically renew and bill your payment method unless canceled under specific conditions. | **2** | `0.80` |
| **Data Retention** | Your personal identifiers, activity logs, or telemetry are preserved even after you close or delete your account. | **1** | `0.80` |

### 🧮 Risk Score Formulation

$$\text{Risk Score} = \sum_{\text{category} \in \text{Found}} \text{Weight}(\text{category}) \quad (\text{Maximum: } 11)$$

- **Low Risk (0 – 3)**: Standard operational agreements with standard clauses and minimal liability shifts.
- **Medium Risk (4 – 7)**: Contains notable commitments such as mandatory auto-renewals or unilateral suspension rights.
- **High Risk (8 – 11)**: Aggressive terms involving forced binding arbitration, third-party data sharing, and long retention windows.

---

## 🏗️ System Architecture & Workflow

```text
               +-------------------------------------------+
               |   Policy Document (PDF / TXT / Raw Text)  |
               +-------------------------------------------+
                                     |
                                     v
               +-------------------------------------------+
               |    Preprocessing & Sentence Segmentation  |
               |       (Header normalizer + NLTK Punkt)    |
               +-------------------------------------------+
                                     |
                                     v
               +-------------------------------------------+
               |      Batch Inference (Mini-batch size: 8) |
               |    BART-Large-MNLI / DistilBART-MNLI      |
               +-------------------------------------------+
                                     |
                                     v
               +-------------------------------------------+
               |     Session Score Cache [Key: (Model, Text)]|
               +-------------------------------------------+
                                     |
                     +---------------+---------------+
                     |                               |
                     v                               v
    +---------------------------------+  +---------------------------------+
    |   Instant Threshold Filtering   |  |   Extractive Summarization      |
    |      (Confidence >= Threshold)  |  |    (Argmax confidence clause)   |
    +---------------------------------+  +---------------------------------+
                     |                               |
                     +---------------+---------------+
                                     |
                                     v
               +-------------------------------------------+
               |       Streamlit Interactive Interface     |
               |  - Risk Badge & Metric Counters           |
               |  - Plain-English Expandable Cards         |
               |  - Full Text with In-Situ Highlights      |
               +-------------------------------------------+
```

### Why Zero-Shot Natural Language Inference (NLI)?

Instead of relying on rigid keyword search or training expensive domain-specific classifiers with scarce training data, this project frames risk detection as **Zero-Shot NLI**:
- **Premise**: An isolated policy sentence (e.g., *"We may disclose your personal identifiers to our corporate affiliates and advertising partners."*).
- **Hypothesis**: `"This clause is about sharing or selling user data to third parties."`
- The model outputs an entailment probability. If the entailment probability exceeds the threshold, the risk is flagged. This enables modular additions of new categories simply by introducing new hypothesis definitions.

---

## 📁 Repository Structure

```
Policy_Risk_Analyzer/
├── .streamlit/
│   └── config.toml         # Custom Streamlit dark/light styling and telemetry options
├── .gitignore              # Ignores __pycache__, virtual environments, and eval results
├── analyzer.py             # Core NLP engine: tokenization, batched NLI, caching, and risk scoring
├── app.py                  # Streamlit application: UI, file uploaders, progress bar, highlighters
├── evaluate.py             # Evaluation harness: latency profiling, classification report, confusion matrix
├── eval_data.csv           # Ground-truth benchmark dataset with labeled test sentences
├── sample_policy.txt       # Ready-to-use sample policy containing multi-category risk clauses
├── requirements.txt        # Pinned Python package dependencies
└── README.md               # Complete project documentation
```

---

## 🚀 Installation & Quickstart

### Prerequisites
- **Python 3.10+**
- Recommended: 8 GB+ RAM (to run BART-large in memory)

### 1. Clone the Repository
```bash
git clone https://github.com/sujal2702/Policy_Risk_Analyzer.git
cd Policy_Risk_Analyzer
```

### 2. Set Up a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Application
```bash
streamlit run app.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

---

## 💻 How to Use

1. **Select Model** in the sidebar:
   - **Accurate (`bart-large-mnli`)**: Best for production and critical audits.
   - **Fast (`distilbart-mnli-12-3`)**: Best for fast checks or limited CPU resources.
2. **Provide Input**:
   - Upload a `.pdf` file (e.g., download a company's privacy policy PDF).
   - Upload a `.txt` document.
   - Or paste text directly into the text box (try copying lines from [sample_policy.txt](sample_policy.txt)).
3. **Click "Analyze"**:
   - The first run downloads the chosen model weights (stored locally in `~/.cache/huggingface/hub/`).
   - The progress bar displays the sentence batching progress in real time.
4. **Tune Sensitivity**:
   - Adjust the **Confidence threshold** slider (0.50 – 0.95). The UI updates instantaneously without re-running the model.
5. **Inspect the Results**:
   - Review the aggregate **Risk Level Badge** and score metrics.
   - Expand the **Summary Cards** for plain-English explanations and extracted evidence.
   - Read the **Highlighted Document** to inspect clauses in their original context.

---

## 🧪 Evaluation & Benchmarks

The project includes an automated evaluation harness in [evaluate.py](evaluate.py) tested against [eval_data.csv](eval_data.csv).

### Running Evaluations

**Evaluate Accurate Model:**
```bash
python evaluate.py
```

**Evaluate Fast Model:**
```bash
python evaluate.py fast
```

### Metrics Produced
- **Per-Sentence Latency**: Average time required per sentence inference.
- **Classification Report**: Precision, Recall, and F1-Score for each individual category.
- **Confusion Matrix**: Full breakdown of true versus predicted labels.
- **Risky vs Safe Binary Assessment**: Evaluates the system's ability to discriminate between risky legal clauses and benign policy boilerplate.

> **Note on Evaluation Philosophy**: In legal risk analysis, **Recall** is prioritized over Precision. Catching a questionable clause with a slight false-positive warning is far safer for the consumer than missing an aggressive arbitration or data-sharing waiver.

---

## ⚙️ Configuration Reference

All core hyperparameters can be adjusted at the top of [analyzer.py](analyzer.py):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MODELS` | `dict` | `{"Accurate...": ..., "Fast...": ...}` | Available Hugging Face zero-shot pipelines. |
| `BATCH_SIZE` | `int` | `8` | Number of sentences processed per forward pass. |
| `THRESHOLD` | `float` | `0.80` | Default minimum entailment probability to flag a risk. |
| `HYPOTHESIS` | `str` | `"This clause is about {}."` | Template used for zero-shot natural language inference. |
| `LABELS` | `dict` | 5 categories | Mapping from model-facing hypothesis strings to user-facing category labels. |
| `WEIGHTS` | `dict` | 1 to 3 | Severity weights for each category used in aggregate risk index calculation. |

---

## ⚠️ Limitations & Roadmap

- **Scanned Documents**: Scanned PDFs composed solely of raster images require an upstream OCR pipeline (e.g., `tesseract` / `easyocr`).
- **CPU Constraints**: On CPU-only environments, very large documents (500+ sentences) may take a few minutes for the initial inference run.
- **Ambiguous Phrasing**: Highly stylized legalese or multi-sentence conditional qualifications may require additional context modeling.
- **Roadmap**:
  - [ ] Add GPU auto-detection (`device=0` in PyTorch).
  - [ ] Support OCR for scanned policy documents.
  - [ ] Export executive audit summaries to PDF / JSON.
  - [ ] Multi-language support for international policies.

---

## 📖 Academic & Industry References

- **OPP-115 Corpus**: Wilson, S. et al. (2016). *The Creation and Analysis of a Large-Scale Corpus of Privacy Policies*. Association for Computational Linguistics.
- **CLAUDETTE System**: Lippi, M. et al. (2019). *CLAUDETTE: An Automated Detector of Potentially Unfair Clauses in Online Terms of Service*. Artificial Intelligence and Law.
- **BART Architecture**: Lewis, M. et al. (2019). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*.

---

## ⚖️ Legal Disclaimer

*This application provides automated screening based on statistical Natural Language Processing heuristics and does not constitute formal legal counsel. Always review flagged terms directly and consult a qualified legal professional for binding legal advice.*
