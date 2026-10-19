"""
app.py - the Streamlit web interface.

Run with:   streamlit run app.py
"""

import html
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from analyzer import THRESHOLD, analyze, load_classifier

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="Policy Risk Analyzer",
    page_icon=str(BASE_DIR / "favicon.png"),
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling: hide Streamlit's menu, footer and "Made with Streamlit" tag,
# and define the risk badges and highlight colours.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 3px;
        font-weight: 600;
        color: #ffffff;
    }
    .badge-Low { background: #2e7d32; }
    .badge-Medium { background: #b45309; }
    .badge-High { background: #b91c1c; }
    .risky {
        background: #fde2e2;
        color: #1a1a1a;
        padding: 1px 3px;
        border-radius: 2px;
    }
    .risky-tag {
        font-size: 0.8em;
        font-weight: 600;
        color: #b91c1c;
    }
    .policy-text { line-height: 1.9; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load the model once and reuse it (Streamlit reruns the script on every click)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_classifier():
    return load_classifier()


# ---------------------------------------------------------------------------
# Read an uploaded PDF or TXT file
# ---------------------------------------------------------------------------
def read_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.title("Terms and Privacy Policy Risk Analyzer")
st.write(
    "Upload or paste a Terms of Service or Privacy Policy. The tool checks every "
    "sentence for five risk types: data sharing, auto-renewal, arbitration, "
    "account termination and data retention."
)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Confidence threshold", 0.50, 0.95, THRESHOLD, 0.05,
        help="Higher: fewer but more certain flags. Lower: catches more, with more false alarms.",
    )
    st.caption("Model: facebook/bart-large-mnli (zero-shot classification)")

uploaded = st.file_uploader("Upload a policy (PDF or TXT)", type=["pdf", "txt"])
pasted = st.text_area("Or paste the policy text", height=220)

text = read_file(uploaded) if uploaded else pasted

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Upload a file or paste some text first.")
        st.stop()

    with st.spinner("Analyzing. The first run downloads the model (about 1.6 GB)."):
        result = analyze(text, get_classifier(), threshold)

    if result["num_sentences"] == 0:
        st.error("No readable sentences found. If this is a scanned PDF, it needs OCR.")
        st.stop()

    # ---- Overview (all values are computed from this document) ----
    level = result["level"]
    st.markdown(
        f"Risk level: <span class='badge badge-{level}'>{level}</span>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk score", f"{result['score']} / {result['max_score']}")
    c2.metric("Sentences analyzed", result["num_sentences"])
    c3.metric("Risky sentences", result["num_risky"])

    # ---- Summary ----
    st.subheader("Summary")
    if not result["summary"]:
        st.success("No risky clauses found above the threshold.")
    for item in result["summary"]:
        with st.expander(f"{item['category']}: {item['explanation']}", expanded=True):
            st.write(f"Evidence (confidence {item['confidence']:.2f}):")
            st.info(item["evidence"])

    # ---- Full text with risky sentences highlighted ----
    st.subheader("Full text with risky sentences highlighted")
    parts = []
    for item in result["classified"]:
        sentence = html.escape(item["sentence"])
        if item["categories"]:
            tags = ", ".join(cat for cat, _ in item["categories"])
            parts.append(
                f"<span class='risky'>{sentence}</span> "
                f"<span class='risky-tag'>[{html.escape(tags)}]</span>"
            )
        else:
            parts.append(sentence)
    st.markdown(
        "<div class='policy-text'>" + " ".join(parts) + "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "This is an automated first check, not legal advice. "
        "Read the flagged clauses yourself before agreeing to them."
    )
