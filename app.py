"""
app.py
------
Streamlit front-end for the Fake News Detector.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from predict import predict

from PIL import Image

icon = Image.open("assets/icon.png")

st.set_page_config(
    page_title="Fake News Detector",
    page_icon=icon,
    layout="centered",
)

# ---------------- Styling ----------------
st.markdown(
    """
    <style>
    .big-font { font-size:22px !important; font-weight:600; }
    .fake-badge {
        background-color:#ffe1e1; color:#b30000; padding:8px 18px;
        border-radius:20px; font-weight:700; display:inline-block;
    }
    .real-badge {
        background-color:#e1ffe6; color:#0a7a2f; padding:8px 18px;
        border-radius:20px; font-weight:700; display:inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col_icon, col_title = st.columns([1, 8])
with col_icon:
    st.image("assets/icon.png", width=60)
with col_title:
    st.title("Fake News Detector")
st.write(
    "Paste a news headline or article below. The model (TF-IDF + Logistic "
    "Regression) will predict whether it's likely **Fake** or **Real**, "
    "along with a confidence score and the words that most influenced the decision."
)

MODEL_MISSING = not (os.path.exists("model/model.pkl") and os.path.exists("model/vectorizer.pkl"))
if MODEL_MISSING:
    st.error(
        "No trained model found. Run `python train.py` first to generate "
        "`model/model.pkl` and `model/vectorizer.pkl`."
    )
    st.stop()

user_text = st.text_area(
    "News article / headline text",
    height=180,
    placeholder="Paste a real news headline or article here...",
)

analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

if analyze:
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            result = predict(user_text)

        label = result["label"]
        confidence = result["confidence"]
        probs = result["probabilities"]
        top_words = result["top_words"]

        st.markdown("### Result")
        if label == "FAKE":
            st.markdown('<span class="fake-badge">⚠️ Likely FAKE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="real-badge">✅ Likely REAL</span>', unsafe_allow_html=True)

        st.markdown(f'<p class="big-font">Confidence: {confidence*100:.1f}%</p>', unsafe_allow_html=True)
        st.progress(confidence)

        col1, col2 = st.columns(2)
        col1.metric("P(FAKE)", f"{probs.get('FAKE', 0)*100:.1f}%")
        col2.metric("P(REAL)", f"{probs.get('REAL', 0)*100:.1f}%")

        if top_words:
            st.markdown("### 🔎 Words that influenced this prediction")
            st.caption(
                f"These words/phrases pushed the model toward predicting **{label}**."
            )
            for word, weight in top_words:
                st.write(f"- **{word}**  _(weight: {weight:.3f})_")
        else:
            st.caption("No strongly influential words were found for this text.")


