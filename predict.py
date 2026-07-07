"""
predict.py
----------
Loads the trained model + vectorizer and exposes a predict() function that
returns the predicted label, a confidence score, and the words that most
influenced the decision (for explainability).

Can also be run directly from the command line:
    python predict.py "Some news article text here"
"""

import sys
import joblib
import numpy as np

MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

_model = None
_vectorizer = None


def _load_artifacts():
    """Load model + vectorizer once and cache them."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def predict(text: str, top_n: int = 8):
    """
    Predict whether `text` is FAKE or REAL news.

    Returns a dict:
        {
            "label": "FAKE" or "REAL",
            "confidence": float (0-1),
            "probabilities": {"FAKE": float, "REAL": float},
            "top_words": [(word, weight), ...]   # words pushing toward the predicted label
        }
    """
    model, vectorizer = _load_artifacts()

    vec = vectorizer.transform([text])
    proba = model.predict_proba(vec)[0]
    classes = model.classes_  # e.g. ['FAKE', 'REAL'] alphabetically
    label_idx = int(np.argmax(proba))
    label = classes[label_idx]
    confidence = float(proba[label_idx])

    probabilities = {cls: float(p) for cls, p in zip(classes, proba)}

    # --- Explainability: which words in THIS text most influenced the prediction ---
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]  # positive -> pushes toward classes_[1], negative -> classes_[0]

    row = vec.tocoo()
    present_idx = row.col
    tfidf_values = row.data

    # Contribution of each present word = tfidf value * logistic regression coefficient
    contributions = tfidf_values * coefs[present_idx]

    # Orient contributions so higher = "pushes toward the predicted label"
    if label == classes[0]:
        contributions = -contributions

    order = np.argsort(contributions)[::-1][:top_n]
    top_words = [
        (feature_names[present_idx[i]], float(contributions[i]))
        for i in order
        if contributions[i] > 0
    ]

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_words": top_words,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python predict.py "news article text here"')
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    result = predict(input_text)

    print(f"\nPrediction: {result['label']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"Probabilities: {result['probabilities']}")
    print("Top influential words:")
    for word, weight in result["top_words"]:
        print(f"  {word:<20} {weight:.4f}")
