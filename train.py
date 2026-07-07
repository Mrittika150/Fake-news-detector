"""
train.py
--------
Loads dataset/news.csv, trains a TF-IDF + Logistic Regression classifier
to detect fake vs real news, evaluates it, and saves:
  - model/model.pkl        (trained LogisticRegression classifier)
  - model/vectorizer.pkl   (fitted TfidfVectorizer)

Usage:
    python train.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "dataset/news.csv"
MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df = df[df["label"].isin(["FAKE", "REAL"])]
    return df


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows -> {df['label'].value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2),
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)

    print("Evaluating...")
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
