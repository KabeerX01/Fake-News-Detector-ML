#!/usr/bin/env python3
"""
Fake News Detector - Training Script
Baseline: TF-IDF + Logistic Regression
"""

import argparse
import joblib
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)         # remove URLs
    text = re.sub(r"<.*?>", " ", text)           # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)        # keep only alphabets
    return " ".join(text.split())

def main(args):
    # Load dataset
    df = pd.read_csv(args.data)

    # Combine title + text if both available
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"]
    else:
        raise ValueError("Dataset must have at least 'text' column")

    # Ensure label column exists
    if args.label_col not in df.columns:
        raise ValueError(f"Dataset must contain a '{args.label_col}' column")

    # Clean text
    df["content"] = df["content"].apply(clean_text)
    X = df["content"]
    y = df[args.label_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train model
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
    except Exception:
        pass

    # Save model
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"✅ Model saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--out", default="models/fake_news_pipeline.joblib", help="Output model path")
    parser.add_argument("--label-col", default="label", help="Name of label column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--max-features", type=int, default=10000, help="Maximum TF-IDF features")
    args = parser.parse_args()
    main(args)
