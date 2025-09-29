#!/usr/bin/env python3
"""
Fake News Detector - Prediction Script
Loads trained pipeline and predicts label for input text.
"""

import sys
import joblib

# Load trained pipeline
model = joblib.load("models/fake_news_pipeline.joblib")

# Take input text from command line
if len(sys.argv) < 2:
    print("Usage: python predict.py \"Your news article text here...\"")
    sys.exit(1)

text = " ".join(sys.argv[1:])
pred = model.predict([text])[0]

# If probabilities available
if hasattr(model, "predict_proba"):
    proba = model.predict_proba([text])[0][1]
    print(f"Prediction: {'FAKE' if pred == 1 else 'REAL'} (probability of fake: {proba:.4f})")
else:
    print(f"Prediction: {'FAKE' if pred == 1 else 'REAL'}")
