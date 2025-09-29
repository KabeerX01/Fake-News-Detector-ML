# Fake News Detector 📰🤖

A Machine Learning project to detect fake vs real news articles using **TF–IDF + Logistic Regression**.

## 🚀 Features
- Preprocesses and cleans news text (title + body)
- Extracts features with TF–IDF (unigrams + bigrams)
- Trains Logistic Regression baseline model
- Evaluates with Accuracy, Precision, Recall, F1, ROC AUC
- Saves trained model with `joblib`
- Predicts fake/real news from new input text

## 📂 Project Structure
Fake-News-Detector/
│
├── images/                    # Plots and visual outputs
│   └── confusion_matrix.png
│
├── scripts/                   # Extra utility scripts
│   └── prepare_data.py        # Merges Fake.csv & True.csv (not uploaded)
│
├── models/                    # Contains trained .joblib model (ignored)
│   └── model.joblib           # <--- Not included in repo due to size
│
├── train.py                   # Trains and evaluates the model
├── predict.py                 # Predicts on new user input
├── requirements.txt           # Dependencies
├── .gitignore                 # Prevents large files from being uploaded
└── README.md

