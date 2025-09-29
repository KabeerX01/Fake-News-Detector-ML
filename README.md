# Fake News Detector 🤖📰

A Machine Learning project to detect fake vs real news using **TF–IDF + Logistic Regression**.
## 🚀 Features

- 🧹 Cleans and processes news text  
- 🔍 Extracts TF–IDF features (unigrams + bigrams)  
- 🤖 Trains Logistic Regression model  
- 📊 Evaluates Accuracy, Precision, Recall, F1, ROC AUC  
- 💾 Saves model using `joblib`  
- 🧠 Predicts fake/real from new text input
## 🧱 Project Structure
```text
Fake-News-Detector/
├── data/             # Contains sample_news.csv only
├── images/           # Contains confusion_matrix.png
├── scripts/          # prepare_data.py (merges datasets)
├── models/           # model.joblib (NOT uploaded)
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```
## 📦 Dataset Download

> ⚠️ GitHub limits uploads to 25MB via web. The full datasets are too large to include here.

Please manually download the datasets and place them inside the `data/` folder:

- [Fake.csv](https://drive.google.com/file/d/17G4BFxxSyaDeClSrYEGxu1_stYNCwCfS/view?usp=drive_link)  
- [True.csv](https://drive.google.com/file/d/1__kluUs62qeiy69OFex5QnXyro24LtoS/view?usp=drive_link)  
- [news_dataset.csv](https://drive.google.com/file/d/1of7lMvgxM5oHn5TvTr9DYrTsp_ydLjw2/view?usp=drive_link)  

Or generate `news_dataset.csv` by running:

```bash
python scripts/prepare_data.py
```
## 📸 Confusion Matrix

This is the confusion matrix generated after evaluating the model:

![Confusion Matrix](images/confusion_matrix.png)

## 🙌 Acknowledgments

- Built with scikit-learn, pandas, and matplotlib
- Inspired by real-world misinformation problems
- Datasets provided by Kaggle / online news sources
## 🧑‍💻 Author

Made with ❤️ by Kabeer Shaikh.

Feel free to contribute or give feedback!
