# Fake News Detector

>An explainable machine learning web application that classifies news articles as **Real** or **Fake** using Natural Language Processing.

**Live Demo:** https://fake-news-detector-ukcuk6mvhyf9m94kgcwccg.streamlit.app

---

## Overview

The rapid spread of misinformation has made it increasingly difficult to distinguish reliable news from misleading content. This project explores how classical Natural Language Processing (NLP) techniques can be used to classify news articles as **Real** or **Fake**.

Instead of building a black-box model, I wanted the prediction process to be transparent. Along with the final prediction, the application displays a confidence score and highlights the words that influenced the model's decision, making it easier to understand why a particular classification was made.

Although transformer-based models often achieve higher accuracy, I deliberately chose **TF-IDF** and **Logistic Regression** because they are lightweight, fast, interpretable, and demonstrate the fundamentals of text classification.

---

## Features

- Predicts whether a news article is **Real** or **Fake**
- Displays prediction confidence
- Shows the most influential words behind each prediction
- Interactive web interface built with Streamlit
- Easy to retrain using a different dataset
- Lightweight and fast inference

---

## Demo

The application allows users to paste a news headline or article into a text box. After clicking **Analyze**, it returns:

- The predicted class (Real or Fake)
- Prediction confidence
- The words that contributed most to the prediction

---

## Technologies Used

- Python
- Pandas
- NumPy
- scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- Joblib

---

## How It Works

### Text Preprocessing

Before training, the text is cleaned and prepared for feature extraction.

### TF-IDF Vectorization

The cleaned text is transformed into numerical vectors using TF-IDF (Term Frequency–Inverse Document Frequency).

Words that appear frequently in a document but are relatively uncommon across the dataset receive higher weights, allowing the model to focus on informative terms rather than common words.

The vectorizer also considers bigrams (two-word phrases) to capture more meaningful language patterns.

### Logistic Regression

The vectorized text is then passed to a Logistic Regression classifier.

During training, the model learns which words are more strongly associated with fake news and which are associated with real news.

At prediction time, the model calculates the probability for each class and the application displays both the confidence score and the most influential words contributing to that prediction.

---

## Project Structure

```text
fake-news-detector/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── news.csv
│
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
│
└── screenshots/
```

---

## Running the Project

Clone the repository.

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

Train the model.

```bash
python train.py
```

Run the Streamlit application.

```bash
streamlit run app.py
```

---

## Dataset

This project was initially developed using a small synthetic dataset for testing the application.

For a more realistic model, I later trained it using the **Fake and Real News Dataset** available on Kaggle, which contains over 44,000 labeled news articles.

Using a real dataset significantly improves the model's ability to generalize compared to template-generated examples.

---

## Limitations

This project is intended as an educational exploration of NLP rather than a production-ready misinformation detector.

Since the dataset mainly contains Reuters articles for the REAL class, the model sometimes learns writing style alongside factual characteristics.

As a result, it may perform less reliably on:

- Short headlines with limited context
- Recently published news
- Informal writing styles
- News sources that differ significantly from the training dataset

These limitations highlight the importance of dataset quality and diversity in machine learning.

---

## Future Improvements

Some improvements I would like to explore include:

- Comparing additional machine learning models
- Fine-tuning transformer models such as DistilBERT
- URL-based article analysis
- Source credibility verification
- Docker support
- Automated testing
- CI/CD using GitHub Actions

---

## What I Learned

This project helped me strengthen my understanding of:

- Natural Language Processing
- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression
- Machine learning workflows
- Model serialization with Joblib
- Building interactive applications with Streamlit
- Model interpretability and explainable AI

One of the most valuable lessons from this project was realizing that a model's performance depends just as much on the quality of its training data as on the algorithm itself.

---

## Screenshots

Screenshots of the application will be added here.

---

## Author

**Mrittika Jahan**

This project was built as part of my machine learning portfolio to explore explainable NLP models and develop practical experience with end-to-end ML application development.
