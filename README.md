# 📰 Fake News Detector

A machine learning web app that classifies news text as **Fake** or **Real** using
TF-IDF vectorization and Logistic Regression, with a Streamlit interface that shows
a confidence score and the words that most influenced the prediction.

**Live demo:** [https://fake-news-detector-ukcuk6mvhyf9m94kgcwccg.streamlit.app](https://fake-news-detector-ukcuk6mvhyf9m94kgcwccg.streamlit.app)
**Tech stack:** Python · Pandas · scikit-learn · TF-IDF · Logistic Regression · Streamlit

---

## Table of Contents
1. [What this project does](#what-this-project-does)
2. [Folder structure](#folder-structure)
3. [Setup from scratch](#setup-from-scratch)
4. [Training the model](#training-the-model)
5. [Running the app](#running-the-app)
6. [Using a real dataset (recommended for portfolio)](#using-a-real-dataset-recommended-for-portfolio)
7. [How the model works](#how-the-model-works)
8. [Deploying it publicly (free)](#deploying-it-publicly-free)
9. [Ideas to extend this project](#ideas-to-extend-this-project)
10. [Making this shine on your portfolio/resume](#making-this-shine-on-your-portfolioresume)

---

## What this project does

- You paste in a news headline or article.
- The app runs it through a trained TF-IDF + Logistic Regression pipeline.
- It shows:
  - **Fake** or **Real** prediction
  - A **confidence score** (probability)
  - The **top words/phrases** that pushed the model toward that prediction (explainability)

---

## Folder structure

```
fake-news-detector/
│
├── app.py                  # Streamlit web app (the UI)
├── train.py                 # Trains the model from dataset/news.csv
├── predict.py                # Loads the trained model and makes predictions
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── news.csv              # Training data (text, label)
│   └── generate_demo_dataset.py  # Creates a synthetic demo dataset
│
├── model/
│   ├── model.pkl              # Saved Logistic Regression model
│   └── vectorizer.pkl         # Saved TF-IDF vectorizer
│
└── screenshots/               # Put app screenshots here for your README/portfolio
```

---

## Setup from scratch

These steps assume you are starting with **nothing installed** beyond a computer.

### 1. Install Python
Download and install Python 3.10+ from [python.org](https://www.python.org/downloads/).
On Windows, check "Add Python to PATH" during install. Verify it worked:

```bash
python3 --version
```

### 2. Get the project files
If you're using this from a folder Claude generated for you, just open a terminal
inside the `fake-news-detector` folder. If you push it to GitHub later:

```bash
git clone https://github.com/<your-username>/fake-news-detector.git
cd fake-news-detector
```

### 3. Create a virtual environment (recommended)
This keeps this project's packages separate from everything else on your machine.

```bash
python3 -m venv venv

# Activate it:
# macOS/Linux:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

That installs: `pandas`, `scikit-learn`, `streamlit`, `joblib`, `numpy`.

---

## Training the model

A demo dataset (`dataset/news.csv`) is already included so you can run everything
immediately. To (re)train the model:

```bash
python train.py
```

This will:
1. Load `dataset/news.csv`
2. Split it into train/test sets
3. Fit a TF-IDF vectorizer on the training text
4. Train a Logistic Regression classifier
5. Print accuracy + a classification report
6. Save `model/model.pkl` and `model/vectorizer.pkl`

If you ever want to regenerate the bundled synthetic demo dataset:

```bash
python dataset/generate_demo_dataset.py
```

---

## Running the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) — open it in your
browser. Paste in text, click **Analyze**, and see the result.

You can also use the command line directly, without the UI:

```bash
python predict.py "Some headline or article text here"
```

---

## Using a real dataset (recommended for portfolio)

⚠️ **Important:** the bundled `dataset/news.csv` is a small **synthetic** dataset built
from templates (see `generate_demo_dataset.py`). It exists so the project runs
out-of-the-box, but a model trained on it will not generalize well to real news —
it will have learned this project's own templates, not genuine linguistic patterns
of misinformation. For a project you show to employers, swap in a real dataset:

**Recommended: Kaggle "Fake and Real News Dataset" (Clement Bisaillon)**
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

It ships as `Fake.csv` and `True.csv`, ~23,000 and ~21,000 articles respectively.

> **Note:** The raw dataset files and the generated `dataset/news.csv` are not
> included in this repo (they exceed GitHub's file size limits). Download
> `Fake.csv` and `True.csv` from the Kaggle link above, place them in `dataset/`,
> then run `py dataset/convert_kaggle_dataset.py` followed by `py train.py`.
Convert it into the format this project expects:

```python
import pandas as pd

fake = pd.read_csv("Fake.csv"); fake["label"] = "FAKE"
real = pd.read_csv("True.csv"); real["label"] = "REAL"

df = pd.concat([fake, real])
df["text"] = df["title"] + " " + df["text"]
df = df[["text", "label"]].sample(frac=1, random_state=42)

df.to_csv("dataset/news.csv", index=False)
```

Then just rerun `python train.py`. Other good alternatives: LIAR dataset, FakeNewsNet.

---

## How the model works

1. **TF-IDF (Term Frequency – Inverse Document Frequency)** turns each article into
   a vector of numbers. Words that appear often in one article but rarely across
   all articles get a high weight — these tend to be the most *distinctive* words.
   This project also includes bigrams (2-word phrases) to capture patterns like
   "share this" or "officials said".
2. **Logistic Regression** learns a weight for each word/phrase that indicates how
   strongly it pushes a prediction toward "Fake" or "Real".
3. At prediction time, the app multiplies each word's TF-IDF value in your input by
   its learned weight, and shows you the words that contributed most — this is what
   powers the "words that influenced this prediction" section.

This is a **linear, interpretable model** — a deliberate choice for a portfolio
project, since you can explain exactly why it made a decision (unlike a black-box
deep learning model).

---
## Limitations

This dataset is from 2016–2017, and most of its REAL articles are Reuters wire
stories with a consistent style (e.g. starting with a location and "(Reuters)").
As a result, the model partly learned "does this sound like a Reuters wire
story" rather than purely detecting misinformation. It can misclassify:
- Short headlines with little context
- Real news about recent events not resembling 2016-17 Reuters style
- Legitimate news written in a dramatic or informal tone

This is a known, common pitfall in fake-news datasets and a good discussion
point on why data quality/recency matters as much as model choice.


 ---

## Deploying it publicly (free)

The easiest way to get a shareable link for your portfolio:

1. Push this project to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click "New app", pick your repo, branch, and `app.py` as the entry point.
4. Deploy — you'll get a public URL like `https://your-app.streamlit.app`.

Make sure `model/model.pkl` and `model/vectorizer.pkl` are committed to the repo
(or that your deployment runs `train.py` on startup) so the app has a model to load.

---

## Ideas to extend this project

These are great "Future Work" bullet points for your README or resume:

- **Better model**: try `PassiveAggressiveClassifier`, `LinearSVC`, or a fine-tuned
  DistilBERT for higher accuracy.
- **Larger/more diverse data**: combine multiple datasets (LIAR, FakeNewsNet, Kaggle)
  and add recent articles so the model doesn't go stale.
- **URL input**: let users paste a link, scrape the article text (`newspaper3k` or
  `trafilatura`), and classify it automatically.
- **Source credibility check**: cross-reference the domain against a known
  low-credibility source list as an extra signal.
- **Model comparison dashboard**: show accuracy/F1 for several algorithms side by side.
- **Unit tests**: add `pytest` tests for `predict.py` to show engineering rigor.
- **Dockerize it**: add a `Dockerfile` so anyone can run it with one command.
- **CI pipeline**: GitHub Actions that runs tests + retrains on every push.

---

## Making this shine on your portfolio/resume

1. **Write a short blog post or LinkedIn post** explaining the problem, your
   approach, and one interesting challenge (e.g., "the model was 100% accurate on
   my synthetic data but I learned that meant nothing until I swapped in a real
   dataset and re-evaluated it" — this kind of honesty is a great signal to
   employers that you understand ML pitfalls like overfitting to synthetic/templated data).
2. **Add real accuracy numbers** from the real dataset (not the synthetic one) to
   your README and resume bullet, e.g. "Achieved 94% test accuracy on a 40k-article
   fake news dataset using TF-IDF + Logistic Regression."
3. **Add screenshots** of the running app to the `screenshots/` folder and embed
   them in this README (`![screenshot](screenshots/demo.png)`).
4. **Deploy it** (see above) and put the live link at the top of this README and in
   your resume/portfolio site — a clickable demo beats a code link every time.
5. **Resume bullet example:**
   > Built and deployed an NLP fake-news classifier (TF-IDF + Logistic Regression,
   > scikit-learn) achieving 94% test accuracy on a 40k-article dataset; shipped an
   > interactive Streamlit app with real-time confidence scores and model
   > explainability.

---

## License

Feel free to use this project as a learning resource or portfolio base. Add your own
license (e.g., MIT) if you plan to publish the repository.
