# Spam Email Classifier

A machine learning project that detects whether an email/message is **Spam** or **Not Spam** using Natural Language Processing (NLP).

The system processes text messages, converts them into numerical features using **TF-IDF vectorization**, and predicts spam using a trained **machine learning classifier**.

---

# Project Overview

Spam detection is a classic **Natural Language Processing (NLP)** problem.
This project builds a complete spam detection pipeline that learns patterns from labeled messages and classifies new emails accordingly.

The pipeline includes:

1. Text cleaning
2. Tokenization
3. Stopword removal
4. TF-IDF feature extraction
5. Model training
6. Prediction on new messages

The trained **model** and **vectorizer** are saved and reused to allow instant predictions without retraining.

---

# Technologies Used

* Python
* Pandas
* NLTK
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression

---

# Dataset

The model is trained on the **SMS Spam Collection Dataset**, which contains **5,500+ labeled SMS messages** categorized as:

* **Spam**
* **Ham (Not Spam)**

Each message is used to train the classifier to recognize spam patterns.

---

# Machine Learning Pipeline

```
Raw Email
   ↓
Text Cleaning
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
TF-IDF Vectorization
   ↓
Trained ML Model
   ↓
Spam / Not Spam Prediction
```

---

# Installation

Clone the repository:

```
git clone https://github.com/sai161812/emaillclassifier.git
cd emailclassifier
```

Install dependencies:

```
pip install pandas nltk scikit-learn
```

Download required NLTK resources:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
```

---

# Model Persistence

The trained model and vectorizer are stored as serialized files:

```
spam_classifier.pkl
vectorizer.pkl
```

This allows the system to **load the trained model instantly** without retraining every time the program runs.

---

# Example Predictions

| Message                               | Prediction |
| ------------------------------------- | ---------- |
| "Claim your free prize now"           | Spam       |
| "Win a free lottery ticket today"     | Spam       |
| "Meeting scheduled at 3 PM tomorrow"  | Not Spam   |
| "Don't forget the project discussion" | Not Spam   |

---

# Possible Improvements

Future enhancements may include:

* Spam probability score
* REST API using FastAPI
* Explainable AI for spam keyword detection
* Real-time email filtering integration
* Web interface for testing messages

---

# Author

**Sai Prasad**
B.Tech – Artificial Intelligence and Data Science

---
