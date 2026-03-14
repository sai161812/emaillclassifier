# Spam Email Classifier

A machine learning project that detects whether an email/message is **Spam** or **Not Spam** using Natural Language Processing (NLP) and a trained classification model.

This project builds a complete spam detection pipeline including text preprocessing, feature extraction using TF-IDF, model training, evaluation, and a reusable prediction system.

---

# Project Overview

Spam detection is a classic Natural Language Processing problem.
This project trains a classifier to automatically identify spam messages by learning patterns from a labeled dataset.

The pipeline performs the following steps:

1. Text cleaning
2. Tokenization and stopword removal
3. Feature extraction using TF-IDF
4. Model training using a classifier
5. Prediction on new messages

The trained model and vectorizer are saved and reused for future predictions without retraining.

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

The model is trained on the **SMS Spam Collection Dataset**, which contains over **5,500 labeled SMS messages** categorized as:

* Spam
* Ham (Not Spam)

Each message is used to train the classifier to recognize spam patterns.

---

# Machine Learning Pipeline

The classifier uses the following pipeline:

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

```

Install dependencies.
```

pip install pandas nltk scikit-learn
```

Download NLTK resources.

```
python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
```

---

# Running the Project

Example usage:

```
Enter email text: Claim your free prize now
Prediction: SPAM
```

Another example:

```
Enter email text: Meeting rescheduled on Monday
Prediction: NOT SPAM
```

---

# Model Persistence

The trained model and vectorizer are stored as serialized files:

```
spam_classifier.pkl
vectorizer.pkl
```

This allows the system to load the model instantly without retraining.

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

---

# Author

Sai Prasad
B.Tech – Artificial Intelligence and Data Science

---
