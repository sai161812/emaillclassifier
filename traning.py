import pandas as pd
import string
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

data = pd.read_csv("datasett/dataset/SMSSpamCollection",sep='\t', names=["label", "message"])
print(data.head())
print(data.shape)

print(data['label'].value_counts())

data['label'] = data['label'].map({'ham':0, 'spam':1})
print(data.tail())

def filter(text):
    
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['message'] = data['message'].apply(filter)
print(data.head())
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def process(text):
    wordtoken = word_tokenize(text)
    filtered = []
    for word in wordtoken:
        if word not in stop_words:
            stemmed_word = stemmer.stem(word)
            filtered.append(stemmed_word)
    return " ".join(filtered)

data['message'] = data['message'].apply(process)

vector = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000,
    min_df=2
)
a = vector.fit_transform(data['message'])
b = data['label']

a_train, a_test, b_train, b_test = train_test_split(
    a,
    b,
    test_size= 0.2,
    random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(a_train, b_train)

b_pred = model.predict(a_test)

accuracy = accuracy_score(b_test, b_pred)
print(f"accuracy is {accuracy}")

print(confusion_matrix(b_test, b_pred))
print(classification_report(b_test, b_pred))

def spam_prediction(mail):
    cleaned = filter(mail)
    processed = process(cleaned)
    vectorized = vector.transform([processed])
    prediction = model.predict(vectorized)

    if prediction[0] == 1:
        return "SPAM"
    else:
        return "NOT SPAM"
    
with open("spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vector, f)