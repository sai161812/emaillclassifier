#this is how to actually use the model in projects

import pickle
from txtcleaner import clean_text, process_text

with open("model/spam_classifier.pkl","rb") as f:
    model = pickle.load(f)
with open("model/vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)

def predict_spam(email):

    cleaned = clean_text(email)
    processed = process_text(cleaned)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        return "SPAM"
    
    return "NOT SPAM"
#import predict_spam function in other projects to actually use it 