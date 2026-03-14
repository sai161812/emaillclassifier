import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#run if necessary
#nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def process_text(text):

    words = word_tokenize(text)
    filtered = []
    for word in words:
        if word not in stop_words:
            filtered.append(word)
    return " ".join(filtered)
