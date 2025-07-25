import streamlit as st
import nltk
import nltk
nltk.data.path.append(r"C:\Users\Heman\AppData\Roaming\nltk_data")
nltk.download('punkt')
nltk.download('stopwords')
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Download NLTK data only if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load the vectorizer and model
tfidf = pickle.load(open('Vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("📩 Email/SMS Spam Detection System")
input_sms = st.text_input("✍️ Enter your SMS/Email message:")


# Text preprocessing function
def TextProcessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    stop_words = set(stopwords.words('english'))
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)


# Predict only if input is not empty
if input_sms:
    # Preprocess
    processed_msg = TextProcessing(input_sms)

    # Vectorize
    vector_input = tfidf.transform([processed_msg])

    # Predict
    result = model.predict(vector_input)[0]

    # Display result
    if result == 'spam' or result == 1:
        st.header("🚫 Yes, the message is SPAM!")
    else:
        st.header("✅ This is NOT a spam message.")
