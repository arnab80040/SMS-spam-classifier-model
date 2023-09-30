import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS spam classifier")

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    for word in text:
        if word.isalnum() == False:
            text.remove(word)

    for word in text:
        if word in stopwords.words('english') or word in string.punctuation:
            text.remove(word)

    y = []
    for word in text:
        y.append(ps.stem(word))
    text = y[:]
    y.clear()
    text = " ".join(text)
    return text



input_sms = st.text_input("Enter the message")

if st.button('Predict'): # that is, if button is clicked
    # 1) preprocess
    transformed_sms = transform_text(input_sms)
    # 2) vectorize
    vector_input = cv.transform([transformed_sms])
    # 3) Predict
    result = model.predict(vector_input)[0]
    # 4) Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")
