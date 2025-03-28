import streamlit as st
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load trained model and data
model = load_model("model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("texts.pkl", "rb"))
classes = pickle.load(open("labels.pkl", "rb"))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else "unknown"

def get_response(intent):
    for i in intents["intents"]:
        if i["tag"] == intent:
            return np.random.choice(i["responses"])
    return "I'm not sure how to respond to that."

# Streamlit UI
st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("🧘 Mental Health Chatbot")
st.markdown("Talk to me about how you're feeling. I'm here to help! 💙")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

user_input = st.chat_input("How do you feel today?")
if user_input:
    st.session_state.messages.append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    intent = predict_class(user_input)
    response = get_response(intent)
    
    st.session_state.messages.append({"role": "assistant", "text": response})
    with st.chat_message("assistant"):
        st.markdown(response)
