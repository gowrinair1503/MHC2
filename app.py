import streamlit as st
import nltk
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

# Load resources
nltk.download('punkt')
lemmatizer = nltk.stem.WordNetLemmatizer()
model = load_model('model.h5')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Load translation models
eng_swa_tokenizer = AutoTokenizer.from_pretrained("Rogendo/en-sw")
eng_swa_model = AutoModelForSeq2SeqLM.from_pretrained("Rogendo/en-sw")
eng_swa_translator = pipeline("text2text-generation", model=eng_swa_model, tokenizer=eng_swa_tokenizer)

def translate_text_eng_swa(text):
    return eng_swa_translator(text, max_length=128, num_beams=5)[0]['generated_text']

swa_eng_tokenizer = AutoTokenizer.from_pretrained("Rogendo/sw-en")
swa_eng_model = AutoModelForSeq2SeqLM.from_pretrained("Rogendo/sw-en")
swa_eng_translator = pipeline("text2text-generation", model=swa_eng_model, tokenizer=swa_eng_tokenizer)

def translate_text_swa_eng(text):
    return swa_eng_translator(text, max_length=128, num_beams=5)[0]['generated_text']

# Language detection
nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=lambda nlp, name: LanguageDetector())
nlp.add_pipe('language_detector', last=True)

def detect_language(text):
    doc = nlp(text)
    return doc._.language['language']

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints):
    if ints:
        tag = ints[0]['intent']
        for i in intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "Sorry, I didn't understand that."

def chatbot_response(msg):
    detected_language = detect_language(msg)
    if detected_language == "en":
        res = get_response(predict_class(msg))
    elif detected_language == "sw":
        translated_msg = translate_text_swa_eng(msg)
        res = get_response(predict_class(translated_msg))
        res = translate_text_eng_swa(res)
    else:
        res = "Sorry, I can only understand English and Swahili."
    return res

# Streamlit UI
st.title("Chatbot with English-Swahili Translation")
st.write("This chatbot can respond in English and Swahili.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You: ", "")
if st.button("Send") and user_input:
    response = chatbot_response(user_input)
    st.session_state.chat_history.append((user_input, response))

for user_msg, bot_msg in st.session_state.chat_history:
    st.write(f"You: {user_msg}")
    st.write(f"Bot: {bot_msg}")



