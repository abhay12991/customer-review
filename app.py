import streamlit as st
import pandas as pd
import pickle
import re

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

@st.cache_resource
def load_model():
    with open('models/lr_tuned.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()
LABEL_MAP = {2: 'Positive', 1: 'Neutral', 0: 'Negative'}

# TODO: replace with exact clean_text from 02_preprocessing.ipynb
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

st.title("📊 Amazon Fine Food Reviews — Sentiment Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Model Comparison", "Live Prediction", "Insights"])

with tab1:
    st.header("Exploratory Data Analysis")
    st.image("outputs/01_class_distribution.png", caption="Class Distribution")
    st.image("outputs/02_review_lengths.png", caption="Review Lengths")
    st.image("outputs/03_wordclouds.png", caption="Word Clouds")

with tab2:
    st.header("Model Comparison")
    df = pd.read_csv("outputs/model_comparison.csv")
    st.dataframe(df)
    st.image("outputs/model_comparison.png")
    st.image("outputs/lstm_training_curves.png")
    st.image("outputs/final_confusion_matrix.png")

with tab3:
    st.header("Try it yourself")
    user_input = st.text_area("Enter a review:")
    if st.button("Predict Sentiment"):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            st.success(f"Predicted Sentiment: **{LABEL_MAP[pred]}**")
        else:
            st.warning("Pehle kuch text likho.")

with tab4:
    st.header("Final Insights")
    st.image("outputs/error_analysis.png")
    st.image("outputs/final_summary.png")
    