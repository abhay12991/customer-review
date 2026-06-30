# import streamlit as st
# import pandas as pd
# import pickle
# import re

# st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# @st.cache_resource
# def load_model():
#     with open('models/lr_tuned.pkl', 'rb') as f:
#         model = pickle.load(f)
#     with open('models/tfidf_vectorizer.pkl', 'rb') as f:
#         vectorizer = pickle.load(f)
#     return model, vectorizer

# model, vectorizer = load_model()
# LABEL_MAP = {2: 'Positive', 1: 'Neutral', 0: 'Negative'}

# # TODO: replace with exact clean_text from 02_preprocessing.ipynb
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-z\s]', '', text)
#     return text

# st.title("📊 Amazon Fine Food Reviews — Sentiment Dashboard")

# tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Model Comparison", "Live Prediction", "Insights"])

# with tab1:
#     st.header("Exploratory Data Analysis")
#     st.image("outputs/01_class_distribution.png", caption="Class Distribution")
#     st.image("outputs/02_review_lengths.png", caption="Review Lengths")
#     st.image("outputs/03_wordclouds.png", caption="Word Clouds")

# with tab2:
#     st.header("Model Comparison")
#     df = pd.read_csv("outputs/model_comparison.csv")
#     st.dataframe(df)
#     st.image("outputs/model_comparison.png")
#     st.image("outputs/lstm_training_curves.png")
#     st.image("outputs/final_confusion_matrix.png")

# with tab3:
#     st.header("Try it yourself")
#     user_input = st.text_area("Enter a review:")
#     if st.button("Predict Sentiment"):
#         if user_input.strip():
#             cleaned = clean_text(user_input)
#             vec = vectorizer.transform([cleaned])
#             pred = model.predict(vec)[0]
#             st.success(f"Predicted Sentiment: **{LABEL_MAP[pred]}**")
#         else:
#             st.warning("Pehle kuch text likho.")

# with tab4:
#     st.header("Final Insights")
#     st.image("outputs/error_analysis.png")
#     st.image("outputs/final_summary.png")


import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# Download stopwords quietly on first run (cached after that)
nltk.download('stopwords', quiet=True)
SW = set(stopwords.words('english'))


@st.cache_resource
def load_model():
    with open('models/lr_tuned.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


model, vectorizer = load_model()
LABEL_MAP = {2: 'Positive', 1: 'Neutral', 0: 'Negative'}


def clean_text(text):
    """
    Matches the Week 2 training pipeline as closely as possible
    without requiring spaCy (too heavy for Streamlit Cloud free tier).
    Steps: lowercase -> remove HTML/URLs -> letters only ->
           remove stopwords -> drop very short words.
    """
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)                 # remove HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)         # remove URLs
    text = re.sub(r"[^a-z\s']", ' ', text)              # keep letters only
    text = re.sub(r'\s+', ' ', text).strip()            # collapse whitespace

    tokens = [w for w in text.split() if w not in SW and len(w) > 2]
    return ' '.join(tokens)


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
    st.image("outputs/final_confusion_matrix.png")

with tab3:
    st.header("Try it yourself")
    user_input = st.text_area(
        "Enter a review:",
        placeholder="e.g. I absolutely loved this product, will buy again!"
    )
    if st.button("Predict Sentiment", type="primary"):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]

            label = LABEL_MAP[pred]
            confidence = proba.max()

            if label == "Positive":
                st.success(f"Predicted Sentiment: **{label}**  (confidence: {confidence:.0%})")
            elif label == "Negative":
                st.error(f"Predicted Sentiment: **{label}**  (confidence: {confidence:.0%})")
            else:
                st.warning(f"Predicted Sentiment: **{label}**  (confidence: {confidence:.0%})")

            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", f"{proba[2]:.0%}")
            col2.metric("Neutral", f"{proba[1]:.0%}")
            col3.metric("Negative", f"{proba[0]:.0%}")
        else:
            st.warning("Please enter a review first.")

with tab4:
    st.header("Final Insights")
    st.image("outputs/error_analysis.png")
    st.image("outputs/final_summary.png")