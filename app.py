import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="Customer Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="centered"
)

# Sidebar
st.sidebar.title("About This Project")
st.sidebar.write("""
This application demonstrates how **deep learning (LSTM)** can be used to analyze customer reviews.

The model predicts sentiment from textual feedback and helps organizations understand customer perception.

Sentiment Classes:
- Positive
- Negative
- Neutral
""")

# Load trained model
model = tf.keras.models.load_model("sentiment_lstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

max_len = 100

labels = {0:"Negative",1:"Positive",2:"Neutral"}

# Title
st.title("Customer Sentiment Analysis using LSTM")

st.markdown("""
This interactive dashboard uses a **Long Short-Term Memory (LSTM)** deep learning model to analyze customer reviews and predict sentiment.

The model was trained on scraped customer feedback data and can classify reviews into:

- **Positive**
- **Negative**
- **Neutral**

Enter a customer review below to see the predicted sentiment.
""")

# Model information
st.markdown("### Model Information")

st.info("""
Model Type: LSTM Neural Network  
Dataset Size: ~10,000 reviews  
Accuracy: ~88%  
Explainability: SHAP  
""")

# User input
user_input = st.text_area("Customer Review")

# Predict sentiment
if st.button("Predict Sentiment"):

    if user_input.strip() != "":

        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq,maxlen=max_len)

        prediction = model.predict(padded)

        probs = prediction[0]
        sorted_probs = sorted(probs, reverse=True)

        # Confidence-aware prediction
        if sorted_probs[0] - sorted_probs[1] < 0.15:
            sentiment = "Neutral (Low Confidence)"
        else:
            sentiment = labels[np.argmax(probs)]

        st.markdown("## Predicted Sentiment")

        if sentiment.startswith("Positive"):
            st.success("Positive 😊")

        elif sentiment.startswith("Negative"):
            st.error("Negative 😡")

        else:
            st.warning("Neutral 😐")

        # Probability chart
        st.markdown("### Prediction Probabilities")

        prob_df = pd.DataFrame({
            "Sentiment":["Negative","Positive","Neutral"],
            "Probability":probs
        })

        st.bar_chart(prob_df.set_index("Sentiment"))

        # Probability table
        st.markdown("### Probability Table")

        st.table(prob_df)

        # Review statistics
        st.markdown("### Review Statistics")

        word_count = len(user_input.split())

        st.write("Word Count:",word_count)

    else:
        st.warning("Please enter a review.")

# Example reviews
st.markdown("### Try Example Reviews")

examples = [
    "This app is amazing and works perfectly",
    "The update made the app crash frequently",
    "The app is okay but could be better"
]

for ex in examples:

    if st.button(ex):

        seq = tokenizer.texts_to_sequences([ex])
        padded = pad_sequences(seq,maxlen=max_len)

        prediction = model.predict(padded)
        probs = prediction[0]
        sentiment = labels[np.argmax(probs)]

        st.write("Prediction:",sentiment)

# Business insight
st.markdown("### Business Insight")

st.write("""
Customer sentiment analysis helps organizations monitor customer satisfaction in real time.

Negative feedback highlights operational issues, while positive feedback reveals strengths that can guide marketing strategies and product improvements.
""")

# Footer
st.markdown("---")

st.markdown("""
Built using **Python, NLP, LSTM Deep Learning, and Streamlit**

Capstone Project – Customer Sentiment Analysis and Business Decision Support
""")