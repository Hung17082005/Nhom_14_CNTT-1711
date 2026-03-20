import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys
sys.path.append(".")

from src.data.cleaner import clean_text
from src.features.builder import build_tfidf
from src.models.supervised import train_classifier
from langdetect import detect, LangDetectException

st.set_page_config(page_title="Hotel Mining", layout="wide")

st.title("🏨 Hotel Review Analysis")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Raw Data Visualizations
    if "rating" in df.columns:
        st.subheader("Raw Rating Distribution")
        plt.rcParams['font.size'] = 6
        fig, ax = plt.subplots(figsize=(2, 1))
        ax.hist(df["rating"].dropna(), bins=10, edgecolor='black')
        ax.set_xlabel("Rating")
        ax.set_ylabel("Frequency")
        ax.tick_params(axis='both', which='major', labelsize=4)
        st.pyplot(fig)

    if "sentiment" in df.columns:
        st.subheader("Raw Sentiment Distribution")
        plt.rcParams['font.size'] = 6
        sentiment_counts = df["sentiment"].value_counts()
        fig, ax = plt.subplots(figsize=(2, 1))
        sentiment_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.tick_params(axis='both', which='major', labelsize=4)
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig)

    if "review" not in df.columns:
        st.error("Dataset must contain 'review' column")
    else:
        df["clean"] = df["review"].apply(clean_text)
        
        st.subheader("Cleaned Data")
        st.dataframe(df[["review", "clean"]].head(10))

        # Word Cloud
        st.subheader("Word Cloud of Cleaned Reviews")
        text = " ".join(review for review in df["clean"].dropna())
        wordcloud = WordCloud(width=200, height=100, background_color='white', max_font_size=20).generate(text)
        plt.rcParams['font.size'] = 6
        fig, ax = plt.subplots(figsize=(2, 1))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        X, vec = build_tfidf(df["clean"])

        # Rating Distribution
        if "rating" in df.columns:
            st.subheader("Rating Distribution")
            plt.rcParams['font.size'] = 6
            fig, ax = plt.subplots(figsize=(2, 1))
            ax.hist(df["rating"].dropna(), bins=10, edgecolor='black')
            ax.set_xlabel("Rating")
            ax.set_ylabel("Frequency")
            ax.tick_params(axis='both', which='major', labelsize=4)
            st.pyplot(fig)

        # Sentiment Distribution
        if "sentiment" in df.columns:
            st.subheader("Sentiment Distribution")
            plt.rcParams['font.size'] = 6
            sentiment_counts = df["sentiment"].value_counts()
            fig, ax = plt.subplots(figsize=(2, 1))
            sentiment_counts.plot(kind='bar', ax=ax)
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            ax.tick_params(axis='both', which='major', labelsize=4)
            ax.tick_params(axis='x', rotation=0)
            st.pyplot(fig)

            # Top keywords for positive and negative
            st.subheader("Top Keywords for Sentiment")
            positive_reviews = df[df["sentiment"] == "positive"]["clean"]
            negative_reviews = df[df["sentiment"] == "negative"]["clean"]

            if not positive_reviews.empty:
                positive_text = " ".join(positive_reviews)
                positive_words = positive_text.split()
                positive_freq = pd.Series(positive_words).value_counts().head(10)
                st.write("**Positive Keywords:**", positive_freq.to_dict())

            if not negative_reviews.empty:
                negative_text = " ".join(negative_reviews)
                negative_words = negative_text.split()
                negative_freq = pd.Series(negative_words).value_counts().head(10)
                st.write("**Negative Keywords:**", negative_freq.to_dict())

            model, report, cm = train_classifier(X, df["sentiment"])
            st.success("Model trained successfully!")
            st.subheader("Model Evaluation")
            st.text(report)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")

import matplotlib.pyplot as plt

classes = sorted(df["sentiment"].dropna().unique())

fig, ax = plt.subplots()
cax = ax.matshow(cm)

plt.title("Confusion Matrix")
fig.colorbar(cax)

ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))

ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

plt.xlabel("Predicted")
plt.ylabel("Actual")

# hiển thị số trong ô
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], va='center', ha='center')

st.pyplot(fig)