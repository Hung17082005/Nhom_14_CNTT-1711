import pandas as pd

from src.data.loader import load_data
from src.data.cleaner import clean_text
from src.features.builder import build_tfidf

from src.mining.association import mine_rules
from src.mining.clustering import run_kmeans
from src.models.supervised import train_classifier
from src.models.regression import train_regression

# Load data
df = load_data("data/raw/hotel_reviews.csv")

# Clean
df["clean"] = df["review"].apply(clean_text)

# Feature
X, vec = build_tfidf(df["clean"])

# === Mining ===
rules = mine_rules(df, "clean")
print(rules.head())

# === Clustering ===
labels, _ = run_kmeans(X, k=5)

# === Classification ===
if "sentiment" in df.columns:
    train_classifier(X, df["sentiment"])

# === Regression ===
if "rating" in df.columns:
    train_regression(X, df["rating"])