from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(corpus, max_features=5000):
    vec = TfidfVectorizer(max_features=max_features)
    X = vec.fit_transform(corpus)
    return X, vec
