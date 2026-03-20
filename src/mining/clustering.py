from sklearn.cluster import KMeans
import numpy as np

def run_kmeans(X, k=5):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def get_top_terms_per_cluster(X, vectorizer, labels, top_n=10):
    terms = vectorizer.get_feature_names_out()
    centers = np.array(X.todense())
    
    results = {}
    for i in set(labels):
        cluster_idx = labels == i
        mean_vec = centers[cluster_idx].mean(axis=0)
        top_terms = [terms[j] for j in mean_vec.argsort()[-top_n:]]
        results[i] = top_terms

    return results