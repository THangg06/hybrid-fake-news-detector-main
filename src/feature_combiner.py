import numpy as np

def combine_features(tfidf_matrix, roberta_embeddings, metadata):
    """Concatenate all feature sets into one final matrix."""
    return np.hstack((tfidf_matrix, roberta_embeddings, metadata))
