import pandas as pd
from preprocessing import apply_text_processing, normalize_metadata
from tfidf_vectorizer import compute_tfidf, save_vectorizer
from roberta_embedder import extract_embeddings
from feature_combiner import combine_features
from classifier import train_xgboost
import joblib

def main():
    print("Loading data...")
    df = pd.read_csv("data/fake_news_with_metadata.csv")
    
    print("\nDataset shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    
    # Debug: Check actual label values
    print("\nLabel column info:")
    print(f"Unique values: {sorted(df['label'].unique())[:10]}")  # Show first 10 unique
    print(f"Value counts:\n{df['label'].value_counts().head(10)}")

    metadata_cols = ['tweet_count', 'retweet_count', 'like_count', 'reply_count', 'user_verified']

    print("\nCleaning text...")
    df = apply_text_processing(df, text_column='title')

    print("Normalizing metadata...")
    df = normalize_metadata(df, metadata_cols)

    print("Generating TF-IDF features...")
    tfidf_features, tfidf_vectorizer = compute_tfidf(df['clean_text'])

    print("Generating RoBERTa embeddings...")
    roberta_features = extract_embeddings(df['roberta_text'])

    print("Combining features...")
    metadata_features = df[metadata_cols].values
    X = combine_features(tfidf_features, roberta_features, metadata_features)
    
    # Handle labels dynamically
    label_column = df['label']
    print(f"\nLabel unique values: {label_column.unique()}")
    print(f"Label dtype: {label_column.dtype}")
    
    # If labels are already numeric, use directly; otherwise map
    if pd.api.types.is_numeric_dtype(label_column):
        y = label_column
    else:
        # Try common label mappings
        if set(label_column.unique()) == {'fake', 'real'}:
            y = label_column.map({'fake': 0, 'real': 1})
        elif set(label_column.unique()) == {'True', 'False'}:
            y = label_column.map({'False': 0, 'True': 1})
        else:
            print(f"Unexpected labels: {label_column.unique()}")
            return
    
    # Remove rows with NaN labels
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices].reset_index(drop=True)
    
    print(f"Valid samples: {len(y)}")

    print("Training XGBoost classifier...")
    train_xgboost(X, y)
    
    # Save vectorizers for prediction
    print("\nSaving vectorizers...")
    save_vectorizer(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(roberta_features, "roberta_model_config.pkl")  # Save for later use

if __name__ == "__main__":
    main()
