import sys
import pandas as pd
import joblib
from preprocessing import apply_text_processing, normalize_metadata
from tfidf_vectorizer import compute_tfidf, load_vectorizer
from roberta_embedder import extract_embeddings
from feature_combiner import combine_features

# =========================
# CONFIG
# =========================

MODEL_PATH = "fake_news_xgboost.pkl"
TFIDF_PATH = "tfidf_vectorizer.pkl"

METADATA_COLS = ['tweet_count', 'retweet_count', 'like_count', 'reply_count', 'user_verified']

LABEL_MAP = {
    0: "Fake",
    1: "Real"
}

# =========================
# LOAD MODEL & VECTORIZER
# =========================

print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded!")

print("Loading TF-IDF vectorizer...")
tfidf_vectorizer = load_vectorizer(TFIDF_PATH)
print("Vectorizer loaded!")

# =========================
# READ INPUT FILE
# =========================

if len(sys.argv) < 2:
    print("Usage: python predict.py test_news.csv")
    sys.exit()

input_file = sys.argv[1]

print("Reading dataset:", input_file)

df = pd.read_csv(input_file)

print("Samples:", len(df))

# =========================
# PREPROCESS DATA (same as training)
# =========================

print("\nPreprocessing...")

# Check for metadata columns
missing = [f for f in METADATA_COLS if f not in df.columns]
if missing:
    print("Missing columns:", missing)
    # Fill with zeros if metadata columns are missing
    for col in missing:
        df[col] = 0

print("Cleaning text...")
df = apply_text_processing(df, text_column='title')

print("Normalizing metadata...")
df = normalize_metadata(df, METADATA_COLS)

print("Generating TF-IDF features...")
tfidf_features, _ = compute_tfidf(df['clean_text'], fitted_vectorizer=tfidf_vectorizer)

print("Generating RoBERTa embeddings...")
roberta_features = extract_embeddings(df['clean_text'])

print("Combining features...")
metadata_features = df[METADATA_COLS].values
X = combine_features(tfidf_features, roberta_features, metadata_features)

# =========================
# PREDICT
# =========================

print("Predicting...")

pred = model.predict(X)

df["prediction"] = pred
df["prediction_label"] = df["prediction"].map(LABEL_MAP)
prob = model.predict_proba(X)[:,1]
df["confidence_real"] = prob

# =========================
# SAVE RESULTS
# =========================

output_file = "prediction_results.csv"

df.to_csv(output_file, index=False)

print("Prediction complete!")
print("Results saved to:", output_file)

print("\nPreview:")
print(df[["title", "prediction_label", "confidence_real"]].head(20))

if "label" in df.columns:

    print("\nEvaluating predictions...")

    df["correct"] = df["prediction"] == df["label"]

    accuracy = df["correct"].mean()

    print("Accuracy:", accuracy)

    print("\nCorrect predictions:", df["correct"].sum())
    print("Wrong predictions:", (~df["correct"]).sum())
    