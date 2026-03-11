import re
import nltk
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(nltk.corpus.stopwords.words('english'))

def clean_text(text: str) -> str:
    """Clean text for TF-IDF (remove noise)."""

    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)

    text = text.lower()

    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]

    return " ".join(tokens)


def apply_text_processing(df: pd.DataFrame, text_column: str = "title") -> pd.DataFrame:
    """
    Create two text pipelines:
    clean_text  -> for TF-IDF
    roberta_text -> for RoBERTa
    """

    df["clean_text"] = df[text_column].apply(clean_text)
    df["roberta_text"] = df[text_column] 

    return df

def normalize_metadata(df: pd.DataFrame, metadata_cols: list) -> pd.DataFrame:
    """Normalize metadata features using MinMaxScaler."""
    scaler = MinMaxScaler()
    df[metadata_cols] = scaler.fit_transform(df[metadata_cols])
    return df
