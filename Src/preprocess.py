import pandas as pd
import re
import joblib
import logging
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

# Load stopwords (safe)
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = text.split()  # ✅ NO NLTK TOKENIZER
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "movies.csv")

def build_model(csv_path=CSV_PATH):

    logging.info("🚀 Building model from CSV...")

    df = pd.read_csv(
        csv_path,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip"
    )

    required_columns = ["genres", "keywords", "overview", "title"]
    df = df[required_columns].dropna().reset_index(drop=True)

    df["combined"] = df["genres"] + " " + df["keywords"] + " " + df["overview"]
    df["cleaned_text"] = df["combined"].apply(preprocess_text)

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])
    cosine_sim = cosine_similarity(tfidf_matrix)

    joblib.dump(df, "df_cleaned.pkl")
    joblib.dump(cosine_sim, "cosine_sim.pkl")

    logging.info("✅ Model files created.")
    return df, cosine_sim
