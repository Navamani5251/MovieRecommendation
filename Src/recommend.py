# recommend.py
import joblib
import logging
import os
import pandas as pd
from preprocess import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

PKL_DF = "df_cleaned.pkl"
PKL_SIM = "cosine_sim.pkl"

logging.info("🔁 Checking model files...")

if not os.path.exists(PKL_DF) or not os.path.exists(PKL_SIM):
    logging.info("⚙️ PKL files not found. Creating from movies.csv...")

    df = pd.read_csv("../movies.csv")

    df['combined'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['overview']
    df['cleaned_text'] = df['combined'].apply(preprocess_text)

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    joblib.dump(df, PKL_DF)
    joblib.dump(cosine_sim, PKL_SIM)

    logging.info("✅ PKL files created successfully.")
else:
    logging.info("📦 Loading existing PKL files...")
    df = joblib.load(PKL_DF)
    cosine_sim = joblib.load(PKL_SIM)



def recommend_movies(movie_name, top_n=5):
    logging.info("🎬 Recommending movies for: '%s'", movie_name)
    idx = df[df['title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        logging.warning("⚠️ Movie not found in dataset.")
        return None
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    logging.info("✅ Top %d recommendations ready.", top_n)
    # Create DataFrame with clean serial numbers starting from 1
    result_df = df[['title']].iloc[movie_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1  # Start from 1 instead of 0
    result_df.index.name = "S.No."


    return result_df

