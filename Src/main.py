# main.py - Modern Cinematic Movie Recommendation Platform
import os
import sys
import json
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from recommend import df, recommend_movies
from omdp_utils import get_movie_details

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CineMatch AI • Cinematic Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Configuration & API Key Loader
# ---------------------------------------------------------
CONFIG_CANDIDATES = [
    "config.json",
    os.path.join(os.path.dirname(__file__), "config.json"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
]
OMDB_API_KEY = ""
for cfg_path in CONFIG_CANDIDATES:
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                OMDB_API_KEY = config.get("OMDB_API_KEY", "")
                if OMDB_API_KEY:
                    break
        except Exception:
            pass

# ---------------------------------------------------------
# Cached Enriched Metadata Fetcher
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_enriched_movie_data(title: str, api_key: str):
    """
    Retrieves plot and poster using the original get_movie_details function,
    and enriches with year, genre, IMDb rating, and runtime for the modern UI.
    Includes fallback to local dataset overview if OMDB plot is missing.
    """
    plot, poster = get_movie_details(title, api_key)
    
    details = {
        "title": title,
        "plot": plot if plot != "N/A" else "",
        "poster": poster,
        "year": "N/A",
        "genre": "N/A",
        "rating": "N/A",
        "runtime": "N/A",
        "director": "N/A",
        "actors": "N/A"
    }

    # Fetch supplementary metadata from OMDb when API key is available
    if api_key and api_key != "N/A":
        try:
            url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
            res = requests.get(url, timeout=3.5).json()
            if res.get("Response") == "True":
                details["year"] = res.get("Year", "N/A")
                details["genre"] = res.get("Genre", "N/A")
                details["rating"] = res.get("imdbRating", "N/A")
                details["runtime"] = res.get("Runtime", "N/A")
                details["director"] = res.get("Director", "N/A")
                details["actors"] = res.get("Actors", "N/A")
                if not details["plot"] and res.get("Plot") and res.get("Plot") != "N/A":
                    details["plot"] = res.get("Plot")
        except Exception:
            pass

    # Fallback to local dataset overview if plot is still blank
    if not details["plot"]:
        try:
            matched = df[df["title"].str.lower() == title.lower()]
            if not matched.empty and "overview" in matched.columns:
                local_overview = matched.iloc[0]["overview"]
                if isinstance(local_overview, str) and local_overview.strip():
                    details["plot"] = local_overview
        except Exception:
            pass

    if not details["plot"]:
        details["plot"] = "Synopsis not available for this title."

    return details


# ---------------------------------------------------------
# Custom Cinematic CSS Design System
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background & Core Layout */
.main, [data-testid="stAppViewContainer"] {
    background: radial-gradient(130% 120% at 50% 0%, #0d1424 0%, #060910 65%, #020407 100%);
    color: #e2e8f0;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1350px !important;
}

/* Brand Hero Section */
.hero-wrapper {
    text-align: center;
    padding: 0.8rem 0 1.5rem 0;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #ffffff 15%, #cbd5e1 55%, #e50914 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}

.hero-tagline {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    font-weight: 400;
}

.stats-bar {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.6rem;
    margin-top: 1rem;
}

.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.8rem;
    background: rgba(22, 30, 48, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #cbd5e1;
    backdrop-filter: blur(10px);
}

/* Search Card Container */
.quick-pick-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.4rem;
}

/* Spotlight Card */
.spotlight-card {
    background: linear-gradient(145deg, rgba(20, 28, 46, 0.85) 0%, rgba(10, 15, 26, 0.95) 100%);
    border: 1px solid rgba(229, 9, 20, 0.25);
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.65), 0 0 25px -5px rgba(229, 9, 20, 0.12);
}

.spotlight-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.75rem;
    background: rgba(229, 9, 20, 0.15);
    border: 1px solid rgba(229, 9, 20, 0.4);
    color: #ff4d4d;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
}

.spotlight-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}

.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.85rem;
    align-items: center;
}

.badge-rating {
    background: rgba(245, 197, 24, 0.15);
    border: 1px solid rgba(245, 197, 24, 0.45);
    color: #f5c518;
    font-weight: 700;
    font-size: 0.8rem;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}

.badge-genre {
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #38bdf8;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
}

.badge-year {
    background: rgba(148, 163, 184, 0.12);
    border: 1px solid rgba(148, 163, 184, 0.25);
    color: #cbd5e1;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
}

.badge-runtime {
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(168, 85, 247, 0.3);
    color: #c084fc;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
}

.spotlight-plot {
    font-size: 0.95rem;
    line-height: 1.6;
    color: #cbd5e1;
    margin-bottom: 0.8rem;
}

/* Movie Card System */
.movie-card {
    background: #0f1626;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.45);
    transition: transform 0.28s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.28s ease, border-color 0.28s ease;
    margin-bottom: 0.75rem;
}

.movie-card:hover {
    transform: translateY(-7px);
    border-color: rgba(229, 9, 20, 0.55);
    box-shadow: 0 18px 36px rgba(0, 0, 0, 0.7), 0 0 20px rgba(229, 9, 20, 0.22);
}

.poster-box {
    position: relative;
    width: 100%;
    padding-top: 148%; /* 2:3 cinematic poster ratio */
    overflow: hidden;
    background: #172033;
}

.poster-box img {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.4s ease;
}

.movie-card:hover .poster-box img {
    transform: scale(1.05);
}

.poster-placeholder {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.25rem;
    text-align: center;
    background: linear-gradient(160deg, #182238 0%, #0d1320 100%);
}

.poster-placeholder .icon {
    font-size: 2.75rem;
    margin-bottom: 0.6rem;
    opacity: 0.7;
}

.poster-placeholder .fallback-text {
    font-size: 0.85rem;
    font-weight: 700;
    color: #94a3b8;
    line-height: 1.3;
}

.card-rank {
    position: absolute;
    top: 8px;
    left: 8px;
    background: rgba(10, 15, 26, 0.85);
    border: 1px solid rgba(229, 9, 20, 0.6);
    color: #ff4d4d;
    font-size: 0.72rem;
    font-weight: 800;
    padding: 0.22rem 0.55rem;
    border-radius: 6px;
    backdrop-filter: blur(6px);
    z-index: 2;
}

.card-info {
    padding: 1rem 0.9rem 0.75rem 0.9rem;
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}

.card-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.3;
}

.card-snippet {
    font-size: 0.8rem;
    line-height: 1.5;
    color: #94a3b8;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: 0.5rem;
}

/* Custom UI Inputs & Buttons */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e50914 0%, #b80610 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.75rem !important;
    box-shadow: 0 4px 18px rgba(229, 9, 20, 0.4) !important;
    transition: all 0.25s ease !important;
}

div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 24px rgba(229, 9, 20, 0.7) !important;
    transform: translateY(-2px) !important;
}

div.stButton > button[kind="secondary"] {
    background: rgba(22, 30, 48, 0.6) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

div.stButton > button[kind="secondary"]:hover {
    background: rgba(35, 47, 72, 0.9) !important;
    border-color: rgba(229, 9, 20, 0.5) !important;
    color: #ffffff !important;
}

/* Streamlit Expander Polishing */
[data-testid="stExpander"] {
    background: rgba(15, 22, 36, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem !important;
}

/* Section Header */
.section-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    color: #ffffff;
    margin: 1.5rem 0 0.3rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-subtext {
    font-size: 0.92rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
movie_list = sorted(df['title'].dropna().unique())

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = "The Dark Knight" if "The Dark Knight" in movie_list else movie_list[0]

if "has_searched" not in st.session_state:
    st.session_state.has_searched = False

if "top_n" not in st.session_state:
    st.session_state.top_n = 5


# ---------------------------------------------------------
# Header & Brand Hero
# ---------------------------------------------------------
st.markdown("""
<div class="hero-wrapper">
    <h1 class="hero-title">CINEMATCH AI</h1>
    <div class="hero-tagline">Intelligent Content-Based Movie Recommendation & Cinematic Discovery</div>
    <div class="stats-bar">
        <span class="stat-pill">🎬 4,380+ Curated Titles</span>
        <span class="stat-pill">🧠 TF-IDF Vector Space</span>
        <span class="stat-pill">📐 Cosine Semantic Similarity</span>
        <span class="stat-pill">⚡ OMDb API Live Metadata</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Quick-Picks Curated Chips Bar
# ---------------------------------------------------------
QUICK_PICKS = [
    "The Dark Knight",
    "Inception",
    "Interstellar",
    "Avatar",
    "Pulp Fiction",
    "The Matrix",
    "Gladiator",
    "Titanic",
    "Toy Story"
]
available_quick_picks = [m for m in QUICK_PICKS if m in movie_list]

st.markdown('<div class="quick-pick-title">⚡ Quick Discover Popular Titles:</div>', unsafe_allow_html=True)
chip_cols = st.columns(len(available_quick_picks))
for idx, q_title in enumerate(available_quick_picks):
    with chip_cols[idx]:
        if st.button(q_title, key=f"quick_chip_{idx}", use_container_width=True):
            st.session_state.selected_movie = q_title
            st.session_state.has_searched = True
            st.rerun()

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Search & Control Panel
# ---------------------------------------------------------
with st.container():
    col_search, col_depth, col_action = st.columns([5, 2, 3], gap="medium")

    with col_search:
        # Determine current selectbox index
        current_idx = 0
        if st.session_state.selected_movie in movie_list:
            current_idx = movie_list.index(st.session_state.selected_movie)

        selected_movie = st.selectbox(
            "🎬 Search or Select Movie:",
            movie_list,
            index=current_idx,
            key="movie_selector",
            help="Type to search through 4,380+ titles in the semantic database."
        )

        # Update session state if dropdown changed
        if selected_movie != st.session_state.selected_movie:
            st.session_state.selected_movie = selected_movie
            # Reset recommendation trigger so user views the spotlight of newly selected film
            st.session_state.has_searched = False
            st.rerun()

    with col_depth:
        top_n = st.selectbox(
            "🎯 Matches Count:",
            [5, 6, 8, 10],
            index=[5, 6, 8, 10].index(st.session_state.top_n) if st.session_state.top_n in [5, 6, 8, 10] else 0,
            help="Choose how many similar movies to retrieve."
        )
        st.session_state.top_n = top_n

    with col_action:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        recommend_clicked = st.button("🚀 Recommend Similar Movies", type="primary", use_container_width=True)
        if recommend_clicked:
            st.session_state.has_searched = True


# ---------------------------------------------------------
# Selected Movie Spotlight Feature
# ---------------------------------------------------------
current_movie_info = fetch_enriched_movie_data(st.session_state.selected_movie, OMDB_API_KEY)

with st.container():
    spot_col1, spot_col2 = st.columns([1, 4], gap="large")

    with spot_col1:
        if current_movie_info["poster"] != "N/A":
            st.markdown(f"""
            <div class="poster-box" style="border-radius: 14px; box-shadow: 0 10px 25px rgba(0,0,0,0.6);">
                <img src="{current_movie_info['poster']}" alt="{current_movie_info['title']}" loading="lazy"/>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="poster-box" style="border-radius: 14px; box-shadow: 0 10px 25px rgba(0,0,0,0.6);">
                <div class="poster-placeholder">
                    <div class="icon">🎬</div>
                    <div class="fallback-text">{current_movie_info['title']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with spot_col2:
        rating_badge = f'<span class="badge-rating">⭐ {current_movie_info["rating"]}/10</span>' if current_movie_info["rating"] != "N/A" else ""
        year_badge = f'<span class="badge-year">{current_movie_info["year"]}</span>' if current_movie_info["year"] != "N/A" else ""
        genre_badge = f'<span class="badge-genre">{current_movie_info["genre"]}</span>' if current_movie_info["genre"] != "N/A" else ""
        runtime_badge = f'<span class="badge-runtime">⏱️ {current_movie_info["runtime"]}</span>' if current_movie_info["runtime"] != "N/A" else ""

        director_text = f" • <b>Director:</b> {current_movie_info['director']}" if current_movie_info['director'] != "N/A" else ""
        cast_text = f"<br><b>Cast:</b> {current_movie_info['actors']}" if current_movie_info['actors'] != "N/A" else ""

        st.markdown(f"""
        <div class="spotlight-card">
            <div class="spotlight-badge">✦ Current Feature Spotlight</div>
            <div class="spotlight-title">{current_movie_info['title']}</div>
            <div class="meta-row">
                {rating_badge}
                {year_badge}
                {genre_badge}
                {runtime_badge}
            </div>
            <div class="spotlight-plot">{current_movie_info['plot']}</div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;">
                {director_text}
                {cast_text}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Recommendation Results Section
# ---------------------------------------------------------
if st.session_state.has_searched:
    st.markdown("""
    <div class="section-heading">
        <span>✨ Top Recommended Movies</span>
    </div>
    <div class="section-subtext">
        Ranked by high-dimensional semantic cosine similarity across genre, keywords, and narrative overview.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Finding similar movies..."):
        recommendations = recommend_movies(st.session_state.selected_movie, top_n=st.session_state.top_n)

    if recommendations is None or recommendations.empty:
        st.warning("Sorry, no recommendations found.")
    else:
        st.success(f"Top similar movies:")

        # Concurrently fetch movie details for fast rendering
        titles_to_fetch = [row['title'] for _, row in recommendations.iterrows()]
        with ThreadPoolExecutor(max_workers=min(8, len(titles_to_fetch))) as executor:
            fetched_results = list(executor.map(lambda t: fetch_enriched_movie_data(t, OMDB_API_KEY), titles_to_fetch))
        details_map = {item["title"]: item for item in fetched_results}

        # Layout cards in a responsive grid
        card_count = len(recommendations)
        cols_per_row = 5 if card_count in [5, 10] else min(4, card_count)

        rec_items = list(recommendations.iterrows())

        for row_start in range(0, card_count, cols_per_row):
            row_items = rec_items[row_start : row_start + cols_per_row]
            grid_cols = st.columns(cols_per_row, gap="medium")

            for col_idx, (idx, row) in enumerate(row_items):
                movie_title = row['title']
                item_info = details_map.get(movie_title, fetch_enriched_movie_data(movie_title, OMDB_API_KEY))
                rank_num = row_start + col_idx + 1

                with grid_cols[col_idx]:
                    # Card Container
                    with st.container():
                        # Poster HTML
                        if item_info["poster"] != "N/A":
                            poster_html = f'<img src="{item_info["poster"]}" alt="{movie_title}" loading="lazy"/>'
                        else:
                            poster_html = f'''
                            <div class="poster-placeholder">
                                <div class="icon">🎬</div>
                                <div class="fallback-text">{movie_title}</div>
                            </div>
                            '''

                        rating_badge = f'<span class="badge-rating" style="font-size: 0.72rem; padding: 0.15rem 0.45rem;">⭐ {item_info["rating"]}</span>' if item_info["rating"] != "N/A" else ""
                        year_badge = f'<span class="badge-year" style="font-size: 0.72rem; padding: 0.15rem 0.45rem;">{item_info["year"]}</span>' if item_info["year"] != "N/A" else ""
                        genre_snippet = item_info["genre"].split(",")[0].strip() if item_info["genre"] != "N/A" else ""
                        genre_badge = f'<span class="badge-genre" style="font-size: 0.7rem; padding: 0.15rem 0.45rem;">{genre_snippet}</span>' if genre_snippet else ""

                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="poster-box">
                                <div class="card-rank">#{rank_num} Match</div>
                                {poster_html}
                            </div>
                            <div class="card-info">
                                <div class="card-title" title="{movie_title}">{movie_title}</div>
                                <div class="card-meta">
                                    {rating_badge}
                                    {year_badge}
                                    {genre_badge}
                                </div>
                                <div class="card-snippet">{item_info['plot']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Storyline Expander
                        with st.expander("📖 Storyline & Details"):
                            st.write(f"**Full Synopsis:** {item_info['plot']}")
                            if item_info["genre"] != "N/A":
                                st.write(f"**Genres:** {item_info['genre']}")
                            if item_info["director"] != "N/A":
                                st.write(f"**Director:** {item_info['director']}")
                            if item_info["actors"] != "N/A":
                                st.write(f"**Cast:** {item_info['actors']}")
                            if item_info["runtime"] != "N/A":
                                st.write(f"**Runtime:** {item_info['runtime']}")

                        # Interactive "Explore Similar" Button
                        if st.button("🔍 Explore Similar", key=f"explore_btn_{rank_num}_{movie_title}", use_container_width=True):
                            st.session_state.selected_movie = movie_title
                            st.session_state.has_searched = True
                            st.rerun()

            # Spacing between rows if multiple
            if row_start + cols_per_row < card_count:
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
else:
    # Helpful prompting state when app opens before user clicks recommend
    st.markdown("""
    <div style="text-align: center; padding: 2.5rem 1rem; background: rgba(15, 22, 36, 0.4); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 16px; margin-top: 1rem;">
        <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🎬</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #ffffff; margin-bottom: 0.4rem;">Ready to Discover Similar Films?</div>
        <div style="font-size: 0.95rem; color: #94a3b8; max-width: 550px; margin: 0 auto 1.25rem auto;">
            Review the feature spotlight above, choose your preferred match depth, or click below to compute content similarity vectors.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    _, center_btn_col, _ = st.columns([1, 1, 1])
    with center_btn_col:
        if st.button("🚀 Recommend Similar Movies", type="primary", key="hero_recommend_btn", use_container_width=True):
            st.session_state.has_searched = True
            st.rerun()

# ---------------------------------------------------------
# Modern Minimalist Footer
# ---------------------------------------------------------
st.markdown("""
<div style="text-align: center; margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.06); color: #64748b; font-size: 0.85rem;">
    <span>CineMatch AI • Built with TF-IDF Vectorization, Cosine Semantic Distance & Streamlit</span>
</div>
""", unsafe_allow_html=True)
