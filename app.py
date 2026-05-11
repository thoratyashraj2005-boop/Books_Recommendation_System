"""
📚 Book Recommendation System
Collaborative Filtering  ·  Content-Based  ·  Hybrid
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import math
import collections
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📚 Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME TOGGLE (🌙 / ☀️)
# ─────────────────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

with st.container():
    tcol1, tcol2 = st.columns([1, 6])
    with tcol1:
        theme_symbol = "🌙" if st.session_state.theme == "dark" else "☀️"
        # Use a button to switch theme (no rerun issues with session_state)
        if st.button(f"{theme_symbol} Theme", use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS (theme-aware)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.theme == "light":
    st.markdown("""
    <style>
        /* Main header */
        .main-header {
            background: linear-gradient(135deg, #fff 0%, #f5f7ff 50%, #e6f3ff 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        }
        .main-header h1 {
            color: #e94560;
            font-size: 2.8rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -1px;
        }
        .main-header p {
            color: #5b6b9a;
            font-size: 1rem;
            margin: 0.5rem 0 0 0;
        }

        /* Book cards */
        .book-card {
            background: linear-gradient(145deg, #ffffff, #f6f7ff);
            border: 1px solid #e9456030;
            border-radius: 14px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .book-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #e94560, #0f3460);
        }
        .book-title {
            color: #0f172a;
            font-size: 1rem;
            font-weight: 700;
            margin: 0 0 0.3rem 0;
            line-height: 1.3;
        }
        .book-author {
            color: #5b6b9a;
            font-size: 0.85rem;
            margin: 0 0 0.5rem 0;
        }
        .book-meta {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 0.5rem;
        }
        .badge {
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        .badge-genre  { background: #e9456022; color: #e94560; border: 1px solid #e9456040; }
        .badge-year   { background: #0f346022; color: #0f62fe; border: 1px solid #0f62fe30; }
        .badge-score  { background: #16a34a22; color: #16a34a; border: 1px solid #16a34a40; }
        .badge-rating { background: #92400e22; color: #b45309; border: 1px solid #b4530940; }

        /* Stats cards */
        .stat-card {
            background: linear-gradient(145deg, #ffffff, #f6f7ff);
            border: 1px solid #e9456030;
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
        }
        .stat-number {
            font-size: 2rem;
            font-weight: 800;
            color: #e94560;
        }
        .stat-label {
            font-size: 0.8rem;
            color: #5b6b9a;
            margin-top: 4px;
        }

        /* Section headers */
        .section-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #0f172a;
            border-left: 4px solid #e94560;
            padding-left: 12px;
            margin: 1.5rem 0 1rem 0;
        }

        /* Metric highlight */
        .metric-box {
            background: #ffffff;
            border: 1px solid #e9456030;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            text-align: center;
        }

        /* Sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {
            background: #f8fafc !important;
        }

        /* Hide default Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Search box */
        .search-result {
            background: #ffffff;
            border: 1px solid #e9456030;
            border-radius: 10px;
            padding: 0.8rem 1.2rem;
            margin: 0.4rem 0;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        .search-result:hover {
            border-color: #e94560;
        }

        /* Progress ring label */
        .accuracy-label {
            font-size: 2.5rem;
            font-weight: 800;
            color: #e94560;
            text-align: center;
        }
        .stTabs [data-baseweb="tab"] {
            color: #5b6b9a;
        }
        .stTabs [aria-selected="true"] {
            color: #e94560 !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
<style>
    /* Main header */
    .main-header {

        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    .main-header p {
        color: #a8b2d8;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Book cards */
    .book-card {
        background: linear-gradient(145deg, #1e1e3a, #252547);
        border: 1px solid #e9456030;
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .book-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #e94560, #0f3460);
    }
    .book-title {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        line-height: 1.3;
    }
    .book-author {
        color: #a8b2d8;
        font-size: 0.85rem;
        margin: 0 0 0.5rem 0;
    }
    .book-meta {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .badge {
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-genre  { background: #e9456022; color: #e94560; border: 1px solid #e9456040; }
    .badge-year   { background: #0f346022; color: #53c3f3; border: 1px solid #53c3f340; }
    .badge-score  { background: #16a34a22; color: #4ade80; border: 1px solid #4ade8040; }
    .badge-rating { background: #92400e22; color: #fbbf24; border: 1px solid #fbbf2440; }

    /* Stats cards */
    .stat-card {
        background: linear-gradient(145deg, #1e1e3a, #252547);
        border: 1px solid #e9456030;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #e94560;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #a8b2d8;
        margin-top: 4px;
    }

    /* Section headers */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        border-left: 4px solid #e94560;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
    }

    /* Metric highlight */
    .metric-box {
        background: #1e1e3a;
        border: 1px solid #e9456030;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }

    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #0f0f23 !important;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Search box */
    .search-result {
        background: #1e1e3a;
        border: 1px solid #e9456030;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
        cursor: pointer;
        transition: border-color 0.2s;
    }
    .search-result:hover {
        border-color: #e94560;
    }

    /* Progress ring label */
    .accuracy-label {
        font-size: 2.5rem;
        font-weight: 800;
        color: #e94560;
        text-align: center;
    }
    .stTabs [data-baseweb="tab"] {
        color: #a8b2d8;
    }
    .stTabs [aria-selected="true"] {
        color: #e94560 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_path = os.path.join(base_dir, "data", "processed.json")
    with open(processed_path) as f:
        raw = json.load(f)

    books_df = pd.DataFrame(raw["books"])
    books_df["year"] = pd.to_numeric(books_df["year"], errors="coerce").fillna(0).astype(int)

    users_df = pd.DataFrame(raw["users"])

    # Build ratings dataframe
    rows = []
    for uid, rated in raw["ratings"].items():
        for isbn, r in rated.items():
            rows.append({"user_id": int(uid), "isbn": isbn, "rating": r})
    ratings_df = pd.DataFrame(rows)

    genres = raw["genres"]
    return books_df, users_df, ratings_df, genres, raw["ratings"]


@st.cache_data(show_spinner=False)
def build_user_item_matrix(ratings_df, books_df):
    """Build and return user-item pivot matrix."""
    pivot = ratings_df.pivot_table(
        index="user_id", columns="isbn", values="rating", fill_value=0
    )
    return pivot


@st.cache_data(show_spinner=False)
def build_svd_model(ratings_df, books_df, n_components=20):
    """SVD-based collaborative filtering model."""
    pivot = build_user_item_matrix(ratings_df, books_df)
    sparse = csr_matrix(pivot.values.astype(float))
    svd = TruncatedSVD(n_components=min(n_components, min(sparse.shape) - 1), random_state=42)
    U = svd.fit_transform(sparse)
    Sigma = np.diag(svd.singular_values_)
    Vt = svd.components_
    predicted = U @ Sigma @ Vt
    pred_df = pd.DataFrame(predicted, index=pivot.index, columns=pivot.columns)
    return pred_df, pivot, svd


@st.cache_data(show_spinner=False)
def build_content_model(books_df):
    """TF-IDF content-based model on title + author + genre."""
    books_df = books_df.copy()
    books_df["soup"] = (
        books_df["title"].fillna("") + " " +
        books_df["author"].fillna("") + " " +
        books_df["genre"].fillna("")
    )
    tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
    tfidf_matrix = tfidf.fit_transform(books_df["soup"])
    return tfidf_matrix, books_df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def cf_recommend(user_id, ratings_raw, books_df, pred_df, pivot, top_n=10, genre_filter=None):
    """Collaborative filtering via SVD predictions."""
    if user_id not in pred_df.index:
        return pd.DataFrame()
    already_rated = set(pivot.loc[user_id][pivot.loc[user_id] > 0].index)
    scores = pred_df.loc[user_id].drop(labels=list(already_rated), errors="ignore")
    scores = scores.sort_values(ascending=False).head(50)
    rec_isbns = scores.index.tolist()

    result = books_df[books_df["isbn"].isin(rec_isbns)].copy()
    score_map = scores.to_dict()
    result["predicted_score"] = result["isbn"].map(score_map)
    result = result.sort_values("predicted_score", ascending=False)

    if genre_filter and genre_filter != "All":
        result = result[result["genre"] == genre_filter]

    return result.head(top_n)


def content_recommend(isbn_query, books_df, tfidf_matrix, top_n=10, genre_filter=None):
    """Content-based recommendations from a seed book."""
    books_reset = books_df.reset_index(drop=True)
    matches = books_reset[books_reset["isbn"] == isbn_query]
    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]
    query_vec = tfidf_matrix[idx]
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    sim_scores = sorted(enumerate(sims), key=lambda x: -x[1])
    sim_scores = [(i, s) for i, s in sim_scores if i != idx][:50]

    indices = [i for i, _ in sim_scores]
    result = books_reset.iloc[indices].copy()
    result["similarity"] = [s for _, s in sim_scores]

    if genre_filter and genre_filter != "All":
        result = result[result["genre"] == genre_filter]

    return result.head(top_n)


def hybrid_recommend(user_id, isbn_query, ratings_raw, books_df, pred_df, pivot,
                     tfidf_matrix, top_n=10, genre_filter=None, cf_weight=0.6):
    """Blend CF + content-based scores."""
    cf_recs  = cf_recommend(user_id, ratings_raw, books_df, pred_df, pivot, top_n=50)
    cb_recs  = content_recommend(isbn_query, books_df, tfidf_matrix, top_n=50)

    if cf_recs.empty:  return cb_recs.head(top_n)
    if cb_recs.empty:  return cf_recs.head(top_n)

    # Normalize scores 0-1
    cf_recs = cf_recs.copy()
    cb_recs = cb_recs.copy()
    cf_min, cf_max = cf_recs["predicted_score"].min(), cf_recs["predicted_score"].max()
    cb_min, cb_max = cb_recs["similarity"].min(), cb_recs["similarity"].max()
    cf_recs["cf_norm"] = (cf_recs["predicted_score"] - cf_min) / (cf_max - cf_min + 1e-9)
    cb_recs["cb_norm"] = (cb_recs["similarity"] - cb_min) / (cb_max - cb_min + 1e-9)

    cf_dict = dict(zip(cf_recs["isbn"], cf_recs["cf_norm"]))
    cb_dict = dict(zip(cb_recs["isbn"], cb_recs["cb_norm"]))
    all_isbns = set(cf_dict) | set(cb_dict)

    hybrid = {
        i: cf_weight * cf_dict.get(i, 0) + (1 - cf_weight) * cb_dict.get(i, 0)
        for i in all_isbns
    }
    ranked = sorted(hybrid.items(), key=lambda x: -x[1])

    result = books_df[books_df["isbn"].isin([i for i, _ in ranked[:50]])].copy()
    score_map = dict(ranked)
    result["hybrid_score"] = result["isbn"].map(score_map)
    result = result.sort_values("hybrid_score", ascending=False)

    if genre_filter and genre_filter != "All":
        result = result[result["genre"] == genre_filter]

    return result.head(top_n)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(pred_df, pivot, test_frac=0.2):
    """Compute RMSE and MAE on a held-out test split."""
    np.random.seed(42)
    actual, predicted = [], []
    for uid in pivot.index:
        rated_cols = pivot.columns[pivot.loc[uid] > 0]
        if len(rated_cols) < 4:
            continue
        test_cols = np.random.choice(rated_cols, size=max(1, int(len(rated_cols)*test_frac)), replace=False)
        for col in test_cols:
            actual.append(pivot.loc[uid, col])
            predicted.append(pred_df.loc[uid, col] if col in pred_df.columns else 3.0)

    actual = np.array(actual)
    predicted = np.clip(np.array(predicted), 1, 5)
    rmse = math.sqrt(np.mean((actual - predicted)**2))
    mae  = np.mean(np.abs(actual - predicted))
    # Precision@K: predicted >=3.5 and actual >=4
    prec = np.mean((predicted >= 3.5) & (actual >= 4)) / (np.mean(predicted >= 3.5) + 1e-9)
    return round(rmse, 4), round(mae, 4), round(prec, 4)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
GENRE_ICONS = {
    "Fiction": "📖", "Mystery": "🔍", "Romance": "💕", "History": "🏛️",
    "Science": "🔬", "Fantasy": "🧙", "Thriller": "⚡", "Biography": "👤",
    "Self-Help": "🌱", "Family": "👨‍👩‍👧", "Cooking": "🍳", "Business": "💼",
    "Horror": "👻",
}

def star_rating(score, max_score=5):
    filled = int(round(score / max_score * 5))
    return "★" * filled + "☆" * (5 - filled)


def render_book_card(book, score=None, score_label="Match", rank=None):
    genre   = book.get("genre", "Fiction")
    icon    = GENRE_ICONS.get(genre, "📚")
    year    = book.get("year", "")
    title   = book.get("title", "Unknown")
    author  = book.get("author", "Unknown")

    score_badge = ""
    if score is not None:
        pct = min(int(score * 20), 100) if score <= 5 else int(min(score * 100, 100))
        score_badge = f'<span class="badge badge-score">⭐ {score:.2f} {score_label}</span>'

    rank_badge = f'<span class="badge badge-rating">#{rank}</span>' if rank else ""

    st.markdown(f"""
    <div class="book-card">
        <div class="book-title">{rank_badge} {title}</div>
        <div class="book-author">✍️ {author}</div>
        <div class="book-meta">
            <span class="badge badge-genre">{icon} {genre}</span>
            {"<span class='badge badge-year'>📅 " + str(year) + "</span>" if year else ""}
            {score_badge}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls (Minimal) ")
    st.markdown("---")

    mode = st.radio(
        "🎯 Recommendation Mode",
        ["👤 By User (CF)"],
        index=0,
        disabled=True,
    )

    st.markdown("---")
    top_n = st.slider("📊 Number of Recommendations", 5, 20, 10)
    st.markdown("---")

    # About section removed (requested: don't write in sidebar)
    
    


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & BUILD MODELS
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("📚 Loading books and building models…"):
    books_df, users_df, ratings_df, genres, ratings_raw = load_data()
    pred_df, pivot, svd_model   = build_svd_model(ratings_df, books_df)
    tfidf_matrix, books_indexed = build_content_model(books_df)

all_genres = ["All"] + genres
book_titles = sorted(books_df["title"].unique().tolist())
title_to_isbn = dict(zip(books_df["title"], books_df["isbn"]))


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Book Recommendation System</h1>
    <p> </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_rec, tab_explore, tab_eval, tab_data = st.tabs([
    "🎯 Recommendations", "🔍 Explore Books", "📈 Performance", "📊 Dataset Stats"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_rec:

    # ── Mode: By User ──────────────────────────────────────────────────────
    if mode == "👤 By User (CF)":
        st.markdown('<div class="section-title">👤 Collaborative Filtering Recommendations</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            user_ids = sorted(users_df["user_id"].tolist())
            user_labels = [f"User {uid:03d}" for uid in user_ids]
            selected_label = st.selectbox("🔎 Select or type a User", user_labels)
            selected_uid   = int(selected_label.split()[1])

        with col2:
            genre_filter = st.selectbox("🏷️ Filter by Genre", all_genres)

        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("🚀 Show Recommendations", use_container_width=True, type="primary")

        if run:
            # User profile
            user_info  = users_df[users_df["user_id"] == selected_uid].iloc[0]
            user_prefs = user_info["preferred_genres"]
            user_rated = ratings_raw.get(str(selected_uid), {})

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="stat-card"><div class="stat-number">{selected_uid}</div><div class="stat-label">User ID</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="stat-card"><div class="stat-number">{len(user_rated)}</div><div class="stat-label">Books Rated</div></div>', unsafe_allow_html=True)
            pref_str = " · ".join(GENRE_ICONS.get(g,"")+" "+g for g in user_prefs[:3])
            c3.markdown(f'<div class="stat-card"><div class="stat-number" style="font-size:1rem;padding-top:8px">{pref_str}</div><div class="stat-label">Preferred Genres</div></div>', unsafe_allow_html=True)
            avg_r = round(np.mean(list(user_rated.values())), 2) if user_rated else 0
            c4.markdown(f'<div class="stat-card"><div class="stat-number">{avg_r}</div><div class="stat-label">Avg Rating Given</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-title">📚 Previously Read (sample)</div>', unsafe_allow_html=True)
            sample_isbns = list(user_rated.keys())[:5]
            sample_books = books_df[books_df["isbn"].isin(sample_isbns)]
            cols = st.columns(min(5, len(sample_books)))
            for i, (_, b) in enumerate(sample_books.iterrows()):
                with cols[i]:
                    r = user_rated.get(b["isbn"], 0)
                    st.markdown(f"""
                    <div class="book-card" style="min-height:120px">
                        <div class="book-title" style="font-size:0.85rem">{b['title'][:45]}…</div>
                        <div class="book-author" style="font-size:0.75rem">{b['author']}</div>
                        <div class="book-meta">
                            <span class="badge badge-rating">{"★"*r}{"☆"*(5-r)}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-title">🎯 Recommended for You</div>', unsafe_allow_html=True)

            with st.spinner("Generating recommendations…"):
                recs = cf_recommend(selected_uid, ratings_raw, books_df, pred_df, pivot,
                                    top_n=top_n, genre_filter=genre_filter)

            if recs.empty:
                st.warning("No recommendations found. Try removing the genre filter.")
            else:
                cols = st.columns(2)
                for i, (_, book) in enumerate(recs.iterrows()):
                    with cols[i % 2]:
                        render_book_card(book, score=book.get("predicted_score"),
                                         score_label="Score", rank=i+1)

    # ── Mode: By Book ──────────────────────────────────────────────────────
    elif mode == "📖 By Book (Content)":
        st.markdown('<div class="section-title">📖 Content-Based Recommendations</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            typed = st.text_input("🔎 Type book title (or choose below)", placeholder="e.g. Classical Mythology")
            matched = [t for t in book_titles if typed.lower() in t.lower()] if typed else book_titles
            selected_title = st.selectbox("📖 Select Book", matched[:200])

        with col2:
            genre_filter = st.selectbox("🏷️ Filter by Genre", all_genres)

        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("🚀 Show Recommendations", use_container_width=True, type="primary")

        if run:
            seed_isbn = title_to_isbn.get(selected_title)
            seed_info = books_df[books_df["isbn"] == seed_isbn]

            if not seed_info.empty:
                seed = seed_info.iloc[0]
                st.markdown(f"""
                <div class="book-card" style="border-color:#e9456080; background:linear-gradient(145deg,#2a1a2e,#2e1a3a)">
                    <div class="book-title" style="font-size:1.1rem">🌟 Seed Book: {seed['title']}</div>
                    <div class="book-author">✍️ {seed['author']}</div>
                    <div class="book-meta">
                        <span class="badge badge-genre">{GENRE_ICONS.get(seed['genre'],'📚')} {seed['genre']}</span>
                        <span class="badge badge-year">📅 {seed['year']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-title">📚 Similar Books</div>', unsafe_allow_html=True)

            with st.spinner("Finding similar books…"):
                recs = content_recommend(seed_isbn, books_indexed, tfidf_matrix,
                                         top_n=top_n, genre_filter=genre_filter)

            if recs.empty:
                st.warning("No recommendations found.")
            else:
                cols = st.columns(2)
                for i, (_, book) in enumerate(recs.iterrows()):
                    with cols[i % 2]:
                        render_book_card(book, score=book.get("similarity"),
                                         score_label="Similarity", rank=i+1)

    # ── Mode: Hybrid ───────────────────────────────────────────────────────
    else:
        # Hybrid disabled; keep only Collaborative Filtering
        pass

        col1, col2 = st.columns(2)
        with col1:
            user_ids    = sorted(users_df["user_id"].tolist())
            user_labels = [f"User {uid:03d}" for uid in user_ids]
            sel_label   = st.selectbox("👤 Select User", user_labels)
            selected_uid = int(sel_label.split()[1])

        with col2:
            typed = st.text_input("📖 Type a book you like", placeholder="e.g. Thriller")
            matched = [t for t in book_titles if typed.lower() in t.lower()] if typed else book_titles
            selected_title = st.selectbox("📖 Select Book", matched[:200])

        col3, col4, col5 = st.columns([2, 2, 1])
        with col3:
            genre_filter = st.selectbox("🏷️ Filter by Genre", all_genres)
        with col4:
            cf_weight = st.slider("⚖️ CF Weight (vs Content)", 0.1, 0.9, 0.6, 0.1,
                                   help="Higher = more collaborative filtering")
        with col5:
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("🚀 Show Recommendations", use_container_width=True, type="primary")

        if run:
            seed_isbn = title_to_isbn.get(selected_title)
            with st.spinner("Blending models…"):
                recs = hybrid_recommend(
                    selected_uid, seed_isbn, ratings_raw, books_df,
                    pred_df, pivot, tfidf_matrix,
                    top_n=top_n, genre_filter=genre_filter, cf_weight=cf_weight
                )

            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="stat-card"><div class="stat-number">{int(cf_weight*100)}%</div><div class="stat-label">CF Weight</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="stat-card"><div class="stat-number">{int((1-cf_weight)*100)}%</div><div class="stat-label">Content Weight</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="stat-card"><div class="stat-number">{len(recs)}</div><div class="stat-label">Books Found</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-title">🎯 Hybrid Recommendations</div>', unsafe_allow_html=True)

            if recs.empty:
                st.warning("No recommendations found.")
            else:
                cols = st.columns(2)
                for i, (_, book) in enumerate(recs.iterrows()):
                    with cols[i % 2]:
                        render_book_card(book, score=book.get("hybrid_score"),
                                         score_label="Hybrid", rank=i+1)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE BOOKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown('<div class="section-title">🔍 Explore the Book Catalog</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        search_q = st.text_input("🔎 Search by title or author", placeholder="Type to filter…")
    with c2:
        genre_exp = st.selectbox("🏷️ Genre", all_genres, key="exp_genre")
    with c3:
        year_range = st.slider("📅 Year Range", 1800, 2010, (1990, 2010))

    filtered = books_df.copy()
    if search_q:
        mask = (
            filtered["title"].str.contains(search_q, case=False, na=False) |
            filtered["author"].str.contains(search_q, case=False, na=False)
        )
        filtered = filtered[mask]
    if genre_exp != "All":
        filtered = filtered[filtered["genre"] == genre_exp]
    filtered = filtered[
        (filtered["year"] >= year_range[0]) | (filtered["year"] == 0)
    ]
    filtered = filtered[
        (filtered["year"] <= year_range[1]) | (filtered["year"] == 0)
    ]

    st.markdown(f"**{len(filtered):,} books** match your filters")
    st.markdown("---")

    display = filtered.head(50)
    cols = st.columns(2)
    for i, (_, book) in enumerate(display.iterrows()):
        with cols[i % 2]:
            render_book_card(book)

    if len(filtered) > 50:
        st.info(f"Showing 50 of {len(filtered):,} results. Refine your search to narrow down.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PERFORMANCE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<div class="section-title">📈 Model Performance Evaluation</div>', unsafe_allow_html=True)
    st.markdown("Metrics computed on a **20% held-out test split** from the rating matrix.")

    if st.button("⚙️ Run Evaluation", type="primary"):
        with st.spinner("Evaluating model…"):
            rmse, mae, prec = compute_metrics(pred_df, pivot)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="stat-card"><div class="stat-number">{rmse}</div><div class="stat-label">RMSE (lower is better)</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><div class="stat-number">{mae}</div><div class="stat-label">MAE (lower is better)</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card"><div class="stat-number">{prec:.1%}</div><div class="stat-label">Precision@K</div></div>', unsafe_allow_html=True)
        accuracy = max(0, min(100, round((1 - rmse/4) * 100)))
        c4.markdown(f'<div class="stat-card"><div class="stat-number">{accuracy}%</div><div class="stat-label">Est. Accuracy</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">📊 Metric Interpretation</div>', unsafe_allow_html=True)

        interp_data = {
            "Metric": ["RMSE", "MAE", "Precision@K", "Est. Accuracy"],
            "Value":  [str(rmse), str(mae), f"{prec:.1%}", f"{accuracy}%"],
            "Meaning": [
                "Root Mean Square Error between predicted and actual ratings (scale 1–5)",
                "Mean Absolute Error — average prediction deviation",
                "Fraction of top-K recommendations that the user actually likes (rating ≥ 4)",
                "Derived from RMSE: (1 − RMSE/4) × 100%",
            ],
            "Benchmark": ["< 1.0 is good", "< 0.8 is good", "> 60% is good", "> 75% is good"],
        }
        st.dataframe(pd.DataFrame(interp_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<div class="section-title">🔍 Prediction Sample</div>', unsafe_allow_html=True)
        sample_uid = pivot.index[0]
        rated_isbns = list(pivot.columns[pivot.loc[sample_uid] > 0])[:8]
        sample_rows = []
        for isbn in rated_isbns:
            actual = pivot.loc[sample_uid, isbn]
            pred   = pred_df.loc[sample_uid, isbn] if isbn in pred_df.columns else 3.0
            title  = books_df[books_df["isbn"]==isbn]["title"].values
            title  = title[0][:40] if len(title) > 0 else isbn
            sample_rows.append({
                "Book": title,
                "Actual Rating": f"{'★'*int(actual)}{'☆'*(5-int(actual))} ({actual})",
                "Predicted":     f"{pred:.2f}",
                "Error":         f"{abs(actual-pred):.2f}",
            })
        st.dataframe(pd.DataFrame(sample_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Click **Run Evaluation** to compute RMSE, MAE and Precision@K on the test split.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATASET STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="stat-card"><div class="stat-number">{len(books_df):,}</div><div class="stat-label">Total Books</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card"><div class="stat-number">{len(users_df):,}</div><div class="stat-label">Users</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-card"><div class="stat-number">{len(ratings_df):,}</div><div class="stat-label">Ratings</div></div>', unsafe_allow_html=True)
    avg_r = round(ratings_df["rating"].mean(), 2)
    c4.markdown(f'<div class="stat-card"><div class="stat-number">{avg_r}</div><div class="stat-label">Avg Rating</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">📚 Books per Genre</div>', unsafe_allow_html=True)
        genre_counts = books_df["genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]
        genre_counts["Icon"] = genre_counts["Genre"].map(GENRE_ICONS)
        genre_counts["Genre"] = genre_counts["Icon"] + " " + genre_counts["Genre"]
        st.bar_chart(genre_counts.set_index("Genre")["Count"])

    with col_b:
        st.markdown('<div class="section-title">⭐ Rating Distribution</div>', unsafe_allow_html=True)
        rating_counts = ratings_df["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Rating", "Count"]
        rating_counts["Rating"] = rating_counts["Rating"].astype(str) + " ★"
        st.bar_chart(rating_counts.set_index("Rating")["Count"])

    st.markdown("---")
    st.markdown('<div class="section-title">📅 Publication Year Distribution</div>', unsafe_allow_html=True)
    year_df = books_df[(books_df["year"] >= 1950) & (books_df["year"] <= 2010)]
    year_counts = year_df["year"].value_counts().sort_index()
    st.area_chart(year_counts)

    st.markdown("---")
    st.markdown('<div class="section-title">📋 Raw Books Sample</div>', unsafe_allow_html=True)
    st.dataframe(
        books_df[["isbn","title","author","year","genre"]].head(100),
        use_container_width=True, hide_index=True
    )
