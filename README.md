# 📚 BookMind — Book Recommendation System

AI-powered book recommendation system built with **Streamlit**, using real book data from Books.csv.

---

## 📁 Project Structure

```
streamlit_app/
├── app.py               ← Main Streamlit application
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
└── data/
    ├── Books.csv        ← 271,360 real book records (ISBN, Title, Author, Year, Publisher)
    ├── ratings.csv      ← Synthetic user-book ratings
    ├── books_small.csv  ← Small sample books dataset
    └── processed.json   ← Pre-processed data (auto-generated)
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🎯 Features

### Tab 1 — Recommendations
| Mode | Algorithm | Input |
|------|-----------|-------|
| 👤 By User (CF) | SVD Matrix Factorisation | Select a user (dropdown) |
| 📖 By Book (Content) | TF-IDF Cosine Similarity | Type or select a book |
| 🔀 Hybrid | Weighted blend of CF + Content | User + seed book |

- **Genre filter** — narrow results to a specific genre  
- **Adjustable count** — 5 to 20 recommendations  
- **CF weight slider** (Hybrid mode) — balance collaborative vs content signals  

### Tab 2 — Explore Books
- Full-text search by title or author  
- Genre and year-range filters  
- Browse 4,840 real books  

### Tab 3 — Performance Evaluation
- **RMSE** — Root Mean Square Error on 20% held-out test split  
- **MAE** — Mean Absolute Error  
- **Precision@K** — fraction of top-K recs actually liked by the user  
- Side-by-side actual vs predicted ratings sample  

### Tab 4 — Dataset Stats
- Books per genre (bar chart)  
- Rating distribution  
- Publication year distribution  
- Raw books table  

---

## 🧠 Algorithms

### Collaborative Filtering (SVD)
```
User-Item Matrix  →  TruncatedSVD  →  Predicted Ratings
```
- Builds a user × book rating matrix (100 users × 4,840 books)
- Applies TruncatedSVD (20 latent factors) to decompose it
- Reconstructs full matrix to get predicted ratings for unseen books
- Recommends highest predicted-rating books the user hasn't read

### Content-Based Filtering
```
Title + Author + Genre  →  TF-IDF  →  Cosine Similarity
```
- Builds TF-IDF vectors from book metadata (3,000 features)
- For a seed book, finds the most cosine-similar books
- Works without any user history

### Hybrid
```
CF Score × w  +  Content Score × (1-w)  →  Ranked List
```
- Normalises CF and content scores to [0, 1]
- Blends with a configurable weight (default 60% CF, 40% content)

---

## 📊 Dataset

| File | Rows | Columns |
|------|------|---------|
| Books.csv | 271,360 | ISBN, Title, Author, Year, Publisher, Image URLs |
| Ratings (synthetic) | ~1,400 | user_id, isbn, rating (1–5) |
| Users (synthetic) | 100 | user_id, name, preferred_genres |

Genres are inferred from book titles using keyword rules. Ratings are synthetically generated with realistic genre preferences to enable collaborative filtering demonstration.

---

## ⚙️ Configuration

Edit these constants in `app.py` to tune the system:

| Constant | Default | Effect |
|----------|---------|--------|
| `n_components` in `build_svd_model` | 20 | SVD latent factors (more = more accurate, slower) |
| `max_features` in `build_content_model` | 3000 | TF-IDF vocabulary size |
| `top_n` slider | 10 | Number of recommendations shown |
| `cf_weight` slider | 0.6 | Hybrid CF vs content balance |

---

## 🛠 Tech Stack

- **Streamlit** — UI framework  
- **Pandas / NumPy** — Data processing  
- **scikit-learn** — TF-IDF, SVD, cosine similarity  
- **SciPy** — Sparse matrix operations  
