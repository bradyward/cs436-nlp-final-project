# CS436 NLP Final Project — Movie Recommendation System

Hybrid movie recommender combining KNN collaborative filtering, BERT sentiment analysis, and content-based boost filtering.

---

## Setup

### 1. Stay in the project's home directory

```bash
cd cs436-nlp-final-project
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **Mac/Linux:** `source .venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Download the following datasets and place them in the `datasets/` directory with this exact structure:

```
datasets/
├── aclImdb/               # Stanford Large Movie Review Dataset
├── ml-1m/                 # MovieLens 1M Dataset
├── imdb_top_1000.csv      # IMDB Top 1000 Movies (Kaggle)
├── tmdb_5000_credits.csv  # TMDB 5000 Credits (Kaggle)
└── tmdb_5000_movies.csv   # TMDB 5000 Movies (Kaggle)
```

**Dataset sources:**
- `aclImdb/` — https://ai.stanford.edu/~amaas/data/sentiment/
- `ml-1m/` — https://grouplens.org/datasets/movielens/1m/
- `imdb_top_1000.csv` - https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows
- `tmdb_5000_credits.csv`, `tmdb_5000_movies.csv` — https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data

### Merge datasets

After placing all files above, run the merging script to generate `datasets/movies.csv`:

```bash
python MergingScript.py
```

This produces a unified movie database used by the boost filtering module.

---

## Running the Code

**Important:** All scripts must be run from the **project root directory**, not from within subfolders.

---

### Main Pipeline

Runs the full recommendation system end-to-end.

```bash
python pipeline.py
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--knn_n` | 50 | KNN candidates generated before BERT reranking |
| `--knn_top` | 10 | Top KNN results in final output |
| `--boost_top` | 10 | Top boost-filtered results in final output |
| `--no_bert` | off | Skip BERT sentiment scoring |
| `--user_id` | None | Use an existing MovieLens user instead of a synthetic one |

**Examples:**

```bash
# Default: synthetic user, BERT enabled, 50 KNN candidates
python pipeline.py

# Use existing MovieLens user #123
python pipeline.py --user_id 123

# Larger candidate pool, smaller output
python pipeline.py --knn_n 100 --knn_top 5 --boost_top 5

# Skip BERT (faster, no GPU needed)
python pipeline.py --no_bert
```

**Output:** Prints a recommendations table with columns `title`, `genre`, and `source` (KNN or Boost).

---

### Evaluate KNN

Measures RMSE and MAE of the KNN model on a held-out test set.

```bash
python knn/evaluate_knn.py
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--k` | 20 | Number of neighbors |
| `--sample` | 5000 | Test pairs to evaluate |
| `--test_frac` | 0.2 | Fraction of ratings held out for test |

**Example:**

```bash
python knn/evaluate_knn.py --k 20 --sample 5000 --test_frac 0.2
```

---

### Generate Baselines

Computes random and popularity baseline metrics (Precision, Recall, F1, RMSE, MAE) for comparison.

```bash
python generate_baselines.py
```

No arguments required.

---

## Project Structure

```
cs436-nlp-final-project/
├── pipeline.py              # Main entry point
├── MergingScript.py         # Dataset merging utility
├── generate_baselines.py    # Baseline metric generation
├── requirements.txt
├── knn/
│   ├── knn.py               # KNN collaborative filtering model
│   ├── load.py              # Dataset loading utilities
│   └── evaluate_knn.py      # KNN evaluation script
├── transformer/
│   ├── bert.py              # BERT sentiment model (train + inference)
│   └── borah_bert.py        # BERT training on movie metadata (borah cluster)
├── boost/
│   └── boost.py             # Content-based boost filtering
├── datasets/                # Place datasets here (see above)
└── OUTPUTS/                 # Generated recommendation outputs
```

---