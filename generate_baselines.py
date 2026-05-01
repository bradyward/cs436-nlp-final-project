import re, sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knn'))
from load import load_ratings, load_movies

N_USERS = 50
TOP_K = 10
SAMPLE_KNN = 5000
SEED = 42

ratings = load_ratings()
movies_df = load_movies()

# Dataset is 975 pos / 301 neg out of 1276 movies
print(f"BERT always positive prediction: {975/1276}%")

# Temporal split 80/20
ratings = ratings.sort_values(["user_id", "timestamp"])
ratings["_rank"] = ratings.groupby("user_id").cumcount()
ratings["_total"] = ratings.groupby("user_id")["user_id"].transform("count")
train = ratings[ratings["_rank"] < ratings["_total"] * 0.8].drop(columns=["_rank", "_total"]).copy()
test  = ratings[ratings["_rank"] >= ratings["_total"] * 0.8].drop(columns=["_rank", "_total"]).copy()
test  = test[test["movie_id"].isin(set(train["movie_id"]))].copy()

# KNN global mean baseline
global_mean = train["rating"].mean()
knn_sample = test.sample(min(SAMPLE_KNN, len(test)), random_state=SEED)
errs = knn_sample["rating"].values - global_mean
print(f"KNN global mean baseline: RMSE={np.sqrt(np.mean(errs**2)):.4f}  MAE={np.mean(np.abs(errs)):.4f}")

# Title normalization
id_to_title = {}
for _, row in movies_df.iterrows():
    t = re.sub(r'\s*\(\d{4}\)\s*$', '', row["title"])
    t = re.sub(r'^(.*),\s*(the|a|an)$', r'\2 \1', t, flags=re.IGNORECASE)
    id_to_title[row["movie_id"]] = t.lower().strip()

# Top-10 most-rated movies globally (by train count)
popular_ids = train["movie_id"].value_counts().head(TOP_K).index.tolist()
popular_titles = {id_to_title[m] for m in popular_ids if m in id_to_title}

# Sample users with relevant items in test
rng = np.random.default_rng(SEED)
eligible = sorted(set(train["user_id"]) & set(test["user_id"]))
sampled = rng.choice(eligible, size=min(N_USERS, len(eligible)), replace=False)

rand_p, rand_r, rand_f = [], [], []
pop_p,  pop_r,  pop_f  = [], [], []

for uid in sampled:
    user_test = test[test["user_id"] == uid]
    relevant = {id_to_title[m] for m in user_test[user_test["rating"] >= 4]["movie_id"].values if m in id_to_title}
    if not relevant:
        continue

    seen = {id_to_title[m] for m in train[train["user_id"] == uid]["movie_id"].values if m in id_to_title}
    unseen_ids = [m for m in id_to_title if id_to_title[m] not in seen]

    # Random baseline
    rand_pick = set(id_to_title[m] for m in rng.choice(unseen_ids, size=min(TOP_K, len(unseen_ids)), replace=False))
    hits = len(rand_pick & relevant)
    rp = hits / TOP_K
    rr = hits / len(relevant)
    rand_p.append(rp); rand_r.append(rr)
    rand_f.append(2*rp*rr/(rp+rr) if (rp+rr) > 0 else 0.0)

    # Popularity baseline (only top 10 highest rated movies)
    hits = len(popular_titles & relevant)
    pp = hits / TOP_K
    pr = hits / len(relevant)
    pop_p.append(pp); pop_r.append(pr)
    pop_f.append(2*pp*pr/(pp+pr) if (pp+pr) > 0 else 0.0)

print(f"Pipeline random baseline:     P@10={np.mean(rand_p):.4f}  R@10={np.mean(rand_r):.4f}  F1@10={np.mean(rand_f):.4f}")
print(f"Pipeline popularity baseline: P@10={np.mean(pop_p):.4f}  R@10={np.mean(pop_r):.4f}  F1@10={np.mean(pop_f):.4f}")

