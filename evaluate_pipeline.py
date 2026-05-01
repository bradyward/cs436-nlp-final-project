import argparse
import re
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knn'))
from load import load_ratings, load_movies
from knn import KNNRecommender


def temporal_split(ratings: pd.DataFrame, test_frac: float = 0.2):
    ratings = ratings.sort_values(["user_id", "timestamp"])
    ratings["_rank"] = ratings.groupby("user_id").cumcount()
    ratings["_total"] = ratings.groupby("user_id")["user_id"].transform("count")
    cutoff = ratings["_total"] * (1 - test_frac)
    train = ratings[ratings["_rank"] < cutoff].drop(columns=["_rank", "_total"]).copy()
    test = ratings[ratings["_rank"] >= cutoff].drop(columns=["_rank", "_total"]).copy()
    known_movies = set(train["movie_id"])
    test = test[test["movie_id"].isin(known_movies)]
    return train, test


def normalize_title(title: str) -> str:
    t = re.sub(r'\s*\(\d{4}\)\s*$', '', title)
    t = re.sub(r'^(.*),\s*(the|a|an)$', r'\2 \1', t, flags=re.IGNORECASE)
    return t.lower().strip()


parser = argparse.ArgumentParser(description="End-to-end Precision/Recall/F1 evaluation")
parser.add_argument("--n_users", type=int, default=50)
parser.add_argument("--k", type=int, default=20)
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--test_frac", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

print("Loading data...")
ratings = load_ratings()
movies_df = load_movies()
print(f"  {len(ratings):,} ratings | {ratings['user_id'].nunique():,} users")

print(f"Temporal split ({int((1-args.test_frac)*100)}/{int(args.test_frac*100)})...")
train, test = temporal_split(ratings, test_frac=args.test_frac)
print(f"  Train: {len(train):,} | Test: {len(test):,}")

print(f"Fitting KNN (k={args.k})...")
model = KNNRecommender(k=args.k)
model.fit(train)

# Build movie_id to normalized title map
id_to_title = {
    row["movie_id"]: normalize_title(row["title"])
    for _, row in movies_df.iterrows()
}

# Sample users that exist in both train and test
train_users = set(train["user_id"].unique())
test_users = set(test["user_id"].unique())
eligible = sorted(train_users & test_users)

rng = np.random.default_rng(args.seed)
sampled = rng.choice(eligible, size=min(args.n_users, len(eligible)), replace=False)

precisions, recalls, f1s = [], [], []
skipped = 0

for i, uid in enumerate(sampled):
    user_test = test[test["user_id"] == uid]
    relevant_ids = set(user_test[user_test["rating"] >= 4]["movie_id"].values)
    relevant_titles = {id_to_title[m] for m in relevant_ids if m in id_to_title}

    if not relevant_titles:
        skipped += 1
        continue

    recs = model.recommend(user_id=uid, movies_df=movies_df, n=args.top_k)
    if recs.empty:
        skipped += 1
        continue

    rec_titles = set(recs["title"].values)
    hits = len(rec_titles & relevant_titles)

    p = hits / args.top_k
    r = hits / len(relevant_titles)
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    precisions.append(p)
    recalls.append(r)
    f1s.append(f)

evaluated = len(precisions)
print(f"\nUsers evaluated : {evaluated} / {len(sampled)} ({skipped} skipped — no relevant items in test set)")
print(f"Precision@{args.top_k}    : {np.mean(precisions):.4f} +/- {np.std(precisions):.4f}")
print(f"Recall@{args.top_k}       : {np.mean(recalls):.4f} +/- {np.std(recalls):.4f}")
print(f"F1@{args.top_k}           : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
