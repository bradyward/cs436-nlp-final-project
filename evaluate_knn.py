import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from load import load_ratings
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


def evaluate(model: KNNRecommender, test: pd.DataFrame):
    errors = []
    for _, row in tqdm(test.iterrows(), total=len(test), desc="Evaluating", unit="pair"):
        pred = model.predict_rating(int(row["user_id"]), int(row["movie_id"]))
        errors.append(pred - row["rating"])

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    return rmse, mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20, help="Number of neighbors for KNN")
    parser.add_argument("--sample", type=int, default=5000, help="Number of test pairs to sample for evaluation")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Fraction of each user's ratings held out for test")
    args = parser.parse_args()

    print("Loading ratings...", flush=True)
    ratings = load_ratings()
    print(f"  {len(ratings):,} ratings | {ratings['user_id'].nunique():,} users | {ratings['movie_id'].nunique():,} movies")

    print(f"Splitting train/test ({int((1 - args.test_frac) * 100)}/{int(args.test_frac * 100)} temporal)...", flush=True)
    train, test = temporal_split(ratings, test_frac=args.test_frac)
    print(f"  Train: {len(train):,}  |  Test: {len(test):,}")

    print(f"Fitting KNN (k={args.k}, cosine similarity)...", flush=True)
    model = KNNRecommender(k=args.k)
    model.fit(train)

    sample_size = min(args.sample, len(test))
    print(f"Evaluating on test set (sample of {sample_size:,} pairs)...", flush=True)
    sample = test.sample(sample_size, random_state=42)
    rmse, mae = evaluate(model, sample)
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
