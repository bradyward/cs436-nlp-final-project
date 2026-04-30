import argparse
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from load import load_ratings, load_movies



class KNNRecommender:
    def __init__(self, k: int = 20):
        self.k = k
        self.model = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
        self.user_index: dict = {}
        self.movie_index: dict = {}
        self.user_means: dict = {}
        self.matrix: np.ndarray = None
        self.global_mean: float = 0.0
        self.train_df: pd.DataFrame = None
        self.ratings_lookup: dict = {}

    def fit(self, train: pd.DataFrame):
        self.train_df = train
        self.global_mean = train["rating"].mean()

        users = sorted(train["user_id"].unique())
        movies = sorted(train["movie_id"].unique())
        self.user_index = {u: i for i, u in enumerate(users)}
        self.movie_index = {m: j for j, m in enumerate(movies)}

        rows, cols, data = [], [], []
        for _, row in tqdm(train.iterrows(), total=len(train), desc="Building matrix", unit="rating"):
            u = self.user_index[row["user_id"]]
            m = self.movie_index[row["movie_id"]]
            rows.append(u)
            cols.append(m)
            data.append(row["rating"])

        num_users = len(users)
        num_movies = len(movies)
        mat = csr_matrix((data, (rows, cols)), shape=(num_users, num_movies), dtype=np.float32)

        self.ratings_lookup = {
            (row["user_id"], row["movie_id"]): row["rating"]
            for _, row in train.iterrows()
        }

        # subtract per-user mean from rated entries
        self.user_means = train.groupby("user_id")["rating"].mean().to_dict()
        mat = mat.astype(np.float64)
        for user_id, user_index in self.user_index.items():
            start, end = mat.indptr[user_index], mat.indptr[user_index + 1]
            mat.data[start:end] -= self.user_means[user_id]

        self.matrix = mat
        self.model.fit(mat)

    def _neighbors(self, user_id: int):
        user_index = self.user_index[user_id]
        distances, indices = self.model.kneighbors(self.matrix[user_index], n_neighbors=self.k + 1)
        # Cosine distance to similarity
        neighbors = []
        for index, dist in zip(indices[0], distances[0]):
            if index != user_index:
                similarity = 1 - dist
                neighbors.append((index, similarity))
        return neighbors

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user_index or movie_id not in self.movie_index:
            return self.global_mean

        movie_index = self.movie_index[movie_id]
        neighbors = self._neighbors(user_id)

        numer, d = 0.0, 0.0
        idx_to_uid = {v: k for k, v in self.user_index.items()}
        for n_idx, sim in neighbors:
            if sim <= 0:
                continue
            n_uid = idx_to_uid[n_idx]
            raw_rating = self.ratings_lookup.get((n_uid, movie_id))
            if raw_rating is None:
                continue

            deviation = raw_rating - self.user_means.get(n_uid, self.global_mean)
            numer += sim * deviation
            d += abs(sim)

        if d == 0:
            return self.user_means.get(user_id, self.global_mean)

        user_mean = self.user_means.get(user_id, self.global_mean)
        return user_mean + (numer / d)

    def recommend(self, user_id: int, movies_df: pd.DataFrame = None, n: int = 10) -> pd.DataFrame:
        if user_id not in self.user_index:
            return pd.DataFrame(columns=["movie_id", "predicted_rating", "title", "genres"])

        rated_movies = set(self.train_df[self.train_df["user_id"] == user_id]["movie_id"].values)
        candidates = [movie for movie in self.movie_index if movie not in rated_movies]

        predictions = [
            (movie, self.predict_rating(user_id, movie)) 
            for movie in tqdm(candidates, desc=f"Scoring candidates for user {user_id}", unit="movie")
            ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        top_n = predictions[:n]
        result = pd.DataFrame(top_n, columns=["movie_id", "rating"])
        if movies_df is not None:
            result = result.merge(movies_df[["movie_id", "title", "genres"]], on="movie_id", how="left")
            # normalize title: strip year suffix "(YYYY)" and lowercase to match movies.csv
            result["title"] = result["title"].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
            result["title"] = result["title"].str.replace(r'^(.*),\s*(the|a|an)$', r'\2 \1', regex=True, flags=__import__('re').IGNORECASE)
            result["title"] = result["title"].str.lower().str.strip()
            # genres: pipe-separated string -> comma-separated string
            result["genre"] = result["genres"].apply(
                lambda g: ", ".join(g.split("|")).lower() if pd.notna(g) else ""
            )
            result = result.drop(columns=["genres"])
            result["rating"] = result["rating"].clip(1.0, 5.0).round(4)
        return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=int, required=True) # User ID to recommend for
    parser.add_argument("--n", type=int, default=10) # Number of recommendations to return
    args = parser.parse_args()

    print("Loading ratings...")
    ratings = load_ratings()
    movies = load_movies()

    print(f"Fitting KNN (k=20, cosine similarity) on all {len(ratings):,} ratings...")
    model = KNNRecommender(k=20)
    model.fit(ratings)

    recs = model.recommend(user_id=args.user_id, movies_df=movies, n=args.n)

    results = [
        {
            "movie_id": int(row["movie_id"]),
            "title": row["title"],
            "genre": row["genre"],
            "rating": float(row["rating"]),
        } for _, row in recs.iterrows()
    ]

    print(json.dumps(results))
