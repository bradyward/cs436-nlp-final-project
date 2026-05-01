import argparse
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from load import load_ratings, load_movies, load_users



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

    def fit(self, train: pd.DataFrame, users: pd.DataFrame = None):
        self.train_df = train
        self.global_mean = train["rating"].mean()

        sorted_users = sorted(train["user_id"].unique())
        sorted_movies = sorted(train["movie_id"].unique())
        self.user_index = {u: i for i, u in enumerate(sorted_users)}
        self.movie_index = {m: j for j, m in enumerate(sorted_movies)}

        row_indices, col_indices, rating_values = [], [], []
        for _, row in tqdm(train.iterrows(), total=len(train), desc="Building matrix", unit="rating"):
            user_pos = self.user_index[row["user_id"]]
            movie_pos = self.movie_index[row["movie_id"]]
            row_indices.append(user_pos)
            col_indices.append(movie_pos)
            rating_values.append(row["rating"])

        num_users = len(sorted_users)
        num_movies = len(sorted_movies)
        mat = csr_matrix((rating_values, (row_indices, col_indices)), shape=(num_users, num_movies), dtype=np.float64)

        self.ratings_lookup = {
            (row["user_id"], row["movie_id"]): row["rating"]
            for _, row in train.iterrows()
        }

        # subtract per-user mean from rated entries
        self.user_means = train.groupby("user_id")["rating"].mean().to_dict()
        for user_id, user_index in self.user_index.items():
            start, end = mat.indptr[user_index], mat.indptr[user_index + 1]
            mat.data[start:end] -= self.user_means[user_id]

        if users is not None:
            # Align users to the same order as user_index
            users_aligned = pd.DataFrame({"user_id": sorted_users})
            users_aligned = users_aligned.merge(users, on="user_id", how="left")

            # Encode gender: M=1, F=0
            users_aligned["gender_enc"] = (users_aligned["gender"] == "M").astype(float)

            # Age and occupation are already numeric in MovieLens 1M
            demo_features = users_aligned[["gender_enc", "age", "occupation"]].fillna(0).values

            # Normalize
            scaler = MinMaxScaler()
            demo_scaled = scaler.fit_transform(demo_features)
            demo_scaled *= 0.2

            demo_sparse = csr_matrix(demo_scaled)
            mat = hstack([mat, demo_sparse], format="csr")

            print(f"Matrix shape with demographics: {mat.shape}")
            print(f"  Rating columns    : {num_movies}")
            print(f"  Demographic columns: {demo_sparse.shape[1]}")
        else:
            print(f"Matrix shape (ratings only): {mat.shape}")
            print("No demographics provided — using ratings only")
        
        self.matrix = mat
        self.model.fit(mat)

    def _neighbors(self, user_id: int):
        user_row = self.user_index[user_id]
        distances, indices = self.model.kneighbors(self.matrix[user_row], n_neighbors=self.k + 1)
        neighbors = []
        for neighbor_row, cosine_dist in zip(indices[0], distances[0]):
            if neighbor_row != user_row:
                similarity = 1 - cosine_dist
                neighbors.append((neighbor_row, similarity))
        return neighbors

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user_index or movie_id not in self.movie_index:
            return self.global_mean

        neighbors = self._neighbors(user_id)
        matrix_idx_to_user_id = {v: k for k, v in self.user_index.items()}
        weighted_sum, total_weight = 0.0, 0.0

        for neighbor_idx, similarity in neighbors:
            if similarity <= 0:
                continue
            neighbor_user_id = matrix_idx_to_user_id[neighbor_idx]
            neighbor_rating = self.ratings_lookup.get((neighbor_user_id, movie_id))
            if neighbor_rating is None:
                continue

            deviation = neighbor_rating - self.user_means.get(neighbor_user_id, self.global_mean)
            weighted_sum += similarity * deviation
            total_weight += abs(similarity)

        if total_weight == 0:
            return self.user_means.get(user_id, self.global_mean)

        user_mean = self.user_means.get(user_id, self.global_mean)
        return user_mean + (weighted_sum / total_weight)

    def recommend(self, user_id, movies_df = None, n = 10) -> pd.DataFrame:
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
            # swap genres from pipe separated to comma separated
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
