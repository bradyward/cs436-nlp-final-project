import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'boost'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'transformer'))

from load import load_ratings, load_movies
from knn import KNNRecommender
from boost import recommendations as boost_recommendations
from bert import load_model, score_text

MOVIES_CSV = "datasets/movies.csv"
BERT_MODEL_PATH = "transformer/bert_final.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie recommendation pipeline")
    parser.add_argument("--user_id", type=int, required=True, help="ML-1M user ID")
    parser.add_argument("--knn_n", type=int, default=50, help="KNN candidates to generate before BERT rerank")
    parser.add_argument("--knn_top", type=int, default=10, help="Top KNN movies in final output")
    parser.add_argument("--boost_top", type=int, default=10, help="Top boosted movies in final output")
    parser.add_argument("--no_bert", action="store_true", help="Skip BERT scoring")
    args = parser.parse_args()

    print("Loading ratings and ml-1m movies...")
    ratings = load_ratings()
    ml1m_movies = load_movies()

    print(f"Fitting KNN on {len(ratings):,} ratings...")
    knn_model = KNNRecommender(k=20)
    knn_model.fit(ratings)

    print(f"Generating {args.knn_n} KNN candidates for user {args.user_id}...")
    knn_df = knn_model.recommend(user_id=args.user_id, movies_df=ml1m_movies, n=args.knn_n)

    if knn_df.empty:
        print(f"No candidates found for user {args.user_id}.")
        sys.exit(1)

    knn_movies = knn_df.to_dict(orient="records")

    if not args.no_bert:
        print("Loading BERT model...")
        bert_model, tokenizer = load_model(BERT_MODEL_PATH)

        print("Scoring candidates with BERT sentiment...")
        movies_db = pd.read_csv(MOVIES_CSV)
        db_titles = movies_db['Title']

        for movie in knn_movies:
            match = movies_db[db_titles == movie['title']]
            text = ""
            if not match.empty:
                row = match.iloc[0]
                overview = str(row.get("Overview", "")) if pd.notna(row.get("Overview")) else ""
                tagline = str(row.get("Tagline", "")) if pd.notna(row.get("Tagline")) else ""
                text = (overview + " " + tagline).strip()
            movie['bert_score'] = score_text(text, bert_model, tokenizer) if text else 0.5

        knn_movies = sorted(knn_movies, key=lambda m: m['bert_score'], reverse=True)

        # Debug bert scores
        for m in knn_movies:
            print(f"  {m['title']:<50} bert={m['bert_score']:.4f}")


    print("Loading movies DB and applying boost...")
    movies_db = pd.read_csv(MOVIES_CSV)

    final = boost_recommendations(
        knn_movies=knn_movies,
        movies_db=movies_db,
        knn_top=args.knn_top,
        boost_top=args.boost_top,
    )

    print("\n=== Final Recommendations ===")
    print(final.to_string(index=False))