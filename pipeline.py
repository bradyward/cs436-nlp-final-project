import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'boost'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'transformer'))

from load import load_ratings, load_movies
from boost import recommendations as boost_recommendations
from bert import cosine_similarity

MOVIES_CSV = "datasets/movies_merged.csv"
BERT_MODEL_PATH = "transformer/bert_final.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie recommendation pipeline")
    parser.add_argument("--movie", type=str, required=True, help="Input movie title (partial match ok)")
    parser.add_argument("--knn_n", type=int, default=50, help="Fan-collaborative candidates fed into BERT + boost")
    parser.add_argument("--boost_top", type=int, default=10, help="Top boosted movies in final output")
    parser.add_argument("--no_bert", action="store_true", help="Skip BERT scoring")
    args = parser.parse_args()

    print("Loading ratings and ml-1m movies...")
    ratings = load_ratings()
    ml1m_movies = load_movies()

    # Match input title to ml-1m movie_id
    title_query = args.movie.lower()
    ml1m_movies['title_clean'] = ml1m_movies['title'].str.lower().str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    matches = ml1m_movies[ml1m_movies['title_clean'].str.contains(title_query, regex=False)]

    if matches.empty:
        print(f"No ml-1m movie found matching: '{args.movie}'")
        sys.exit(1)

    input_movie = matches.iloc[0]
    input_movie_id = input_movie['movie_id']
    print(f"Matched: {input_movie['title']} (id={input_movie_id})")

    # Find fans: users who rated input movie highly.
    # Auto-drop threshold from 4 down to 1 until fans found.
    fans = pd.DataFrame()
    for threshold in [4.0, 3.0, 2.0, 1.0]:
        fans = ratings[(ratings['movie_id'] == input_movie_id) & (ratings['rating'] >= threshold)]
        if not fans.empty:
            print(f"Found {len(fans)} fans at rating >= {threshold}")
            break

    if fans.empty:
        print(f"No ratings found for '{input_movie['title']}'")
        sys.exit(1)

    fan_ids = fans['user_id'].unique()

    # Collect other movies those fans rated, exclude input movie
    fan_ratings = ratings[
        (ratings['user_id'].isin(fan_ids)) &
        (ratings['movie_id'] != input_movie_id)
    ]

    # Aggregate: score = mean_rating * log(count+1) to balance quality vs popularity
    agg = fan_ratings.groupby('movie_id').agg(
        mean_rating=('rating', 'mean'),
        count=('rating', 'count')
    ).reset_index()
    agg['score'] = agg['mean_rating'] * np.log1p(agg['count'])
    agg = agg.sort_values('score', ascending=False).head(args.knn_n)

    # Merge with ml-1m titles/genres, normalize to match movies_merged.csv
    agg = agg.merge(ml1m_movies[['movie_id', 'title', 'genres']], on='movie_id', how='left')
    agg['title'] = agg['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    agg['title'] = agg['title'].str.replace(r'^(.*),\s*(the|a|an)$', r'\2 \1', regex=True, flags=__import__('re').IGNORECASE)
    agg['title'] = agg['title'].str.lower().str.strip()
    agg['genre'] = agg['genres'].apply(lambda g: g.replace('|', ', ').lower() if pd.notna(g) else "")
    agg['rating'] = agg['mean_rating'].clip(1.0, 5.0).round(4)

    knn_movies = agg[['movie_id', 'title', 'genre', 'rating']].to_dict(orient='records')

    if not args.no_bert:
        print("Loading MiniLM model...")
        from sentence_transformers import SentenceTransformer
        minilm = SentenceTransformer('all-MiniLM-L6-v2')

        print("Scoring candidates with MiniLM semantic similarity...")
        movies_db = pd.read_csv(MOVIES_CSV)
        db_titles = movies_db['Title']

        def get_text(title):
            match = movies_db[db_titles == title]
            if match.empty:
                return ""
            row = match.iloc[0]
            parts = []
            for col in ["Overview", "Tagline", "Genre", "Director", "Actors"]:
                val = row.get(col)
                if val is not None and pd.notna(val) and str(val).strip():
                    parts.append(str(val).strip())
            return " ".join(parts)

        input_title = input_movie['title'].replace(r'\s*\(\d{4}\)\s*$', '').lower().strip()
        input_text = get_text(input_title) or input_title
        input_embedding = minilm.encode(input_text)

        for movie in knn_movies:
            text = get_text(movie['title'])
            if text:
                candidate_embedding = minilm.encode(text)
                movie['bert_score'] = cosine_similarity(input_embedding, candidate_embedding)
            else:
                movie['bert_score'] = 0.0

        knn_movies = sorted(knn_movies, key=lambda m: m['bert_score'], reverse=True)

        for m in knn_movies:
            print(f"  {m['title']:<50} sim={m['bert_score']:.4f}")

    print("Loading movies DB and applying boost...")
    movies_db = pd.read_csv(MOVIES_CSV)

    final = boost_recommendations(
        knn_movies=knn_movies,
        movies_db=movies_db,
        knn_top=args.knn_n,
        boost_top=args.boost_top,
    )

    print("\n=== Final Recommendations ===")
    print(final.to_string(index=False))