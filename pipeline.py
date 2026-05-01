import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'boost'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'transformer'))
from load import load_ratings, load_movies, load_users
from knn import KNNRecommender
from boost import recommendations as boost_recommendations
from bert import load_model, score_text



MOVIES_CSV = "datasets/movies.csv"
BERT_MODEL_PATH = "transformer/bert_final.pt"
FAKE_USER_ID = 999999

FAKE_USER_OCCUPATION = 0  # 0=other/not specified
# USER 1: Sci-Fi / Action Fan
FAKE_USER_AGE = 28
FAKE_USER_GENDER = "M" 
USER_REVIEWS = [
    {"title": "The Matrix", "rating": 5.0, "review": "Mind blowing sci-fi, completely changed how I see movies."},
    {"title": "The Rock", "rating": 4.0, "review": "High-octane 90s action at its absolute best."},
    {"title": "GoldenEye", "rating": 3.0, "review": "A solid Bond film, but feels a bit formulaic now."}
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie recommendation pipeline")
    parser.add_argument("--knn_n", type=int, default=50) # KNN candidates before BERT rerank
    parser.add_argument("--knn_top", type=int, default=10) # Top KNN movies in final output
    parser.add_argument("--boost_top", type=int, default=10) # Top boosted movies in final output
    parser.add_argument("--no_bert", action="store_true") # Skip BERT scoring
    args = parser.parse_args()

    print("Loading ratings and ml-1m movies...")
    ratings = load_ratings()
    ml1m_movies = load_movies()
    users = load_users()

    # Score user reviews with BERT
    print("Loading BERT model...")
    bert_model, tokenizer = load_model(BERT_MODEL_PATH)

    print("Scoring user reviews with BERT")
    for review in USER_REVIEWS:
        review["bert_score"] = score_text(review["review"], bert_model, tokenizer)
        print(f"  {review['title']:<30} rating={review['rating']:.1f}  bert={review['bert_score']:.4f}")

    # Inject fake user into ratings
    # Resolve title -> movie_id via ml1m_movies
    title_to_id = {}
    for _, row in ml1m_movies.iterrows():
        clean = row["title"].lower()
        clean = __import__("re").sub(r'\s*\(\d{4}\)\s*$', '', clean).strip()
        clean = __import__("re").sub(r'^(.*),\s*(the|a|an)$', r'\2 \1', clean, flags=__import__("re").IGNORECASE).strip()
        title_to_id[clean] = row["movie_id"]

    fake_rows = []
    for review in USER_REVIEWS:
        movie_id = title_to_id.get(review["title"].lower().strip())
        if movie_id is None:
            print(f"Warning: '{review['title']}' not found in ml-1m — skipping injection")
            continue
        # Combined weight: rating scaled by bert confidence
        injected_rating = review["rating"] * review["bert_score"]
        injected_rating = float(np.clip(injected_rating, 1.0, 5.0))
        fake_rows.append({
            "user_id":   FAKE_USER_ID,
            "movie_id":  movie_id,
            "rating":    injected_rating,
            "timestamp": 0,
        })

    if not fake_rows:
        print("No user reviews matched ml-1m movies. Exiting.")
        sys.exit(1)

    ratings = pd.concat([ratings, pd.DataFrame(fake_rows)], ignore_index=True)
    print(f"Injected fake user {FAKE_USER_ID} with {len(fake_rows)} ratings into dataset.")

    # Inject fake user demographics so KNN can use age/gender in similarity
    fake_user_demo = pd.DataFrame([{
        "user_id":    FAKE_USER_ID,
        "gender":     FAKE_USER_GENDER,
        "age":        FAKE_USER_AGE,
        "occupation": FAKE_USER_OCCUPATION,
        "zip":        "00000",
    }])
    users = pd.concat([users, fake_user_demo], ignore_index=True)
    print(f"Fake user demographics: age={FAKE_USER_AGE}, gender={FAKE_USER_GENDER}, occupation={FAKE_USER_OCCUPATION}")

    # KNN
    print(f"Fitting KNN on {len(ratings):,} ratings")
    knn_model = KNNRecommender(k=20)
    knn_model.fit(ratings, users=users)

    print(f"Generating {args.knn_n} KNN candidates for fake user")
    knn_df = knn_model.recommend(user_id=FAKE_USER_ID, movies_df=ml1m_movies, n=args.knn_n)

    if knn_df.empty:
        print("No KNN candidates found for fake user")
        sys.exit(1)

    knn_movies = knn_df.to_dict(orient="records")

    # BERT rerank KNN candidates by overview sentiment
    if not args.no_bert:
        print("Scoring KNN candidates with BERT sentiment")
        movies_db = pd.read_csv(MOVIES_CSV)
        db_titles = movies_db['Title']

        for movie in knn_movies:
            match = movies_db[db_titles == movie['title']]
            text = ""
            if not match.empty:
                row = match.iloc[0]
                overview = str(row.get("Overview", "")) if pd.notna(row.get("Overview")) else ""
                tagline  = str(row.get("Tagline",  "")) if pd.notna(row.get("Tagline"))  else ""
                text = (overview + " " + tagline).strip()
            movie['bert_score'] = score_text(text, bert_model, tokenizer) if text else 0.5

        knn_movies = sorted(knn_movies, key=lambda m: m['bert_score'], reverse=True)

        for m in knn_movies:
            print(f"  {m['title']:<50} bert={m['bert_score']:.4f}")

    # Boost
    print("Loading movies DB and applying boost")
    movies_db = pd.read_csv(MOVIES_CSV)

    final = boost_recommendations(
        knn_movies=knn_movies,
        movies_db=movies_db,
        knn_top=args.knn_top,
        boost_top=args.boost_top,
    )

    print("\n=== Final Recommendations ===")
    print(final.to_string(index=False))
