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

FAKE_USER_AGE = 28
FAKE_USER_GENDER = "M" 
USER_REVIEWS = [
    {"title": "The Matrix", "rating": 5.0, "review": "Mind blowing sci-fi, completely changed how I see movies."},
    {"title": "The Rock", "rating": 4.0, "review": "High-octane 90s action at its absolute best."},
    {"title": "GoldenEye", "rating": 3.0, "review": "A solid Bond film, but feels a bit formulaic now."}
]


parser = argparse.ArgumentParser(description="Movie recommendation pipeline")
parser.add_argument("--knn_n", type=int, default=50) # KNN candidates before BERT rerank
parser.add_argument("--knn_top", type=int, default=10) # Top KNN movies in final output
parser.add_argument("--boost_top", type=int, default=10) # Top boosted movies in final output
parser.add_argument("--no_bert", action="store_true") # Skip BERT scoring
parser.add_argument("--user_id", type=int, default=None) # Use existing ml-1m user instead of fake user
args = parser.parse_args()

print("Loading ratings and ml-1m movies...")
ratings = load_ratings()
ml1m_movies = load_movies()
users = load_users()

print("Loading BERT model...")
bert_model, tokenizer = load_model(BERT_MODEL_PATH)

if args.user_id is not None:
    target_user_id = args.user_id
    user_ratings = ratings[ratings["user_id"] == target_user_id]
    if user_ratings.empty:
        print(f"Error: user_id {target_user_id} not found in ml-1m dataset.")
        sys.exit(1)
    print(f"Using existing user {target_user_id} with {len(user_ratings)} ratings.")
    user_demo = users[users["user_id"] == target_user_id]
    if not user_demo.empty:
        demo_row = user_demo.iloc[0]
        print(f"User demographics: age={demo_row['age']}, gender={demo_row['gender']}, occupation={demo_row['occupation']}")
else:
    target_user_id = FAKE_USER_ID

    # Score user reviews with BERT
    print("Scoring user reviews with BERT")
    for review in USER_REVIEWS:
        review["bert_score"] = score_text(review["review"], bert_model, tokenizer)
        print(f"  {review['title']:<30} rating={review['rating']:.1f}  bert={review['bert_score']:.4f}")

    # Inject fake user into ratings
    # Resolve title -> movie_id via ml1m_movies
    title_to_id = {}
    for _, movie_row in ml1m_movies.iterrows():
        normalized_title = movie_row["title"].lower()
        normalized_title = __import__("re").sub(r'\s*\(\d{4}\)\s*$', '', normalized_title).strip()
        normalized_title = __import__("re").sub(r'^(.*),\s*(the|a|an)$', r'\2 \1', normalized_title, flags=__import__("re").IGNORECASE).strip()
        title_to_id[normalized_title] = movie_row["movie_id"]

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

print(f"Generating {args.knn_n} KNN candidates for user {target_user_id}")
knn_df = knn_model.recommend(user_id=target_user_id, movies_df=ml1m_movies, n=args.knn_n)

if knn_df.empty:
    print(f"No KNN candidates found for user {target_user_id}")
    sys.exit(1)

knn_movies = knn_df.to_dict(orient="records")

# BERT rerank KNN candidates by overview sentiment
if not args.no_bert:
    print("Scoring KNN candidates with BERT sentiment")
    movies_db = pd.read_csv(MOVIES_CSV)
    db_titles = movies_db['Title']

    for movie in knn_movies:
        db_match = movies_db[db_titles == movie['title']]
        scoring_text = ""
        if not db_match.empty:
            db_row = db_match.iloc[0]
            overview = str(db_row.get("Overview", "")) if pd.notna(db_row.get("Overview")) else ""
            tagline  = str(db_row.get("Tagline",  "")) if pd.notna(db_row.get("Tagline"))  else ""
            scoring_text = (overview + " " + tagline).strip()
        movie['bert_score'] = score_text(scoring_text, bert_model, tokenizer) if scoring_text else 0.5

    knn_movies = sorted(knn_movies, key=lambda movie: movie['bert_score'], reverse=True)

    for movie in knn_movies:
        print(f"  {movie['title']:<50} bert={movie['bert_score']:.4f}")

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