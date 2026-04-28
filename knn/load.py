import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "ml-1m"


def load_ratings() -> pd.DataFrame:
    path = DATA_DIR / "ratings.dat"
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    df["rating"] = df["rating"].astype(float)
    return df


def load_users() -> pd.DataFrame:
    path = DATA_DIR / "users.dat"
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1",
    )


def load_movies() -> pd.DataFrame:
    path = DATA_DIR / "movies.dat"
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )


def load_all() -> tuple:
    return load_ratings(), load_users(), load_movies()
