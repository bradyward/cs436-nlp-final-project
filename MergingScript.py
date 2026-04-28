import pandas as pd
from collections import Counter
import re
import ast

movies5000unclean = pd.read_csv("tmdb_5000_movies.csv")
movies1000unclean = pd.read_csv("imdb_top_1000.csv")
movies5000credits = pd.read_csv("tmdb_5000_credits.csv")

movies5000 = movies5000unclean.copy()
movies1000 = movies1000unclean.copy()

# Get Director and the first 4 stars for movies5000credits
marker = '"job": "Director", "name":'
pattern = rf'{re.escape(marker)}\s*"([^"]+)"'
movies5000credits['crew'] = (
movies5000credits['crew'].apply(
    lambda x: ", ".join(re.findall(pattern, x)) if isinstance(x, str) else None))

# Get the first 4 stars from the list of stars
marker = '"name":'
pattern = rf'{re.escape(marker)}\s*"([^"]+)"'
movies5000credits['cast'] = (
movies5000credits['cast'].apply(
    lambda x: ", ".join(re.findall(pattern, x)[:4]) if isinstance(x, str) else None))

# Now merge the movies5000credits to movies5000
movies5000 = pd.merge(movies5000, movies5000credits, on="title")

# Merge all the stars from movies1000 together
movies1000['Actors'] = (
    movies1000[['Star1','Star2','Star3','Star4']].astype(str).agg(", ".join, axis=1)
)

#Clean keywords on movies5000
marker = '"name":'
pattern = rf'{re.escape(marker)}\s*"([^"]+)"'
movies5000["keywords"] = (
movies5000["keywords"].apply(
    lambda x: ", ".join(re.findall(pattern, x)) if isinstance(x, str) else None))

# Drop non-needed columns
movies5000 = movies5000.drop(['budget', 'homepage', 'original_language', 'production_companies', 'production_countries', 'runtime', 'spoken_languages', 'status', 'movie_id', 'id', 'revenue', 
                             'vote_count', 'popularity'], axis=1)
movies1000 = movies1000.drop(['Poster_Link', 'Certificate', 'Runtime', 'Star1','Star2','Star3','Star4', 'Gross', 'Meta_score', 'No_of_Votes'], axis=1)

# Rename columns
movies5000.rename(columns={'genres': 'Genre', 'overview': 'Overview', 'cast': 'Actors', 'crew': 'Director', 'vote_average': 'Rating', 'keywords': 'Keywords', 'tagline': 'Tagline', 'title': 'Title', 'release_date': 'Year'}, inplace = True)
movies1000.rename(columns={'Series_Title': 'Title', 'IMDB_Rating': 'Rating', 'Released_Year': 'Year'}, inplace = True)

# Clean the genre section
marker = '"name":'
pattern = rf'{re.escape(marker)}\s*"([^"]+)"'
movies5000["Genre"] = (
movies5000["Genre"].apply(
    lambda x: ", ".join(re.findall(pattern, x)) if isinstance(x, str) else None))

#For the sake of the future merging, lowercase all the info in the db
def lower_all(v):
    if isinstance(v, str):
        return v.lower()
    return v

for df in [movies5000, movies1000]:
    for col in df.columns:
        df[col] = df[col].apply(lower_all)

#Clean the year
movies5000['Year'] = movies5000['Year'].astype(str).str[:4]

#Add missed year + delete row
movies1000.loc[movies1000['Title'] == "apollo 13", 'Year'] = 1995
movies5000 = movies5000.dropna(subset=['Year'])

# Create a key
movies5000['key'] = movies5000['Title'] + "_" + movies5000['Year'].astype(str)
movies1000['key'] = movies1000['Title'] + "_" + movies1000['Year'].astype(str)

known_keys = set(movies5000['key'])
truly_new_movies = movies1000[~movies1000['key'].isin(known_keys)]
truly_new_movies = truly_new_movies.drop_duplicates(subset='key')
movies = pd.concat([movies5000, truly_new_movies], ignore_index=True)

#Clean some genres
movies['Genre'] = movies['Genre'].replace('0.0', 'NaN')
movies['Genre'] = movies['Genre'].replace('', 'Nan')
movies['Genre'] = movies['Genre'].replace('music', 'musical')

# Clean the data being used for KNN for then merge
movies1m = pd.read_csv("ml-1m/movies.dat", sep="::", engine="python", encoding="latin1", header=None)
movies1m = movies1m.drop(columns=[0])
movies1m = movies1m.rename(columns={1: 'Title', 2: 'Genre'})
movies1m['Title'] = movies1m['Title'].apply(lambda x: x[:-7])
movies1m['Genre'] = movies1m['Genre'].apply(lambda x: x.replace('|', ', '))
movies1m['Genre'] = movies1m['Genre'].apply(lambda x: x.replace('Children\'s', 'family'))
for col in movies1m.columns:
    movies1m[col] = movies1m[col].apply(lower_all)
movies1m['Title'] = movies1m['Title'].apply(lambda t: re.sub(r'^(.*),\s*(the|a|an)$', r'\2 \1', t))

#Merging the new dataset movies1m
mask = (~movies1m['Title'].isin(movies['Title'])) & (~movies1m['Title'].isin(movies['original_title']))
new_rows = movies1m[mask]
movies = pd.concat([movies, new_rows], ignore_index=True)
