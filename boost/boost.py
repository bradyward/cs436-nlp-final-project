import pandas as pd
from collections import Counter

## Create a profile

def profile_creation(knn_movies: list, movies_db: pd.DataFrame) -> dict:
    profile = {
        "directors": Counter(),
        "actors":    Counter(),
        "genres":    Counter(),
        "keywords":  Counter(),
        "found":     [],      # movies found with full info aka director, actors, keywords, genre
        "not_found": [],      # movies with only genre and title
    }

    #Load all the titles
    db_titles = movies_db['Title']

    #Loop for all the recommended movies (the movies that came from knn_movies)
    for movie in knn_movies:
        title = movie['title']
        rating = float(movie.get("rating", 3.0)) #The default weight if unable to get (3.0)
        genres = movie.get("genre", "")
        weight = rating / 5.0 #The default weight normalized from 0 to 1 

        #This adds the genres to the profile regardless of how much info we have on the row
        for genre in [g.strip() for g in genres.split(",") if g.strip()]:
            profile["genres"][genre] += weight
        #Matches the movie on db
        match = movies_db[db_titles == title]

        #If match is not empty then,
        if not match.empty: 
            row = match.iloc[0] #get row
    
            full_info = pd.notna(row.get("Director")) #load to full_info
    
            #If its one of rows that contains more then genre enter the if statement
            if full_info:
                #Add the title to the list of found movies and all of the other infos
                profile["found"].append(title)
                if pd.notna(row.get("Director")):
                    for director in str(row["Director"]).split(","):
                        profile["directors"][director.strip()] += weight
    
                if pd.notna(row.get("Actors")):
                    for actor in str(row["Actors"]).split(","):
                        profile["actors"][actor.strip()] += weight
    
                if pd.notna(row.get("Keywords")):
                    for keyword in str(row["Keywords"]).split(","):
                        profile["keywords"][keyword.strip()] += weight
            else:
                profile["not_found"].append(title) #case where row has only genre and title
        else:
            profile["not_found"].append(title) #case where unable to find a match for the movie
    return profile

## Logic for the scores
# The idea for the scores is the following:
# Same directors have the tendency of doing similar movies, therefore same directors add the weight of .30
# Same actors and genre add the weight of .25 since they will create similar movies
# Keywords may and may not be relevant so they add .20 weight
# If not full info, then we use the score of genre only

def scores(row, profile, seen): 
    #Load the row to check for scores, the profile to add the scores to, and information about whether the title was shared in the knn list

    title = str(row["Title"])
    # If movie was watched previously just return, no need to weight
    if title in seen:
        return None, None

    #Initialize the score and the reason array for the weights the movie receives
    score = 0.0
    reasons = []
    full_info = pd.notna(row.get("Director")) #load to full_info

    #Two vertences, one where we have a way to check for scores for director, actors, etc.
    #and another one where only the genre
    if full_info:
        if pd.notna(row.get("Director")) and profile["directors"]:
            for director in str(row["Director"]).split(","):
                director.strip()
                if director in profile["directors"]:
                    #Contribution checks for the biggest valued director aka favorite director
                    #and divide it by it so that the boost for each director acts accordingly
                    contribution = profile["directors"][director] / max(profile["directors"].values()) 
                    score += 0.3 * contribution #0.3 weight
                    reasons.append(f"director: {director}")
                    break
                    
        if pd.notna(row.get("Actors")) and profile["actors"]:
            actor_score = 0
            matched_actor = []
            for actor in str(row["Actors"]).split(","):
                actor = actor.strip()
                if actor in profile["actors"]:
                    actor_score += profile["actors"][actor]
                    matched_actor.append(actor)
            if actor_score > 0:
                #Contribution is slightly different than director, get all the score for all the actors for the movie
                #divide it by the most liked actor and then normalize it with 1, meaning that if a movie has a total score
                #of more than one, it will receive full actor boost.
                contribution = min(actor_score / max(profile["actors"].values()), 1.0)
                score += 0.25 * contribution #0.25 weight
                reasons.append(f"actor: {actor}")
                
        if pd.notna(row.get("Keywords")) and profile["keywords"]:
            keyword_score = 0
            for kw in str(row["Keywords"]).split(","):
                kw = kw.strip()
                if kw in profile["keywords"]:
                    keyword_score += profile["keywords"][kw]
            if keyword_score > 0:
                #Same logic as the actors
                contribution = min(keyword_score / max(profile["keywords"].values()), 1.0)
                score += 0.20 * contribution #0.20 weight
                reasons.append("keywords match")
        
        if pd.notna(row.get("Genre")) and profile["genres"]:
            movie_genres = [g.strip() for g in str(row["Genre"]).split(",")]
            matched = [g for g in movie_genres if g in profile["genres"]]
            if matched:
                genre_score  = sum(profile["genres"][g] for g in matched)
                #Same logic as actors and keywords.
                contribution = min(genre_score / max(profile["genres"].values()), 1.0)
                score += 0.25 * contribution   #0.25 weight
                reasons.append(f"genre only fallback: {matched}")
    
    #Only able to check the genre
    else:
        if pd.notna(row.get("Genre")) and profile["genres"]:
            movie_genres = [g.strip() for g in str(row["Genre"]).split(",")]
            matched = [g for g in movie_genres if g in profile["genres"]]

            if matched:
                # Score is now based on how many genres are a match and how frequent they are
                genre_score  = sum(profile["genres"][g] for g in matched)
                contribution = min(genre_score / max(profile["genres"].values()), 1.0)
                score += 1.0 * contribution   # full weight since it's all we have
                reasons.append(f"genre only fallback: {matched}")

    return round(score, 4), reasons

#The actual boost method, gets the list, the movies DF and the top k
def boost(knn_movies: list, movies_db: pd.DataFrame, top: int = 10) -> pd.DataFrame:

    #Create a profile, also adds to seen the titles recommended by the knn
    profile = profile_creation(knn_movies, movies_db)
    seen = set(m["title"].lower().strip() for m in knn_movies)
    results = []

    #Loop through every movie in the DF and score it
    #using the created user profile
    for _, row in movies_db.iterrows():
        score, reasons = scores(row, profile, seen)
        #Only store movies that score was bigger than 0
        if score is not None and score > 0:
            results.append({
                "title":       row["Title"],
                "genre":       row.get("Genre",    ""),
                "director":    row.get("Director", ""),
                "actors":      row.get("Actors",   ""),
                "rating":      row.get("Rating",   None),
                "boost_score": score,
                "reasons":     str(reasons),
            })
    #If there are no movies that scored above 0
    #Then no boosted movies
    if not results:
        print("No boosted movies found.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("boost_score", ascending=False).head(top)
    return results_df.reset_index(drop=True)

#Merges the recommendations for both knn and boosted movies to make a final recommendation
def recommendations(knn_movies: list, movies_db: pd.DataFrame,
                           knn_top: int = 5, boost_top: int = 5) -> pd.DataFrame:

    #Selects the top knn movies
    knn_df = pd.DataFrame(knn_movies).head(knn_top)
    knn_df["source"] = "KNN"

    #Selects the top boosted movies
    boosted_df = boost(knn_movies=knn_movies, movies_db=movies_db, top=boost_top)
    boosted_df["source"] = "Boost"

    #Combine both boosted and knn movies
    final = pd.concat([knn_df, boosted_df], ignore_index=True)

    return final[["title", "genre", "source"]]
