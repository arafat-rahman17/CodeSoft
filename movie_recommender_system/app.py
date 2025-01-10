

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({
    'title': ['Inception', 'Interstellar', 'The Dark Knight', 'Memento', 'The Prestige', 'Shutter Island'],
    'tags': ['dream heist sci-fi', 'space sci-fi epic', 'hero vigilante action', 'memory mystery thriller', 'magic mystery drama', 'psychological thriller mystery']
})


cv = CountVectorizer(stop_words='english')
vectorized_matrix = cv.fit_transform(movies['tags'])


similarity = cosine_similarity(vectorized_matrix)

def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]  # Get the index of the movie
        distances = similarity[movie_index]  # Get similarity scores
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]  # Top 5 similar movies

        
        print(f"Movies similar to '{movie}':")
        for i in movie_list:
            print(movies.iloc[i[0]].title)
    except IndexError:
        print(f"'{movie}' not found in the dataset. Please try a different title.")


recommend('Inception')
