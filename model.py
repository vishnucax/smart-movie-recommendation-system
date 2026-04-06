import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_data(filepath='final_movies.csv'):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return pd.DataFrame()
    df['content'] = df['content'].fillna('')
    df['title'] = df['title'].fillna('')
    return df

def get_tfidf_matrix(df):
    if df.empty:
        return None, None
    tfidf = TfidfVectorizer(stop_words='english')
    # Fit and transform
    tfidf_matrix = tfidf.fit_transform(df['content'])
    return tfidf, tfidf_matrix

def smart_search(query, df):
    """
    Determine if query hits a movie title, actor, director, genre, or language
    Heuristics: exact match or partial match on these fields.
    """
    if df.empty:
        return 'none', pd.DataFrame()
        
    q = query.lower().strip()
    
    # 1. Exact or partial movie title
    title_matches = df[df['title'].str.lower().str.contains(q, na=False, regex=False)]
    
    # 2. Cast match
    cast_matches = df[df['cast'].str.lower().str.contains(q, na=False, regex=False)]
    
    # 3. Director match
    dir_matches = df[df['director'].str.lower().str.contains(q, na=False, regex=False)]
    
    # 4. Genre match
    genre_matches = df[df['genres'].str.lower().str.contains(q, na=False, regex=False)]
    
    # Prioritize exact title match
    exact_title = df[df['title'].str.lower() == q]
    if len(exact_title) > 0:
        return 'title', exact_title
        
    if len(title_matches) > 0:
        return 'title', title_matches
    if len(cast_matches) > 0:
        return 'cast', cast_matches
    if len(dir_matches) > 0:
        return 'director', dir_matches
    if len(genre_matches) > 0:
        return 'genre', genre_matches
    
    return 'none', pd.DataFrame()

def get_recommendations(movie_title, df, tfidf_matrix, top_n=10):
    if df.empty or tfidf_matrix is None:
        return pd.DataFrame()
        
    # Find index of movie
    idx_list = df[df['title'].str.lower() == movie_title.lower()].index.tolist()
    if not idx_list:
        return pd.DataFrame()
        
    idx = idx_list[0]
    
    # Get vector for this movie (1 x V)
    movie_vector = tfidf_matrix[idx]
    
    # Compute cosine similarity on the fly against all vectors (N x V -> output N)
    sim_scores = linear_kernel(movie_vector, tfidf_matrix).flatten()
    
    # Get indices of top_n movies (excl. itself)
    sim_indices = sim_scores.argsort()[-(top_n+1):][::-1]
    
    # Filter out the movie itself
    sim_indices = [i for i in sim_indices if i != idx][:top_n]
    
    return df.iloc[sim_indices]
