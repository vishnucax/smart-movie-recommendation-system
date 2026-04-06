import pandas as pd
import ast
import os
import re

def safe_parse(val):
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

def get_names(obj_list):
    if isinstance(obj_list, list):
        return [i.get('name', '') for i in obj_list if isinstance(i, dict) and 'name' in i]
    return []

def get_director(obj_list):
    if isinstance(obj_list, list):
        for i in obj_list:
            if isinstance(i, dict) and i.get('job') == 'Director':
                return [i.get('name', '')]
    return []

def get_top_cast(obj_list, top_n=5):
    names = get_names(obj_list)
    return names[:top_n]

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def process_datasets(raw_dir, output_path):
    print("Starting data processing...")
    
    # 1. Process Kaggle TMDB chunks
    movies_path = os.path.join(raw_dir, 'movies_metadata.csv')
    credits_path = os.path.join(raw_dir, 'credits.csv')
    keywords_path = os.path.join(raw_dir, 'keywords.csv')
    
    # Read movies_metadata
    print("Reading movies_metadata.csv...")
    try:
        df_movies = pd.read_csv(movies_path, low_memory=False)
        # Keep only required cols
        cols = ['id', 'title', 'overview', 'genres', 'original_language', 'vote_average']
        df_movies = df_movies[[c for c in cols if c in df_movies.columns]]
        # Clean ID
        df_movies = df_movies[df_movies['id'].apply(lambda x: str(x).isdigit())]
        df_movies['id'] = df_movies['id'].astype(int)
    except Exception as e:
        print(f"Skipping movies_metadata.csv: {e}")
        df_movies = pd.DataFrame(columns=['id', 'title', 'overview'])

    # Read credits
    print("Reading credits.csv...")
    try:
        df_credits = pd.read_csv(credits_path)
        if 'id' not in df_credits.columns and 'movie_id' in df_credits.columns:
            df_credits.rename(columns={'movie_id': 'id'}, inplace=True)
        df_credits['id'] = pd.to_numeric(df_credits['id'], errors='coerce')
        df_credits.dropna(subset=['id'], inplace=True)
        df_credits['id'] = df_credits['id'].astype(int)
        
        # Merge movies and credits
        df_movies = df_movies.merge(df_credits, on='id', how='left')
    except Exception as e:
        print(f"Skipping credits.csv: {e}")

    # Read keywords
    print("Reading keywords.csv...")
    try:
        df_keywords = pd.read_csv(keywords_path)
        df_keywords['id'] = pd.to_numeric(df_keywords['id'], errors='coerce')
        df_keywords.dropna(subset=['id'], inplace=True)
        df_keywords['id'] = df_keywords['id'].astype(int)
        
        df_movies = df_movies.merge(df_keywords, on='id', how='left')
    except Exception as e:
        print(f"Skipping keywords.csv: {e}")

    print("Parsing JSON fields from Kaggle TMDB payload...")
    # Parse JSON fields
    if 'genres' in df_movies.columns:
        df_movies['genres'] = df_movies['genres'].apply(safe_parse).apply(get_names)
    if 'cast' in df_movies.columns:
        df_movies['cast'] = df_movies['cast'].apply(safe_parse).apply(get_top_cast)
    if 'crew' in df_movies.columns:
        df_movies['director'] = df_movies['crew'].apply(safe_parse).apply(get_director)
    if 'keywords' in df_movies.columns:
        df_movies['keywords'] = df_movies['keywords'].apply(safe_parse).apply(get_names)
        
    df_movies.drop(columns=['crew', 'id'], inplace=True, errors='ignore')

    df_list = [df_movies]

    # 2. Process Netflix Dataset
    netflix_path = os.path.join(raw_dir, 'netflix_titles.csv')
    print("Reading netflix_titles.csv...")
    try:
        df_net = pd.read_csv(netflix_path)
        df_net = df_net[df_net['type'] == 'Movie'] # Only movies
        # map columns to: title, overview, genres, cast, director, original_language, vote_average
        df_net.rename(columns={
            'description': 'overview',
            'listed_in': 'genres'
        }, inplace=True)
        
        # Convert comma separated strings to lists
        for col in ['genres', 'cast', 'director']:
            if col in df_net.columns:
                df_net[col] = df_net[col].apply(lambda x: [i.strip() for i in str(x).split(',')] if not pd.isna(x) else [])
        
        # Netflix dataset has no numerical rating in 'rating' column (it's TV-MA etc)
        df_net['vote_average'] = 0.0
        df_net['original_language'] = 'en'
        df_net['keywords'] = [[] for _ in range(len(df_net))]
        
        cols = ['title', 'overview', 'genres', 'cast', 'director', 'original_language', 'vote_average', 'keywords']
        df_net = df_net[[c for c in cols if c in df_net.columns]]
        df_list.append(df_net)
    except Exception as e:
        print(f"Skipping netflix_titles.csv: {e}")

    # 3. Combine DataFrames
    print("Concatenating all mapped datasets...")
    final_df = pd.concat(df_list, ignore_index=True)
    
    # Drop records without title or empty title
    final_df.dropna(subset=['title'], inplace=True)
    final_df = final_df[final_df['title'].str.strip() != '']
    
    # Remove duplicates
    final_df.drop_duplicates(subset=['title'], keep='first', inplace=True)
    
    # Fill NAs
    for col in ['genres', 'keywords', 'cast', 'director']:
        if col not in final_df.columns:
            final_df[col] = [[] for _ in range(len(final_df))]
        else:
            final_df[col] = final_df[col].apply(lambda x: x if isinstance(x, list) else [])
            
    final_df['overview'] = final_df['overview'].fillna('')
    final_df['original_language'] = final_df['original_language'].fillna('en')
    final_df['vote_average'] = pd.to_numeric(final_df['vote_average'], errors='coerce').fillna(5.0)

    print("Building 'content' representation...")
    # Define a helper to join lists
    def join_list(lst):
        return ' '.join([str(i).replace(" ", "") for i in lst])
    
    # Text aggregation: We strip spaces from names so "Tom Hanks" becomes "TomHanks" - improves exact person matching in TF-IDF
    final_df['content'] = (
         final_df['overview'] + " " +
         final_df['genres'].apply(lambda x: ' '.join(x)) + " " +
         final_df['keywords'].apply(join_list) + " " +
         final_df['cast'].apply(join_list) + " " +
         final_df['director'].apply(join_list)
    )
    final_df['content'] = final_df['content'].apply(clean_text)

    print("Saving final dataset...")
    # Convert lists back to string representation (comma separated) for CSV storage to keep it clean
    for col in ['genres', 'keywords', 'cast', 'director']:
        final_df[col] = final_df[col].apply(lambda x: ', '.join(x))
        
    final_df.to_csv(output_path, index=False)
    print(f"Preprocessing completed. Shape: {final_df.shape}. Saved to {output_path}")

if __name__ == "__main__":
    raw_dir = "raw datasets" # User said "use datasets from ./raw datastes"
    output_path = "final_movies.csv"
    
    # Fallback to local raw if they mispelt it
    if not os.path.exists(raw_dir) and os.path.exists("raw_datasets"):
        raw_dir = "raw_datasets"
        
    process_datasets(raw_dir, output_path)
