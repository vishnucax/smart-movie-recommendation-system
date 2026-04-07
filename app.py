import streamlit as st
import pandas as pd
import requests
from model import load_data, get_tfidf_matrix, smart_search, get_recommendations

TMDB_API_KEY = "05bbd8e0feb6f309f8d228ff2c7fba73"

st.set_page_config(page_title="Smart Movie Recommender", layout="wide", page_icon="🎬")

# Custom CSS for modern dark theme and responsive layout
st.markdown("""
<style>
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }
    .movie-card {
        background-color: #1e1e2d;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
        border-left: 5px solid #FF416C;
        transition: transform 0.2s ease-in-out;
    }
    .movie-card:hover {
        transform: scale(1.02);
    }
    .movie-title {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: bold;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
    }
    .rating-badge {
        background-color: #FFD700;
        color: #000;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 1rem;
        font-weight: bold;
    }
    .movie-tag {
        background-color: #2c2c44;
        color: #00d2ff;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        margin-right: 8px;
        display: inline-block;
        margin-bottom: 10px;
        border: 1px solid #3a3a5a;
    }
    .overview-text {
        color: #dcdcdc;
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 15px;
    }
    .meta-text {
        color: #888;
        font-size: 0.9rem;
    }
    .meta-highlight {
        color: #FF4B2B;
        font-weight: bold;
    }
    .carousel-container {
        display: flex;
        overflow-x: auto;
        gap: 15px;
        padding: 20px 0;
        scrollbar-width: none; /* Firefox */
    }
    .carousel-container::-webkit-scrollbar {
        display: none; /* Safari and Chrome */
    }
    .carousel-item {
        flex: 0 0 auto;
        width: 150px;
        position: relative;
        transition: transform 0.3s ease;
        cursor: pointer;
    }
    .carousel-item:hover {
        transform: scale(1.08);
        z-index: 10;
    }
    .carousel-item img {
        width: 100%;
        height: 225px;
        object-fit: cover;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.6);
    }
    .carousel-title {
        color: #fff;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 8px;
        text-align: center;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .category-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.5rem;
        font-weight: bold;
        color: #e5e5e5;
        margin-top: 30px;
        margin-bottom: 5px;
        border-left: 4px solid #FF416C;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Smart Movie Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Find the perfect movie based on title, actor, director, or genre.</div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_movie_data():
    df = load_data()
    return df

@st.cache_resource(show_spinner=False)
def get_model(_df):
    tfidf, matrix = get_tfidf_matrix(_df)
    return tfidf, matrix

with st.spinner("Connecting to Movie Database and Initializing AI..."):
    df = load_movie_data()
    if not df.empty:
        tfidf, tfidf_matrix = get_model(df)

if df.empty:
    st.error("Movie dataset `final_movies.csv` not found. Please ensure data preprocessing has completed successfully.")
    st.stop()

# Search Component
search_col, stat_col = st.columns([3, 1])
with search_col:
    query = st.text_input("Search:", placeholder="e.g. The Matrix, Christopher Nolan, Action, Brad Pitt...", label_visibility="collapsed")
    
with stat_col:
    limit = st.slider("Results limit", min_value=5, max_value=30, value=10)

col_btn, _ = st.columns([1, 4])
with col_btn:
    search_btn = st.button("🔍 Search Engine", use_container_width=True)

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_poster(movie_title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        response = requests.get(url)
        data = response.json()
        if data.get('results'):
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        pass
    return "https://via.placeholder.com/150x225/1e1e2d/ffffff?text=No+Poster"

def render_movie(row):
    genres_html = ""
    for g in str(row['genres']).split(','):
        if g.strip():
            genres_html += f'<span class="movie-tag">{g.strip()}</span>'
            
    overview = str(row['overview'])
    if len(overview) > 300:
        overview = overview[:300] + "..."
        
    poster_url = fetch_poster(row['title'])
        
    st.markdown(f"""
    <div class="movie-card">
        <div style="display: flex; gap: 20px; align-items: flex-start;">
            <div style="flex: 0 0 150px;">
                <img src="{poster_url}" style="width: 150px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.5);">
            </div>
            <div style="flex: 1;">
                <div class="movie-title">
                    {row['title']} 
                    <span class="rating-badge">⭐ {row['vote_average']}</span>
                </div>
                <div style="margin-bottom: 15px;">{genres_html}</div>
                <div class="overview-text">{overview}</div>
                <div class="meta-text">
                    <span class="meta-highlight">Cast:</span> {row['cast']} <br/>
                    <span class="meta-highlight">Director:</span> {row['director']} | 
                    <span class="meta-highlight">Language:</span> {str(row['original_language']).upper()}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_movie_carousel(category_name, movies_df, limit=15):
    st.markdown(f'<div class="category-header">{category_name}</div>', unsafe_allow_html=True)
    
    items_html = ""
    for _, row in movies_df.head(limit).iterrows():
        poster_url = fetch_poster(row['title'])
        items_html += f"""
        <div class="carousel-item" title="{row['title']}">
            <img src="{poster_url}" alt="{row['title']}">
            <div class="carousel-title">{row['title']}</div>
        </div>
        """
        
    carousel_html = f"""
    <div class="carousel-container">
        {items_html}
    </div>
    """
    st.markdown(carousel_html, unsafe_allow_html=True)

if search_btn and query:
    q_type, match_df = smart_search(query, df)
    
    if q_type == 'none':
        st.warning("No matches found. Try another movie, actor, director, or genre.")
    else:
        st.success(f"Matched your search intelligently by: **{q_type.upper()}**")
        
        if q_type == 'title':
            target_movie = match_df.iloc[0]
            st.subheader("Movie Found:")
            render_movie(target_movie)
            
            st.markdown("---")
            st.markdown(f"### 🎯 People who liked **{target_movie['title']}** also liked:")
            recs = get_recommendations(target_movie['title'], df, tfidf_matrix, top_n=limit)
            
            if not recs.empty:
                for _, row in recs.iterrows():
                    render_movie(row)
            else:
                st.info("No recommendations available for this movie.")
                
        else:
            # Sort by rating
            match_df = match_df.sort_values(by='vote_average', ascending=False).head(limit)
            st.markdown(f"### Top {len(match_df)} Movies for {q_type.capitalize()}: {query}")
            for _, row in match_df.iterrows():
                render_movie(row)

elif not query:
    # No search active, show Netflix Home
    
    # Category 1: Trending Now
    trending = df.sort_values(by='vote_average', ascending=False).head(50)
    render_movie_carousel("Trending Now", trending, limit=15)
    
    # Category 2: Action & Adventure
    action = df[df['genres'].str.contains('Action|Adventure', case=False, na=False)].sort_values(by='vote_average', ascending=False)
    render_movie_carousel("Action & Adventure", action, limit=15)
    
    # Category 3: Sci-Fi Hits
    scifi = df[df['genres'].str.contains('Science Fiction|Sci-Fi', case=False, na=False)].sort_values(by='vote_average', ascending=False)
    render_movie_carousel("Sci-Fi Hits", scifi, limit=15)

    # Category 4: Comedy
    comedy = df[df['genres'].str.contains('Comedy', case=False, na=False)].sort_values(by='vote_average', ascending=False)
    render_movie_carousel("Top Comedies", comedy, limit=15)

    # Category 5: Christopher Nolan
    nolan = df[df['director'].str.contains('Christopher Nolan', case=False, na=False)].sort_values(by='vote_average', ascending=False)
    if not nolan.empty:
        render_movie_carousel("Christopher Nolan Masterpieces", nolan, limit=15)

# Display some stats quietly at the bottom
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #666;'><small>Powering recommendations for {len(df):,} movies.</small></p>", unsafe_allow_html=True)
