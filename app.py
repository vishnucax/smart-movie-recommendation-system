import streamlit as st
import pandas as pd
from model import load_data, get_tfidf_matrix, smart_search, get_recommendations

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

def render_movie(row):
    genres_html = ""
    for g in str(row['genres']).split(','):
        if g.strip():
            genres_html += f'<span class="movie-tag">{g.strip()}</span>'
            
    overview = str(row['overview'])
    if len(overview) > 300:
        overview = overview[:300] + "..."
        
    st.markdown(f"""
    <div class="movie-card">
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
    """, unsafe_allow_html=True)

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

# Display some stats quietly at the bottom
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #666;'><small>Powering recommendations for {len(df):,} movies.</small></p>", unsafe_allow_html=True)
