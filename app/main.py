"""
ReelSense: Explainable Movie Recommender System
Streamlit UI Application - WITH MODEL CLASS DEFINITIONS
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os


# ============================================================================
# MODEL CLASS DEFINITIONS (MUST MATCH TRAINING SCRIPT)
# ============================================================================

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class PopularityRecommender:
    def __init__(self):
        self.popular_movies = None
        self.movie_data = None
    
    def fit(self, ratings_df, movies_df):
        """Train popularity model"""
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_stats.columns = ['movieId', 'rating_count', 'rating_mean']
        
        movie_stats = movie_stats[movie_stats['rating_count'] >= 50]
        
        C = movie_stats['rating_mean'].mean()
        m = movie_stats['rating_count'].quantile(0.75)
        
        movie_stats['popularity_score'] = (
            (movie_stats['rating_count'] / (movie_stats['rating_count'] + m)) * movie_stats['rating_mean'] +
            (m / (movie_stats['rating_count'] + m)) * C
        )
        
        self.popular_movies = movie_stats.merge(movies_df, on='movieId')
        self.popular_movies = self.popular_movies.sort_values('popularity_score', ascending=False)
        self.movie_data = movies_df
        
        return self
    
    def recommend(self, user_id=None, n=10):
        """Get top N popular movies"""
        recommendations = self.popular_movies.head(n)[['movieId', 'title', 'genres', 'popularity_score']]
        return recommendations.to_dict('records')
    
    def explain(self, user_id, movie_id):
        """Generate explanation for recommendation"""
        movie = self.popular_movies[self.popular_movies['movieId'] == movie_id]
        if len(movie) > 0:
            movie = movie.iloc[0]
            return f"This movie is highly popular with {int(movie['rating_count'])} ratings and an average score of {movie['rating_mean']:.2f}/5.0"
        return "Popular movie among users"


class UserUserCF:
    def __init__(self, k=20):
        self.k = k
        self.user_item_matrix = None
        self.user_similarity = None
        self.movies_df = None
    
    def fit(self, ratings_df, movies_df):
        """Train user-user CF model"""
        self.user_item_matrix = ratings_df.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        user_matrix = self.user_item_matrix.values
        self.user_similarity = cosine_similarity(user_matrix)
        
        self.movies_df = movies_df
        return self
    
    def recommend(self, user_id, n=10):
        """Get top N recommendations for user"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        
        user_similarities = self.user_similarity[user_idx]
        similar_user_indices = user_similarities.argsort()[::-1][1:self.k+1]
        
        user_ratings = self.user_item_matrix.iloc[user_idx]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        predictions = {}
        for movie_id in unrated_movies:
            if movie_id in self.user_item_matrix.columns:
                movie_ratings = self.user_item_matrix[movie_id].iloc[similar_user_indices]
                similarities = user_similarities[similar_user_indices]
                
                rated_mask = movie_ratings > 0
                if rated_mask.sum() > 0:
                    pred = np.sum(movie_ratings[rated_mask] * similarities[rated_mask]) / np.sum(similarities[rated_mask])
                    predictions[movie_id] = pred
        
        top_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
        
        recommendations = []
        for movie_id, score in top_movies:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if len(movie_info) > 0:
                movie_info = movie_info.iloc[0]
                recommendations.append({
                    'movieId': movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'predicted_rating': score
                })
        
        return recommendations
    
    def explain(self, user_id, movie_id):
        """Generate explanation"""
        return f"Recommended because users similar to you enjoyed this movie"


class ItemItemCF:
    def __init__(self, k=20):
        self.k = k
        self.item_similarity = None
        self.user_ratings = None
        self.movies_df = None
    
    def fit(self, ratings_df, movies_df):
        """Train item-item CF model"""
        item_user_matrix = ratings_df.pivot(
            index='movieId',
            columns='userId',
            values='rating'
        ).fillna(0)
        
        self.item_similarity = cosine_similarity(item_user_matrix.values)
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=item_user_matrix.index,
            columns=item_user_matrix.index
        )
        
        self.user_ratings = ratings_df.groupby('userId').apply(
            lambda x: dict(zip(x['movieId'], x['rating']))
        ).to_dict()
        
        self.movies_df = movies_df
        return self
    
    def recommend(self, user_id, n=10):
        """Get top N recommendations"""
        if user_id not in self.user_ratings:
            return []
        
        user_rated_movies = self.user_ratings[user_id]
        
        predictions = {}
        for rated_movie, rating in user_rated_movies.items():
            if rated_movie in self.item_similarity.index:
                similar_movies = self.item_similarity[rated_movie]
                
                for movie_id, similarity in similar_movies.items():
                    if movie_id not in user_rated_movies:
                        if movie_id not in predictions:
                            predictions[movie_id] = 0
                        predictions[movie_id] += similarity * rating
        
        top_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
        
        recommendations = []
        for movie_id, score in top_movies:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if len(movie_info) > 0:
                movie_info = movie_info.iloc[0]
                recommendations.append({
                    'movieId': movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'predicted_rating': score
                })
        
        return recommendations
    
    def explain(self, user_id, movie_id):
        """Generate explanation"""
        if user_id in self.user_ratings:
            user_movies = list(self.user_ratings[user_id].keys())
            if movie_id in self.item_similarity.index:
                similar = self.item_similarity[movie_id][user_movies].nlargest(3)
                similar_titles = self.movies_df[self.movies_df['movieId'].isin(similar.index)]['title'].tolist()
                if similar_titles:
                    return f"Similar to movies you liked: {', '.join(similar_titles[:2])}"
        return "Based on your viewing history"


class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_matrix = None
        self.movie_indices = None
        self.movies_df = None
        self.user_profiles = {}
    
    def fit(self, ratings_df, movies_df):
        """Train content-based model"""
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(movies_df['content_features'])
        self.movie_indices = movies_df['movieId'].values
        self.movies_df = movies_df
        
        for user_id in ratings_df['userId'].unique():
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
            
            liked_movies = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].values
            
            if len(liked_movies) > 0:
                liked_indices = [np.where(self.movie_indices == mid)[0][0] 
                               for mid in liked_movies if mid in self.movie_indices]
                
                if liked_indices:
                    user_profile = np.array(self.tfidf_matrix[liked_indices].mean(axis=0)).flatten()
                    self.user_profiles[user_id] = user_profile
        
        return self
    
    def recommend(self, user_id, n=10):
        """Get top N content-based recommendations"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        similarities = cosine_similarity([user_profile], self.tfidf_matrix)[0]
        
        top_indices = similarities.argsort()[::-1][:n*2]
        
        recommendations = []
        for idx in top_indices:
            if len(recommendations) >= n:
                break
            
            movie_id = self.movie_indices[idx]
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            
            recommendations.append({
                'movieId': movie_id,
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'similarity_score': similarities[idx]
            })
        
        return recommendations
    
    def explain(self, user_id, movie_id):
        """Generate explanation"""
        movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(movie) > 0:
            genres = movie.iloc[0]['genres']
            return f"Matches your interest in genres: {genres}"
        return "Based on content similarity to your preferences"


class HybridRecommender:
    def __init__(self, models, weights=None):
        """
        models: dict of {name: model}
        weights: dict of {name: weight}
        """
        self.models = models
        self.weights = weights or {name: 1.0/len(models) for name in models}
        self.movies_df = None
    
    def fit(self, movies_df):
        self.movies_df = movies_df
        return self
    
    def recommend(self, user_id, n=10):
        """Get hybrid recommendations"""
        all_recommendations = {}
        
        for name, model in self.models.items():
            try:
                recs = model.recommend(user_id, n=n*2)
                weight = self.weights[name]
                
                for rec in recs:
                    movie_id = rec['movieId']
                    score = rec.get('predicted_rating', rec.get('popularity_score', rec.get('similarity_score', 1.0)))
                    
                    if movie_id not in all_recommendations:
                        all_recommendations[movie_id] = {
                            'score': 0,
                            'count': 0,
                            'sources': []
                        }
                    
                    all_recommendations[movie_id]['score'] += score * weight
                    all_recommendations[movie_id]['count'] += 1
                    all_recommendations[movie_id]['sources'].append(name)
            except Exception as e:
                continue
        
        for movie_id in all_recommendations:
            all_recommendations[movie_id]['final_score'] = (
                all_recommendations[movie_id]['score'] / all_recommendations[movie_id]['count']
            )
        
        top_movies = sorted(
            all_recommendations.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )[:n]
        
        recommendations = []
        for movie_id, data in top_movies:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if len(movie_info) > 0:
                movie_info = movie_info.iloc[0]
                recommendations.append({
                    'movieId': movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'hybrid_score': data['final_score'],
                    'sources': data['sources']
                })
        
        return recommendations
    
    def explain(self, user_id, movie_id, sources):
        """Generate hybrid explanation"""
        explanations = []
        for source in sources:
            if source in self.models:
                exp = self.models[source].explain(user_id, movie_id)
                explanations.append(f"{source}: {exp}")
        
        return " | ".join(explanations)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🎬 ReelSense - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .explanation-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

# @st.cache_resource
# def load_models():
#     """Load all trained models"""
#     models = {}
#     model_status = {}
    
#     # Check multiple possible paths for models
#     possible_paths = [
#         'app/models',
#         'models',
#         './models',
#         '../models'
#     ]
    
#     models_dir = None
#     for path in possible_paths:
#         if os.path.exists(path):
#             models_dir = path
#             break
    
#     if models_dir is None:
#         st.error("❌ Models folder not found! Please ensure models are in 'app/models/' or 'models/' directory")
#         return {}, {}
    
#     # Model files to load
#     model_files = {
#         'popularity': 'popularity_model.pkl',
#         'user_cf': 'user_cf_model.pkl',
#         'content': 'content_model.pkl',
#         'hybrid': 'hybrid_model.pkl'
#     }
    
#     for name, filename in model_files.items():
#         filepath = os.path.join(models_dir, filename)
#         try:
#             with open(filepath, 'rb') as f:
#                 models[name] = pickle.load(f)
#             model_status[name] = "✅ Loaded"
#         except FileNotFoundError:
#             model_status[name] = "❌ Not found"
#         except Exception as e:
#             model_status[name] = f"⚠️ Error: {str(e)[:30]}"
    
#     return models, model_status

@st.cache_resource
def train_all_models(ratings_df, movies_df):
    st.info("⚙️ Training models on startup (cloud mode)...")

    # Prepare content features if missing
    if 'content_features' not in movies_df.columns:
        movies_df['content_features'] = (
            movies_df['title'].fillna('') + ' ' +
            movies_df['genres'].fillna('')
        )

    popularity = PopularityRecommender().fit(ratings_df, movies_df)
    user_cf = UserUserCF().fit(ratings_df, movies_df)
    item_cf = ItemItemCF().fit(ratings_df, movies_df)
    content = ContentBasedRecommender().fit(ratings_df, movies_df)

    hybrid = HybridRecommender(
        models={
            'popularity': popularity,
            'user_cf': user_cf,
            'content': content
        },
        weights={
            'popularity': 0.2,
            'user_cf': 0.5,
            'content': 0.3
        }
    ).fit(movies_df)

    models = {
        'popularity': popularity,
        'user_cf': user_cf,
        'content': content,
        'hybrid': hybrid
    }

    return models


@st.cache_data
def load_data():
    """Load movie data - flexible path handling"""
    # Try multiple possible locations
    possible_paths = ['', 'app/', '../', './data/']
    
    for base_path in possible_paths:
        try:
            movies_path = base_path + 'movies.csv'
            ratings_path = base_path + 'ratings.csv'
            
            if os.path.exists(movies_path) and os.path.exists(ratings_path):
                movies_df = pd.read_csv(movies_path)
                ratings_df = pd.read_csv(ratings_path)
                
                # Basic preprocessing
                if 'title' in movies_df.columns:
                    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')
                
                return movies_df, ratings_df, True
        except Exception as e:
            continue
    
    # If no data found, create minimal sample
    st.warning("⚠️ Dataset files not found. Using minimal sample data.")
    movies_df = pd.DataFrame({
        'movieId': range(1, 101),
        'title': [f'Movie {i} ({2000+i})' for i in range(1, 101)],
        'genres': ['Action|Adventure'] * 100,
        'year': [str(2000+i) for i in range(1, 101)]
    })
    
    ratings_df = pd.DataFrame({
        'userId': np.repeat(range(1, 51), 10),
        'movieId': np.random.randint(1, 101, 500),
        'rating': np.random.uniform(3.0, 5.0, 500),
        'timestamp': pd.date_range(start='2020-01-01', periods=500, freq='H')
    })
    
    return movies_df, ratings_df, False

@st.cache_data
def load_extra_data():
    """Load optional tags and links data"""
    tags_df = None
    links_df = None
    
    possible_paths = ['', 'app/', '../', './data/']
    
    for base_path in possible_paths:
        try:
            tags_path = base_path + 'tags.csv'
            links_path = base_path + 'links.csv'
            
            if os.path.exists(tags_path):
                tags_df = pd.read_csv(tags_path)
            if os.path.exists(links_path):
                links_df = pd.read_csv(links_path)
        except:
            pass
    
    return tags_df, links_df

@st.cache_data
def load_metadata():
    """Load model metadata"""
    possible_paths = ['', 'app/', '../', './data/']
    
    for path in possible_paths:
        try:
            with open(path + "model_metadata.json", 'r') as f:
                return json.load(f)
        except:
            continue
    
    return {
        'training_date': 'Not Available',
        'dataset_stats': {
            'total_ratings': 'N/A',
            'total_users': 'N/A',
            'total_movies': 'N/A'
        }
    }

# Load everything
# Try to load pickles locally, otherwise train
# models, model_status = load_models()

# if not models:
#     st.warning("⚠️ No pickled models found. Training models now...")
#     models = train_all_models(ratings_df, movies_df)
#     model_status = {k: "⚙️ Trained on startup" for k in models}

# movies_df, ratings_df, data_loaded = load_data()
# tags_df, links_df = load_extra_data()   # ✅ ADD THIS
# metadata = load_metadata()

# =========================
# LOAD DATA FIRST
# =========================
movies_df, ratings_df, data_loaded = load_data()
tags_df, links_df = load_extra_data()
metadata = load_metadata()

# =========================
# TRAIN MODELS (CLOUD MODE)
# =========================
st.info("🚀 Initializing recommender models...")

models = train_all_models(ratings_df, movies_df)
model_status = {k: "⚙️ Trained on startup" for k in models}


# ============================================================================
# SIDEBAR - USER SELECTION & SETTINGS
# ============================================================================

st.sidebar.title("🎬 ReelSense")
st.sidebar.markdown("### Explainable Movie Recommendations")

# Show model loading status
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Models Status")
for model_name, status in model_status.items():
    st.sidebar.text(f"{model_name}: {status}")

# User selection
st.sidebar.markdown("---")
st.sidebar.subheader("👤 User Selection")

if data_loaded and len(ratings_df) > 0:
    unique_users = sorted(ratings_df['userId'].unique())
    selected_user = st.sidebar.selectbox(
        "Select User ID",
        unique_users,
        index=0
    )
else:
    selected_user = st.sidebar.number_input("Enter User ID", min_value=1, value=1)

# Model selection
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Model Selection")

available_models = []
if 'hybrid' in models:
    available_models.append('Hybrid (Best)')
if 'popularity' in models:
    available_models.append('Popularity-Based')
if 'user_cf' in models:
    available_models.append('User-User CF')
if 'content' in models:
    available_models.append('Content-Based')

if not available_models:
    st.sidebar.error("No models loaded!")
    available_models = ['No models available']

model_choice = st.sidebar.radio(
    "Choose Recommendation Model",
    available_models,
    index=0
)

# Number of recommendations
n_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

# Display metadata
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.metric("Total Ratings", metadata['dataset_stats'].get('total_ratings', 'N/A'))
st.sidebar.metric("Total Users", metadata['dataset_stats'].get('total_users', 'N/A'))
st.sidebar.metric("Total Movies", metadata['dataset_stats'].get('total_movies', 'N/A'))
st.sidebar.caption(f"Last trained: {metadata.get('training_date', 'N/A')}")

st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💻 Project Team")

st.sidebar.markdown("""
**Lead Developer:** Suchana Hazra  
**Developer 1:** Argha Pal <br> 
**Developer 2:** Meghma Das  
""")


# ============================================================================
# REST OF THE UI CODE (Same as before - tabs, recommendations, etc.)
# ============================================================================

# Header
st.markdown('<div class="main-header">🎬 ReelSense</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Explainable Movie Recommender with Diversity Optimization</div>',
    unsafe_allow_html=True
)

# Show data loading status
if data_loaded:
    st.markdown(f"""
    <div class="success-box">
        ✅ <strong>System Ready!</strong> Loaded {len(movies_df):,} movies and {len(ratings_df):,} ratings
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Using sample data. Please ensure movies.csv and ratings.csv are available.")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Recommendations",
    "📊 Analytics", 
    "🔍 Explore Movies",
    "ℹ️ About"
])

# TAB 1: RECOMMENDATIONS
with tab1:
    # st.markdown("### 🎥 Featured Movie / Banner")
    st.markdown("### 🎥 Featured Banner")

    st.header(f"Recommendations for User {selected_user}")
    
    # Get user's watching history
    user_ratings = ratings_df[ratings_df['userId'] == selected_user]
    if len(user_ratings) > 0:
        user_ratings = user_ratings.sort_values('rating', ascending=False)
    
    if len(user_ratings) > 0:
        st.subheader("📚 Your Watching History")
        
        # Display top rated movies
        top_rated = user_ratings.head(5).merge(movies_df, on='movieId', how='left')
        
        cols = st.columns(5)
        for idx, (_, movie) in enumerate(top_rated.iterrows()):
            with cols[idx]:
                st.markdown(f"""
                <div style='text-align: center; padding: 10px;'>
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                border-radius: 10px; padding: 20px; color: white; min-height: 150px;'>
                        <h4 style='font-size: 14px; margin: 0;'>{movie.get('title', 'Unknown')[:30]}...</h4>
                        <p style='margin: 10px 0;'>⭐ {movie['rating']:.1f}/5.0</p>
                        <p style='font-size: 11px; opacity: 0.9;'>{movie.get('genres', 'N/A')}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Get recommendations button
    if st.button("🎬 Get Recommendations", type="primary", use_container_width=True):
        if not models:
            st.error("❌ No models loaded! Please check your models folder.")
        else:
            with st.spinner("Generating personalized recommendations..."):
                # Select model based on choice
                selected_model = None
                if model_choice == 'Hybrid (Best)' and 'hybrid' in models:
                    selected_model = models['hybrid']
                elif model_choice == 'Popularity-Based' and 'popularity' in models:
                    selected_model = models['popularity']
                elif model_choice == 'User-User CF' and 'user_cf' in models:
                    selected_model = models['user_cf']
                elif model_choice == 'Content-Based' and 'content' in models:
                    selected_model = models['content']
                else:
                    # Fallback to first available model
                    selected_model = list(models.values())[0] if models else None
                
                if selected_model is None:
                    st.error("❌ Selected model not available!")
                else:
                    # Get recommendations
                    try:
                        recommendations = selected_model.recommend(selected_user, n=n_recommendations)
                        
                        st.success(f"✅ Generated {len(recommendations)} recommendations using {model_choice}!")
                        
                        # Display recommendations
                        st.subheader("🎯 Your Personalized Recommendations")
                        
                        for idx, rec in enumerate(recommendations, 1):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                # Movie card
                                score = rec.get('hybrid_score', rec.get('predicted_rating', rec.get('popularity_score', rec.get('similarity_score', 0))))
                                
                                st.markdown(f"""
                                <div class="movie-card">
                                    <h3>#{idx} {rec.get('title', 'Unknown Movie')}</h3>
                                    <p><strong>Genres:</strong> {rec.get('genres', 'N/A')}</p>
                                    <p><strong>Score:</strong> {score:.3f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Explanation
                                try:
                                    if 'sources' in rec and hasattr(selected_model, 'explain'):
                                        explanation = selected_model.explain(
                                            selected_user, 
                                            rec['movieId'], 
                                            rec['sources']
                                        )
                                    elif hasattr(selected_model, 'explain'):
                                        explanation = selected_model.explain(selected_user, rec['movieId'])
                                    else:
                                        explanation = "Recommended based on your preferences and viewing history."
                                except Exception as e:
                                    explanation = "Recommended based on your preferences and viewing history."
                                
                                st.markdown(f"""
                                <div class="explanation-box" 
     style="background-color: #e8f4fd; 
            border-left: 5px solid #2196f3; 
            padding: 12px; 
            border-radius: 6px; 
            color: #0d47a1;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
    <strong>💡 Why this recommendation?</strong><br>
    {explanation}
</div>

                                """, unsafe_allow_html=True)
                            
                            with col2:
                                # Quick actions
                                st.markdown("### Quick Actions")
                                if st.button("👍 Like", key=f"like_{idx}"):
                                    st.toast("Added to favorites! 💚")
                                if st.button("👎 Not Interested", key=f"dislike_{idx}"):
                                    st.toast("Noted! We'll improve your recommendations.")
                                if st.button("ℹ️ More Info", key=f"info_{idx}"):
                                    st.info(f"MovieID: {rec.get('movieId', 'N/A')}")
                    
                    except Exception as e:
                        st.error(f"❌ Error generating recommendations: {str(e)}")
                        st.exception(e)

# TAB 2: ANALYTICS
# with tab2:
#     st.header("📊 User Analytics")
    
#     if len(user_ratings) > 0:
#         col1, col2, col3, col4 = st.columns(4)
        
#         col1.metric("Movies Rated", len(user_ratings))
#         col2.metric("Average Rating", f"{user_ratings['rating'].mean():.2f}")
#         col3.metric("Highest Rating", f"{user_ratings['rating'].max():.1f}")
#         col4.metric("Total Watch Time", f"{len(user_ratings) * 2:.0f}h")
        
#         # Rating distribution
#         st.subheader("Your Rating Distribution")
#         fig = px.histogram(
#             user_ratings,
#             x='rating',
#             nbins=10,
#             title="Your Rating Pattern",
#             labels={'rating': 'Rating', 'count': 'Number of Movies'},
#             color_discrete_sequence=['#667eea']
#         )
#         fig.update_layout(height=400)
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info(f"No rating history found for User {selected_user}")
# with tab2:
   
with tab2:
    st.header("📊 User & Dataset Analytics (EDA)")

    # ======================
    # GLOBAL THEME
    # ======================
    import plotly.io as pio
    pio.templates.default = "plotly_dark"

    QUAL_COLORS = px.colors.qualitative.Set2
    BOLD_COLORS = px.colors.qualitative.Bold

    # ======================
    # OVERALL DATASET METRICS
    # ======================
    st.subheader("📌 Overall Dataset Overview")

    st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #1f2933;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", ratings_df['userId'].nunique())
    col2.metric("Total Movies", movies_df['movieId'].nunique())
    col3.metric("Total Ratings", len(ratings_df))
    col4.metric("Total Tags", len(tags_df) if tags_df is not None else "N/A")

    st.markdown("---")

    # ======================
    # ⭐ OVERALL RATING DISTRIBUTION
    # ======================
    st.subheader("⭐ Overall Rating Distribution")

    fig = px.histogram(
        ratings_df,
        x='rating',
        nbins=10,
        title="Distribution of All Ratings",
        color_discrete_sequence=[BOLD_COLORS[0]]
    )
    fig.update_layout(height=420, title_font_size=20, bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # 👤 RATINGS PER USER
    # ======================
    st.subheader("👤 Ratings Per User")

    ratings_per_user = ratings_df.groupby('userId').size().reset_index(name='num_ratings')

    fig = px.histogram(
        ratings_per_user,
        x='num_ratings',
        nbins=30,
        title="Number of Ratings per User",
        color_discrete_sequence=[QUAL_COLORS[2]]
    )
    fig.update_traces(opacity=0.85)
    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # 🎬 MOVIE POPULARITY
    # ======================
    st.subheader("🎬 Movie Popularity (Ratings per Movie)")

    ratings_per_movie = ratings_df.groupby('movieId').size().reset_index(name='num_ratings')
    popular_movies = ratings_per_movie.merge(movies_df, on='movieId', how='left')
    top_popular = popular_movies.sort_values('num_ratings', ascending=False).head(15)

    fig = px.bar(
        top_popular,
        x='num_ratings',
        y='title',
        orientation='h',
        title="Top 15 Most Rated Movies",
        color_discrete_sequence=[QUAL_COLORS[4]]
    )
    fig.update_layout(height=520, title_font_size=20,
                      yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # 🏆 HIGHEST RATED MOVIES
    # ======================
    st.subheader("🏆 Highest Rated Movies (Min 50 Ratings)")

    movie_stats = ratings_df.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        count_rating=('rating', 'count')
    ).reset_index()

    high_quality = movie_stats[movie_stats['count_rating'] >= 50]
    high_quality = high_quality.merge(movies_df, on='movieId', how='left')
    top_rated = high_quality.sort_values('avg_rating', ascending=False).head(15)

    fig = px.bar(
        top_rated,
        x='avg_rating',
        y='title',
        orientation='h',
        title="Top 15 Highest Rated Movies (≥50 Ratings)",
        color_discrete_sequence=[BOLD_COLORS[3]]
    )
    fig.update_layout(height=520, title_font_size=20,
                      yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # 🎭 GENRE DISTRIBUTION
    # ======================
    st.subheader("🎭 Genre Distribution")

    genres_series = movies_df['genres'].str.split('|').explode()
    genre_counts = genres_series.value_counts().reset_index()
    genre_counts.columns = ['genre', 'count']

    fig = px.pie(
        genre_counts.head(15),
        names='genre',
        values='count',
        title="Top 15 Genres by Movie Count",
        color_discrete_sequence=px.colors.qualitative.G10,
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # 🏷️ TAG ANALYSIS
    # ======================
    if tags_df is not None:
        st.subheader("🏷️ Tag Analysis")

        top_tags = tags_df['tag'].value_counts().head(20).reset_index()
        top_tags.columns = ['tag', 'count']

        fig = px.bar(
            top_tags,
            x='count',
            y='tag',
            orientation='h',
            title="Top 20 Most Common Tags",
            color_discrete_sequence=[QUAL_COLORS[1]]
        )
        fig.update_layout(height=600, title_font_size=20)
        st.plotly_chart(fig, use_container_width=True)

        tags_per_movie = tags_df.groupby('movieId').size().reset_index(name='num_tags')
        fig = px.histogram(
            tags_per_movie,
            x='num_tags',
            nbins=30,
            title="Distribution of Tags per Movie",
            color_discrete_sequence=[BOLD_COLORS[5]]
        )
        fig.update_traces(opacity=0.85)
        st.plotly_chart(fig, use_container_width=True)

    # ======================
    # ⏳ RATINGS OVER TIME
    # ======================
    st.subheader("⏳ Ratings Over Time")

    if np.issubdtype(ratings_df['timestamp'].dtype, np.number):
        ratings_df['datetime'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    else:
        ratings_df['datetime'] = pd.to_datetime(ratings_df['timestamp'])

    ratings_by_year = ratings_df.groupby(
        ratings_df['datetime'].dt.year
    ).size().reset_index(name='count')

    fig = px.line(
        ratings_by_year,
        x='datetime',
        y='count',
        title="Ratings Activity Over Years"
    )
    fig.update_traces(line=dict(width=4))
    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # 🙋 USER-SPECIFIC ANALYTICS
    # ======================
    st.markdown("---")
    st.subheader(f"🙋 Personalized Analytics for User {selected_user}")

    user_ratings = ratings_df[ratings_df['userId'] == selected_user]

    if len(user_ratings) > 0:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Movies Rated", len(user_ratings))
        col2.metric("Average Rating", f"{user_ratings['rating'].mean():.2f}")
        col3.metric("Highest Rating", f"{user_ratings['rating'].max():.1f}")
        col4.metric("Lowest Rating", f"{user_ratings['rating'].min():.1f}")

        fig = px.histogram(
            user_ratings,
            x='rating',
            nbins=10,
            title="Your Rating Distribution",
            color_discrete_sequence=[BOLD_COLORS[6]]
        )
        st.plotly_chart(fig, use_container_width=True)

        user_movies = user_ratings.merge(movies_df, on='movieId', how='left')
        user_genres = user_movies['genres'].str.split('|').explode()
        user_genre_counts = user_genres.value_counts().reset_index()
        user_genre_counts.columns = ['genre', 'count']

        fig = px.bar(
            user_genre_counts.head(10),
            x='count',
            y='genre',
            orientation='h',
            title="Your Top Genres",
            color_discrete_sequence=[QUAL_COLORS[6]]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No ratings available for this user.")


# TAB 3: EXPLORE MOVIES
with tab3:
    st.header("🔍 Explore Movie Catalog")
    
    search_query = st.text_input("🔎 Search movies", placeholder="Enter movie title...")
    
    filtered_movies = movies_df.copy()
    
    if search_query:
        filtered_movies = filtered_movies[
            filtered_movies['title'].str.contains(search_query, case=False, na=False)
        ]
    
    st.subheader(f"Found {len(filtered_movies)} movies")
    
    # Display in grid
    movies_per_row = 4
    for i in range(0, min(len(filtered_movies), 20), movies_per_row):
        cols = st.columns(movies_per_row)
        for j, (_, movie) in enumerate(filtered_movies.iloc[i:i+movies_per_row].iterrows()):
            with cols[j]:
                st.markdown(f"""
                <div style='background: #f8f9fa; border-radius: 8px; padding: 15px; 
                            margin: 5px; min-height: 200px; border: 2px solid #e9ecef;'>
                    <h4 style='font-size: 14px; color: #333;'>{movie['title'][:40]}...</h4>
                    <p style='font-size: 12px; color: #666;'><strong>Genres:</strong> {movie.get('genres', 'N/A')}</p>
                    <p style='font-size: 11px; color: #999;'>ID: {movie['movieId']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    if len(filtered_movies) > 20:
        st.info(f"Showing first 20 of {len(filtered_movies)} results.")

# TAB 4: ABOUT
with tab4:
    st.header("ℹ️ About ReelSense")
    
    st.markdown("""
    ## 🎬 ReelSense: Explainable Movie Recommender System
    
    ReelSense provides personalized, explainable movie recommendations using multiple algorithms.
    
    ### ✨ Key Features
    
    - **🎯 Personalized**: Tailored to your taste
    - **💡 Explainable**: Clear reasons for each recommendation
    - **🤖 Multiple Models**: Hybrid approach for best results
    
    ### 🛠️ Models Loaded
    """)
    
    for model_name, status in model_status.items():
        st.markdown(f"- **{model_name.replace('_', ' ').title()}**: {status}")
    
    st.markdown("""
    ### 📊 System Statistics
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models Loaded", len([s for s in model_status.values() if "✅" in s]))
    col2.metric("Total Models", len(model_status))
    col3.metric("Movies", f"{len(movies_df):,}")
    col4.metric("Ratings", f"{len(ratings_df):,}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>🎬 <strong>ReelSense</strong> - Explainable Movie Recommender System</p>
    <p style='font-size: 0.9rem;'>Built with ❤️ using Streamlit & Python</p>
</div>
""", unsafe_allow_html=True)