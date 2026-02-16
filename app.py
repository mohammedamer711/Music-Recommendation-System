# import streamlit as st
# import math
# import random
# import copy

# # -----------------------------
# # Page Config
# # -----------------------------
# st.set_page_config(page_title="Music Recommender", layout="centered")


# st.title("🎵 Music Recommendation System")
# st.write("K-Means Clustering + KNN (User Feature Input)")

# # -----------------------------
# # Sample Dataset
# # -----------------------------
# original_songs = [
#     {"id": 1, "name": "Song A", "features": [0.8, 0.7, 120, 0.9, 0.1, -5, 0.0, 0.05]},
#     {"id": 2, "name": "Song B", "features": [0.75, 0.65, 118, 0.85, 0.15, -6, 0.02, 0.04]},
#     {"id": 3, "name": "Song C", "features": [0.2, 0.3, 90, 0.2, 0.9, -12, 0.8, 0.1]},
#     {"id": 4, "name": "Song D", "features": [0.25, 0.35, 95, 0.25, 0.85, -10, 0.75, 0.08]},
#     {"id": 5, "name": "Song E", "features": [0.6, 0.8, 130, 0.7, 0.2, -4, 0.05, 0.03]},
#     {"id": 6, "name": "Song F", "features": [0.55, 0.75, 128, 0.65, 0.25, -5, 0.04, 0.02]},
# ]

# feature_names = [
#     "Energy",
#     "Danceability",
#     "Tempo",
#     "Valence",
#     "Acousticness",
#     "Loudness",
#     "Instrumentalness",
#     "Speechiness"
# ]

# songs = copy.deepcopy(original_songs)

# # -----------------------------
# # Utilities
# # -----------------------------
# def normalize_features(dataset):
#     feature_length = len(dataset[0]["features"])

#     mins = [min(song["features"][i] for song in dataset) for i in range(feature_length)]
#     maxs = [max(song["features"][i] for song in dataset) for i in range(feature_length)]

#     for song in dataset:
#         song["features"] = [
#             (song["features"][i] - mins[i]) / (maxs[i] - mins[i])
#             if maxs[i] != mins[i] else 0
#             for i in range(feature_length)
#         ]

#     return mins, maxs


# def normalize_query(query_features, mins, maxs):
#     normalized = []
#     for i in range(len(query_features)):
#         if maxs[i] != mins[i]:
#             value = (query_features[i] - mins[i]) / (maxs[i] - mins[i])
#         else:
#             value = 0
#         normalized.append(value)
#     return normalized


# def euclidean(a, b):
#     return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


# def k_means(dataset, k=2, iters=10):
#     centroids = random.sample([s["features"] for s in dataset], k)

#     for _ in range(iters):
#         clusters = {i: [] for i in range(k)}

#         for song in dataset:
#             idx = min(range(k),
#                       key=lambda c: euclidean(song["features"], centroids[c]))
#             clusters[idx].append(song)

#         for i in range(k):
#             if clusters[i]:
#                 centroids[i] = [
#                     sum(song["features"][d] for song in clusters[i]) / len(clusters[i])
#                     for d in range(len(dataset[0]["features"]))
#                 ]

#     return clusters, centroids


# def knn(query_features, cluster, k=3):
#     ranked = sorted(
#         [(euclidean(query_features, s["features"]), s)
#          for s in cluster],
#         key=lambda x: x[0]
#     )
#     return [s for _, s in ranked[:k]]


# # -----------------------------
# # Sidebar Inputs
# # -----------------------------
# st.sidebar.header("🎛 Enter Your Preferred Features")

# user_features = []

# for i, feature in enumerate(feature_names):
#     if feature == "Tempo":
#         value = st.sidebar.slider(feature, 80, 150, 110)
#     elif feature == "Loudness":
#         value = st.sidebar.slider(feature, -20, 0, -6)
#     else:
#         value = st.sidebar.slider(feature, 0.0, 1.0, 0.5)
#     user_features.append(value)

# clusters_k = st.sidebar.slider("Clusters (K-Means)", 2, 4, 2)
# neighbors_k = st.sidebar.slider("Neighbors (KNN)", 1, len(songs), 3)

# # -----------------------------
# # Recommend Button
# # -----------------------------
# if st.sidebar.button("Recommend"):

#     mins, maxs = normalize_features(songs)

#     normalized_query = normalize_query(user_features, mins, maxs)

#     clusters, centroids = k_means(songs, k=clusters_k)

#     cluster_idx = min(range(len(centroids)),
#                       key=lambda i: euclidean(normalized_query, centroids[i]))

#     st.subheader("🎧 Recommended Songs")

#     recs = knn(normalized_query, clusters[cluster_idx], k=neighbors_k)

#     for r in recs:
#         st.success(r["name"])

#     st.subheader("📦 Cluster Composition")
#     for i, cl in clusters.items():
#         st.write(f"Cluster {i + 1}: {[s['name'] for s in cl]}")

# # -----------------------------
# # Print Normalized Dataset
# # -----------------------------
# st.markdown("---")
# st.subheader("📊 Normalized Dataset Used")

# mins, maxs = normalize_features(songs)

# for song in songs:
#     feature_dict = {
#         feature_names[i]: round(song["features"][i], 3)
#         for i in range(len(feature_names))
#     }
#     st.write(f"{song['name']}: {feature_dict}")

# st.caption("If you can see this, Streamlit is working correctly.")
import streamlit as st
import math
import random
import copy
import pandas as pd
import requests
from io import StringIO

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Music Recommender System", 
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header with gradient */
    .main-header {
        background: linear-gradient(135deg, #1DB954 0%, #191414 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Song card styling */
    .song-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 8px solid #1DB954;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        animation: slideIn 0.5s ease-out;
    }
    
    .song-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(29, 185, 84, 0.2);
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Cluster box styling */
    .cluster-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
        padding: 1.2rem;
        border-radius: 15px;
        border: none;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .cluster-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Feature tag styling */
    .feature-tag {
        background: linear-gradient(135deg, #1DB95420 0%, #19141420 100%);
        color: #4a5568;
        padding: 0.4rem 1rem;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.2rem;
        border: 1px solid #1DB95440;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1DB954 0%, #191414 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 30px;
        width: 100%;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(29, 185, 84, 0.5);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    /* Divider styling */
    .custom-divider {
        background: linear-gradient(90deg, transparent, #1DB954, #191414, transparent);
        height: 2px;
        margin: 2rem 0;
    }
    
    /* Loading spinner */
    .loading-spinner {
        text-align: center;
        padding: 2rem;
        color: #1DB954;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header Section
# -----------------------------
st.markdown("""
    <div class="main-header">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎵 Spotify Music Recommender</h1>
        <p style="font-size: 1.2rem; opacity: 0.95; max-width: 600px; margin: 0 auto;">
            Discover real songs using Spotify's audio features
        </p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# Load Real Dataset
# -----------------------------
@st.cache_data
def load_music_dataset():
    """Load a real Spotify dataset"""
    try:
        # Using a publicly available Spotify dataset (Top 2000 songs)
        url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
        
        with st.spinner("Loading real Spotify dataset... 🎵"):
            df = pd.read_csv(url)
            
            # Select relevant features and clean data
            feature_columns = [
                'danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
            ]
            
            # Take a sample of 100 songs for better performance (you can adjust this)
            df_sample = df[['track_name', 'track_artist'] + feature_columns].dropna().head(100)
            
            # Create songs list in our required format
            songs = []
            for idx, row in df_sample.iterrows():
                # Combine artist and track name
                song_name = f"{row['track_name']} - {row['track_artist']}"
                
                # Create features list in order: Energy, Danceability, Tempo, Valence, 
                # Acousticness, Loudness, Instrumentalness, Speechiness
                features = [
                    row['energy'],           # Energy
                    row['danceability'],      # Danceability
                    row['tempo'],             # Tempo
                    row['valence'],           # Valence
                    row['acousticness'],      # Acousticness
                    row['loudness'],          # Loudness
                    row['instrumentalness'],  # Instrumentalness
                    row['speechiness']        # Speechiness
                ]
                
                songs.append({
                    "id": idx,
                    "name": song_name[:50] + "..." if len(song_name) > 50 else song_name,  # Truncate long names
                    "features": features
                })
            
            return songs, df_sample
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.info("Falling back to sample dataset...")
        
        # Fallback to sample dataset if loading fails
        fallback_songs = [
            {"id": 1, "name": "Blinding Lights - The Weeknd", "features": [0.8, 0.7, 171, 0.9, 0.1, -5, 0.0, 0.05]},
            {"id": 2, "name": "Dance Monkey - Tones and I", "features": [0.75, 0.85, 98, 0.85, 0.15, -6, 0.02, 0.04]},
            {"id": 3, "name": "Someone Like You - Adele", "features": [0.2, 0.3, 135, 0.2, 0.9, -12, 0.8, 0.1]},
            {"id": 4, "name": "Shape of You - Ed Sheeran", "features": [0.8, 0.8, 96, 0.9, 0.25, -4, 0.0, 0.08]},
            {"id": 5, "name": "Bohemian Rhapsody - Queen", "features": [0.4, 0.3, 144, 0.3, 0.7, -8, 0.05, 0.03]},
            {"id": 6, "name": "Despacito - Luis Fonsi", "features": [0.7, 0.8, 178, 0.8, 0.2, -5, 0.0, 0.06]},
        ]
        return fallback_songs, pd.DataFrame()

# Load the dataset
songs, original_df = load_music_dataset()
songs = copy.deepcopy(songs)  # Create a copy for normalization

# -----------------------------
# Feature Definitions
# -----------------------------
feature_names = [
    "Energy",
    "Danceability",
    "Tempo",
    "Valence",
    "Acousticness",
    "Loudness",
    "Instrumentalness",
    "Speechiness"
]

feature_emojis = {
    "Energy": "⚡",
    "Danceability": "💃",
    "Tempo": "🎵",
    "Valence": "😊",
    "Acousticness": "🎸",
    "Loudness": "🔊",
    "Instrumentalness": "🎻",
    "Speechiness": "🗣️"
}

feature_ranges = {
    "Energy": (0.0, 1.0),
    "Danceability": (0.0, 1.0),
    "Tempo": (60.0, 200.0),  # Typical tempo range
    "Valence": (0.0, 1.0),
    "Acousticness": (0.0, 1.0),
    "Loudness": (-60.0, 0.0),  # Wider range for Spotify
    "Instrumentalness": (0.0, 1.0),
    "Speechiness": (0.0, 1.0)
}

feature_descriptions = {
    "Energy": "Perceptual measure of intensity and activity (0-1)",
    "Danceability": "How suitable a track is for dancing (0-1)",
    "Tempo": "Overall estimated tempo in BPM (Beats Per Minute)",
    "Valence": "Musical positiveness (0=negative/sad, 1=positive/happy)",
    "Acousticness": "Confidence that track is acoustic (0-1)",
    "Loudness": "Overall loudness in decibels (dB)",
    "Instrumentalness": "Predicts if track contains no vocals (0-1)",
    "Speechiness": "Presence of spoken words (0-1)"
}

# -----------------------------
# Utilities
# -----------------------------
def normalize_features(dataset):
    feature_length = len(dataset[0]["features"])

    mins = [min(song["features"][i] for song in dataset) for i in range(feature_length)]
    maxs = [max(song["features"][i] for song in dataset) for i in range(feature_length)]

    for song in dataset:
        song["features"] = [
            (song["features"][i] - mins[i]) / (maxs[i] - mins[i])
            if maxs[i] != mins[i] else 0
            for i in range(feature_length)
        ]

    return mins, maxs


def normalize_query(query_features, mins, maxs):
    normalized = []
    for i in range(len(query_features)):
        if maxs[i] != mins[i]:
            value = (query_features[i] - mins[i]) / (maxs[i] - mins[i])
            # Clamp values to [0, 1] to avoid any floating point issues
            value = max(0.0, min(1.0, value))
        else:
            value = 0
        normalized.append(value)
    return normalized


def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def k_means(dataset, k=2, iters=10):
    # Use random centroids from the dataset
    centroids = random.sample([s["features"] for s in dataset], k)

    for _ in range(iters):
        clusters = {i: [] for i in range(k)}

        for song in dataset:
            idx = min(range(k),
                      key=lambda c: euclidean(song["features"], centroids[c]))
            clusters[idx].append(song)

        for i in range(k):
            if clusters[i]:
                centroids[i] = [
                    sum(song["features"][d] for song in clusters[i]) / len(clusters[i])
                    for d in range(len(dataset[0]["features"]))
                ]

    return clusters, centroids


def knn(query_features, cluster, k=3):
    ranked = sorted(
        [(euclidean(query_features, s["features"]), s)
         for s in cluster],
        key=lambda x: x[0]
    )
    return [s for _, s in ranked[:k]]

# -----------------------------
# Main Layout
# -----------------------------
col1, col2 = st.columns([1.2, 2], gap="large")

with col1:
    st.markdown("### 🎛️ Your Music Profile")
    st.markdown("Adjust the sliders to find songs that match your taste:")
    
    user_raw_features = []  # Store raw values for display
    user_normalized_features = []  # Store normalized values for processing
    
    # Create tabs for better organization
    input_tabs = st.tabs(["Main Features", "Advanced Features"])
    
    with input_tabs[0]:
        for feature in ["Energy", "Danceability", "Tempo", "Valence"]:
            min_val, max_val = feature_ranges[feature]
            
            # Set default value based on feature type
            if feature == "Tempo":
                default_val = 120.0
            else:
                default_val = 0.5
            
            # Use format spec to ensure consistent types
            step = 1.0 if feature == "Tempo" else 0.01
            
            st.markdown(f"**{feature_emojis[feature]} {feature}**")
            value = st.slider(
                label="",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=float(step),
                key=f"slider_{feature}",
                label_visibility="collapsed",
                format="%f" if step < 1 else "%d"
            )
            user_raw_features.append(value)
            
            # Store raw value (will be normalized later)
            user_normalized_features.append(value)
    
    with input_tabs[1]:
        for feature in ["Acousticness", "Loudness", "Instrumentalness", "Speechiness"]:
            min_val, max_val = feature_ranges[feature]
            
            # Set default value based on feature type
            if feature == "Loudness":
                default_val = -10.0
            else:
                default_val = 0.5
            
            # Use format spec to ensure consistent types
            step = 1.0 if feature == "Loudness" else 0.01
            
            st.markdown(f"**{feature_emojis[feature]} {feature}**")
            value = st.slider(
                label="",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=float(step),
                key=f"slider_{feature}",
                label_visibility="collapsed",
                format="%f" if step < 1 else "%d"
            )
            user_raw_features.append(value)
            
            # Store raw value (will be normalized later)
            user_normalized_features.append(value)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dataset info
    st.markdown(f"**📊 Dataset Stats**")
    st.markdown(f"Total songs: {len(songs)}")
    
    # Algorithm settings in an expander
    with st.expander("⚙️ Algorithm Settings", expanded=False):
        clusters_k = st.slider(
            "Number of Clusters (K-Means)", 
            min_value=2, 
            max_value=5, 
            value=3,
            step=1,
            help="More clusters = more granular grouping"
        )
        neighbors_k = st.slider(
            "Number of Neighbors (KNN)", 
            min_value=1, 
            max_value=min(10, len(songs)), 
            value=3,
            step=1,
            help="How many similar songs to recommend"
        )
    
    recommend_button = st.button("🎯 Get Personalized Recommendations", use_container_width=True)

with col2:
    if recommend_button:
        # Normalize the dataset
        mins, maxs = normalize_features(songs)
        
        # Normalize user features with clamping
        normalized_query = normalize_query(user_normalized_features, mins, maxs)
        
        # Run K-means
        clusters, centroids = k_means(songs, k=clusters_k)
        
        # Find closest cluster
        cluster_idx = min(range(len(centroids)),
                          key=lambda i: euclidean(normalized_query, centroids[i]))
        
        # Display user profile visualization
        st.markdown("### 🎨 Your Music Fingerprint!!")
        
        # Create feature progress bars - ensure values are in [0, 1]
        for i, feature in enumerate(feature_names):
            # Double-check value is in valid range
            progress_value = max(0.0, min(1.0, normalized_query[i]))
            
            col_feat, col_val, col_bar = st.columns([2, 1, 3])
            with col_feat:
                st.markdown(f"**{feature_emojis[feature]} {feature}**")
            with col_val:
                st.markdown(f"**{progress_value*100:.1f}%**")
            with col_bar:
                st.progress(progress_value)
        
        # Recommendations section
        st.markdown("---")
        st.markdown("### 🎧 Recommended For You")
        recs = knn(normalized_query, clusters[cluster_idx], k=neighbors_k)
        
        # Create recommendation cards in columns
        rec_cols = st.columns(len(recs))
        for idx, (col, rec) in enumerate(zip(rec_cols, recs)):
            with col:
                # Calculate match score
                distance = euclidean(normalized_query, rec["features"])
                match_score = max(0.0, min(100.0, 100 - (distance * 100)))
                
                # Extract artist and song name if possible
                song_parts = rec['name'].split(' - ')
                if len(song_parts) == 2:
                    song_name, artist = song_parts
                else:
                    song_name = rec['name']
                    artist = "Unknown Artist"
                
                st.markdown(f"""
                    <div class="song-card">
                        <h3 style="margin:0; color:#1DB954; font-size: 1.2rem;">{song_name}</h3>
                        <p style="margin:0.3rem 0; color: #718096; font-size: 0.9rem;">{artist}</p>
                        <div style="margin: 1rem 0;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                                <span>Match Score</span>
                                <span style="color: #1DB954; font-weight: 600;">{match_score:.1f}%</span>
                            </div>
                            <div style="background: #e9ecef; border-radius: 10px; height: 6px;">
                                <div style="background: linear-gradient(90deg, #1DB954, #191414); width: {match_score}%; height: 100%; border-radius: 10px;"></div>
                            </div>
                        </div>
                        <div style="margin-top: 1rem;">
                            <span class="feature-tag">{feature_emojis['Energy']} {rec['features'][0]:.2f}</span>
                            <span class="feature-tag">{feature_emojis['Danceability']} {rec['features'][1]:.2f}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Cluster information
        st.markdown("---")
        st.markdown("### 📊 Cluster Analysis")
        
        # Cluster metrics
        metric_cols = st.columns(clusters_k)
        for i, (col, (cluster_id, cluster_songs)) in enumerate(zip(metric_cols, clusters.items())):
            with col:
                is_user_cluster = cluster_id == cluster_idx
                if is_user_cluster:
                    st.success(f"**Cluster {cluster_id + 1}**\n\n{len(cluster_songs)} songs\n\n✅ Your cluster")
                else:
                    st.info(f"**Cluster {cluster_id + 1}**\n\n{len(cluster_songs)} songs")
        
        # Cluster composition (show first few songs in each cluster)
        st.markdown("#### Top Songs in Each Cluster")
        cluster_cols = st.columns(clusters_k)
        for i, (col, (cluster_id, cluster_songs)) in enumerate(zip(cluster_cols, clusters.items())):
            with col:
                with st.container():
                    st.markdown(f"**Cluster {cluster_id+1}**")
                    # Show first 3 songs in each cluster
                    for j, song in enumerate(cluster_songs[:3]):
                        song_short = song['name'][:25] + "..." if len(song['name']) > 25 else song['name']
                        st.markdown(f"{j+1}. 🎵 {song_short}")

# -----------------------------
# Dataset Section
# -----------------------------
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

with st.expander("📚 View Dataset Details", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Normalized Dataset", "Raw Data Sample", "Feature Info"])
    
    with tab1:
        st.markdown("### Dataset After Normalization")
        
        # Create DataFrame for display
        data_for_table = []
        for song in songs:
            row = {'Song': song['name'][:30] + "..." if len(song['name']) > 30 else song['name']}
            for i, feature in enumerate(feature_names):
                row[feature] = round(song['features'][i], 3)
            data_for_table.append(row)
        
        # Use Streamlit's native dataframe
        df = pd.DataFrame(data_for_table)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with tab2:
        if not original_df.empty:
            st.markdown("### Sample of Original Data")
            st.dataframe(original_df[['track_name', 'track_artist', 'danceability', 'energy', 'tempo']].head(10))
        else:
            st.info("Using sample dataset - no raw data available")
    
    with tab3:
        st.markdown("### Feature Descriptions")
        for feature, desc in feature_descriptions.items():
            with st.expander(f"{feature_emojis[feature]} {feature}"):
                st.write(desc)
                min_val, max_val = feature_ranges[feature]
                st.write(f"**Range:** {min_val} - {max_val}")

# Footer
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #718096;">
        <p style="font-size: 0.9rem;">
            🎵 Powered by Spotify audio features | 
            Data from TidyTuesday Spotify dataset |
            K-Means Clustering + KNN
        </p>
    </div>
""", unsafe_allow_html=True)

