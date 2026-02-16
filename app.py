import streamlit as st
import math, random, copy, pandas as pd

st.set_page_config(page_title="Music Recommender System",
                   page_icon="🎵", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
*{font-family:'Inter',sans-serif}

.main-header{
 background:linear-gradient(135deg,#1DB954,#191414);
 padding:2rem;border-radius:20px;color:white;text-align:center;margin-bottom:2rem}

.song-card{
 background:white;
 padding:1.5rem;
 border-radius:15px;
 margin:.5rem 0;
 border-left:8px solid #1DB954;
 box-shadow:0 4px 15px rgba(0,0,0,.05);
 color:#191414;
}

.feature-tag{
 background:rgba(29,185,84,0.15);
 color:#191414;
 padding:.3rem .8rem;
 border-radius:20px;
 font-size:.75rem;
 margin:.2rem;
 display:inline-block;
 font-weight:500;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
<h1>🎵 Spotify Music Recommender</h1>
<p>Discover real songs using Spotify audio features</p>
</div>
""", unsafe_allow_html=True)

# ---------- Dataset ----------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_sample.csv")

    cols = ['danceability','energy','loudness','speechiness',
            'acousticness','instrumentalness','liveness','valence','tempo']

    df = df[['track_name','artists'] + cols].dropna()

    songs = [{
        "id": i,
        "name": f"{r.track_name} - {r.artists}"[:60],
        "features": [
            r.energy,
            r.danceability,
            r.tempo,
            r.valence,
            r.acousticness,
            r.loudness,
            r.instrumentalness,
            r.speechiness
        ]
    } for i, r in df.iterrows()]

    return songs, df

# 🔥 THIS WAS MISSING
songs, original_df = load_data()
songs = copy.deepcopy(songs)

# ---------- Feature Definitions ----------
features = ["Energy","Danceability","Tempo","Valence",
            "Acousticness","Loudness","Instrumentalness","Speechiness"]

emojis = ["⚡","💃","🎵","😊","🎸","🔊","🎻","🗣️"]

ranges = {
 "Energy":(0,1),"Danceability":(0,1),"Tempo":(60,200),"Valence":(0,1),
 "Acousticness":(0,1),"Loudness":(-60,0),
 "Instrumentalness":(0,1),"Speechiness":(0,1)
}

# ---------- ML Utils ----------
def normalize(data):
    mins=[min(s["features"][i] for s in data) for i in range(8)]
    maxs=[max(s["features"][i] for s in data) for i in range(8)]
    for s in data:
        s["features"]=[(s["features"][i]-mins[i])/(maxs[i]-mins[i])
                       if maxs[i]!=mins[i] else 0 for i in range(8)]
    return mins,maxs

def norm_query(q,mins,maxs):
    return [max(0,min(1,(q[i]-mins[i])/(maxs[i]-mins[i])
            if maxs[i]!=mins[i] else 0)) for i in range(8)]

def dist(a,b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(8)))

def kmeans(data,k,iters=10):
    cent=random.sample([s["features"] for s in data],k)
    for _ in range(iters):
        cl={i:[] for i in range(k)}
        for s in data:
            idx=min(range(k),key=lambda c:dist(s["features"],cent[c]))
            cl[idx].append(s)
        for i in cl:
            if cl[i]:
                cent[i]=[sum(s["features"][d] for s in cl[i])/len(cl[i])
                         for d in range(8)]
    return cl,cent

def knn(q,cluster,k):
    return sorted(cluster,key=lambda s:dist(q,s["features"]))[:k]

# ---------- Layout ----------
col1,col2=st.columns([1,2])

with col1:
    st.markdown("### 🎛️ Your Music Profile")
    user=[]
    for f in features:
        minv,maxv=ranges[f]
        default=120 if f=="Tempo" else (-10 if f=="Loudness" else .5)
        step=1 if f in ["Tempo","Loudness"] else .01
        user.append(st.slider(f, float(minv),float(maxv),float(default),float(step)))

    clusters_k=st.slider("K-Means Clusters",2,5,3)
    neighbors_k=st.slider("K-Neighbors",1,10,3)
    go=st.button("🎯 Get Recommendations")

with col2:
    if go:
        mins,maxs=normalize(songs)
        q=norm_query(user,mins,maxs)
        clusters,cent=kmeans(songs,clusters_k)
        idx=min(range(clusters_k),key=lambda i:dist(q,cent[i]))
        recs=knn(q,clusters[idx],neighbors_k)

        # Music Fingerprint
        st.markdown("### 🎨 Your Music Fingerprint")
        for i,f in enumerate(features):
            st.progress(q[i])
            st.caption(f"{emojis[i]} {f}: {q[i]*100:.1f}%")

        # Recommendations
        st.markdown("### 🎧 Recommended For You")
        cols=st.columns(len(recs))
        for c,r in zip(cols,recs):
            with c:
                score=max(0,100-dist(q,r["features"])*100)
                tags="".join([
                    f'<span class="feature-tag">{emojis[i]} {r["features"][i]:.2f}</span>'
                    for i in range(8)
                ])
                st.markdown(f"""
                <div class="song-card">
                <h3>{r['name']}</h3>
                <p style="color:#1DB954;font-weight:600">Match: {score:.1f}%</p>
                <div>{tags}</div>
                </div>
                """,unsafe_allow_html=True)

        # Cluster Analysis
        st.markdown("### 📊 Cluster Analysis")
        cols=st.columns(clusters_k)
        for i,(c,cluster) in enumerate(zip(cols,clusters.values())):
            with c:
                if i==idx:
                    st.success(f"Cluster {i+1}\n{len(cluster)} songs\nYour cluster")
                else:
                    st.info(f"Cluster {i+1}\n{len(cluster)} songs")

        st.markdown("### Top Songs in Each Cluster")
        cols=st.columns(clusters_k)
        for i,(c,cluster) in enumerate(zip(cols,clusters.values())):
            with c:
                st.markdown(f"**Cluster {i+1}**")
                for j,s in enumerate(cluster[:3]):
                    st.write(f"{j+1}. 🎵 {s['name'][:30]}")

# Dataset Viewer
with st.expander("📚 View Dataset"):
    df=pd.DataFrame([{"Song":s["name"],**{
        features[i]:round(s["features"][i],3)
        for i in range(8)}} for s in songs])
    st.dataframe(df,use_container_width=True)

st.caption("🎵 Spotify audio features | K-Means + KNN")
