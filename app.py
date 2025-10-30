import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import base64

#Page Configuration
st.set_page_config(page_title="Movie Recommender", layout="wide")

#Title
st.title("🎬 Movie Recommendation System")
st.write("Get personalized movie recommendations!")

#Background 
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("BrandAssets_Logos_02-NSymbol.jpg")

#Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_data.csv")
    df = df[['title', 'listed_in', 'description']].dropna()
    df['text'] = df['listed_in'] + " " + df['description']
    return df

df = load_data()

#Feature extraction
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])

    # Reduce dimensionality for clustering
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X.toarray())

    # Cluster movies
    kmeans = KMeans(n_clusters=10, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_pca)

    return vectorizer, pca, kmeans, X, X_pca, df

vectorizer, pca, kmeans, X, X_pca, df = train_model(df)

#Recommenation Function
def recommend_movie(title, n=5):
    if title not in df['title'].values:
        st.warning("Movie not found in dataset.")
        return pd.DataFrame()

    idx = df[df['title'] == title].index[0]
    movie_cluster = df.loc[idx, 'cluster']
    
    # Get all movies in the same cluster
    cluster_movies = df[df['cluster'] == movie_cluster]
    cluster_indices = cluster_movies.index

    # Compute similarity
    sim = cosine_similarity(X[idx], X[cluster_indices]).flatten()
    similar_indices = cluster_indices[sim.argsort()[-n-1:-1][::-1]]

    return df.loc[similar_indices, ['title', 'listed_in', 'description']]

# UI
selected_movie = st.selectbox("🎥 Select a movie:", df['title'].sort_values().unique())
num_recs = st.slider("Number of recommendations:", 3, 10, 5)

if st.button("🔍 Recommend"):
    recs = recommend_movie(selected_movie, n=num_recs)
    if not recs.empty:
        st.subheader(f"Movies similar to **{selected_movie}**:")
        for _, row in recs.iterrows():
            st.markdown(f"**🎞 {row['title']}**")
            st.caption(f"🧩 Genres: {row['listed_in']}")
            st.write(row['description'])
            st.markdown("---")
