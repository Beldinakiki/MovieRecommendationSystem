# MovieRecommendationSystem

A content-based movie recommendation system built with Python, Streamlit, and scikit-learn, using Netflix movie data.

## Features

- Interactive web app for personalized movie recommendations
- Clustering movies by genre and description using TF-IDF, PCA, and KMeans
- Visualizations: genre distribution, word cloud, cluster analysis
- Select a movie and get similar recommendations instantly

## How It Works

1. **Data Loading**: Reads and preprocesses netflix_data.csv.
2. **Feature Extraction**: Combines genres and descriptions, vectorizes text, reduces dimensionality.
3. **Clustering**: Groups movies into clusters based on content similarity.
4. **Recommendation**: Finds movies most similar to your selection using cosine similarity.

## Usage

1. Install requirements:
    ```
    pip install streamlit pandas scikit-learn matplotlib seaborn wordcloud
    ```
2. Run the app:
    ```
    streamlit run app.py
    ```
3. Open the web interface, select a movie, and view recommendations.

## Files

- `app.py`: Streamlit web app
- `engine.ipynb`: Data analysis and model development notebook
- `netflix_data.csv`: Movie dataset
- `README.md`: Project documentation
