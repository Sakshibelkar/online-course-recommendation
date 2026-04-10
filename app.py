import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data 
@st.cache_data
def load_data():
    df = pd.read_excel('online_course_recommendation_v2 (1).xlsx')
    df = df.drop_duplicates().dropna()
    return df

df = load_data()

# Build matrices (cached so it only runs once)
@st.cache_data
def build_matrices(df):
    # User-Item Matrix
    user_item = df.pivot_table(index='user_id', columns='course_name', values='rating').fillna(0)
    
    # TF-IDF for Content-Based
    tfidf = TfidfVectorizer(stop_words='english')
    unique = df.drop_duplicates(subset=['course_name']).copy()
    tfidf_matrix = tfidf.fit_transform(unique['course_name'].fillna(''))
    content_sim = pd.DataFrame(linear_kernel(tfidf_matrix, tfidf_matrix),
                               index=unique['course_name'], columns=unique['course_name'])
    return user_item, content_sim

user_item_matrix, content_sim_df = build_matrices(df)

# Page setup
st.set_page_config(page_title="Course Recommender", page_icon="📚", layout="wide")
st.title(" Online Course Recommendation System")
st.markdown("---")

# Sidebar
method = st.sidebar.selectbox("Choose Algorithm", [
    "Popularity-Based",
    "Content-Based",
    "User-Based CF",
    "Item-Based CF",
    "Hybrid"
])
top_n = st.sidebar.slider("Number of Recommendations", 3, 10, 5)

# 1. Popularity
if method == "Popularity-Based":
    st.subheader(" Most Popular Courses")
    popular = df['course_name'].value_counts().head(top_n).index.tolist()
    for i, course in enumerate(popular, 1):
        st.write(f"{i}. {course}")

# 2. Content-Based
elif method == "Content-Based":
    st.subheader(" Content-Based Recommendations")
    selected = st.selectbox("Pick a course you liked:", content_sim_df.index.tolist())
    if st.button("Recommend"):
        recs = content_sim_df[selected].sort_values(ascending=False).iloc[1:top_n+1].index.tolist()
        for i, c in enumerate(recs, 1):
            st.write(f"{i}. {c}")

# 3. User-Based
elif method == "User-Based CF":
    st.subheader(" User-Based Recommendations")
    user_id = st.selectbox("Select User ID:", user_item_matrix.index.tolist())
    if st.button("Recommend"):
        user_sim = pd.DataFrame(cosine_similarity(user_item_matrix),
                                index=user_item_matrix.index, columns=user_item_matrix.index)
        sim_users = user_sim[user_id].sort_values(ascending=False).iloc[1:6].index
        scores = user_item_matrix.loc[sim_users].mean().sort_values(ascending=False)
        taken = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
        recs = scores.drop(taken, errors='ignore').head(top_n).index.tolist()
        for i, c in enumerate(recs, 1):
            st.write(f"{i}. {c}")

# 4. Item-Based
elif method == "Item-Based CF":
    st.subheader(" Item-Based Recommendations")
    user_id = st.selectbox("Select User ID:", user_item_matrix.index.tolist())
    if st.button("Recommend"):
        item_sim = pd.DataFrame(cosine_similarity(user_item_matrix.T),
                                index=user_item_matrix.columns, columns=user_item_matrix.columns)
        scores = item_sim.dot(user_item_matrix.loc[user_id]).sort_values(ascending=False)
        taken = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
        recs = scores.drop(taken, errors='ignore').head(top_n).index.tolist()
        for i, c in enumerate(recs, 1):
            st.write(f"{i}. {c}")

# 5. Hybrid
elif method == "Hybrid":
    st.subheader(" Hybrid Recommendations")
    user_id = st.selectbox("Select User ID:", user_item_matrix.index.tolist())
    course_name = st.selectbox("Pick a course you liked:", content_sim_df.index.tolist())
    if st.button("Recommend"):
        content_scores = content_sim_df[course_name]
        user_sim = pd.DataFrame(cosine_similarity(user_item_matrix),
                                index=user_item_matrix.index, columns=user_item_matrix.index)
        sim_users = user_sim[user_id].sort_values(ascending=False).iloc[1:6].index
        user_scores = user_item_matrix.loc[sim_users].mean()
        common = content_scores.index.intersection(user_scores.index)
        hybrid = (content_scores[common] * 0.5) + (user_scores[common] * 0.5)
        taken = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
        recs = hybrid.drop(taken, errors='ignore').sort_values(ascending=False).head(top_n).index.tolist()
        for i, c in enumerate(recs, 1):
            st.write(f"{i}. {c}")

st.markdown("---")
st.caption("Built with using Streamlit")
