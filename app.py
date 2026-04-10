
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved data
df = pickle.load(open('df.pkl', 'rb'))
user_item_matrix = pickle.load(open('user_item_matrix.pkl', 'rb'))
content_sim_df = pickle.load(open('content_sim.pkl', 'rb'))

# Page setup
st.set_page_config(page_title="Course Recommender", page_icon="📚", layout="wide")
st.title("📚 Online Course Recommendation System")
st.markdown("---")

# Sidebar to pick the algorithm
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
    st.subheader("🔥 Most Popular Courses")
    popular = df['course_name'].value_counts().head(top_n).index.tolist()
    for i, course in enumerate(popular, 1):
        st.write(f"{i}. {course}")

# 2. Content-Based
elif method == "Content-Based":
    st.subheader("📖 Content-Based Recommendations")
    course_list = content_sim_df.index.tolist()
    selected = st.selectbox("Pick a course you liked:", course_list)
    if st.button("Recommend"):
        recs = content_sim_df[selected].sort_values(ascending=False).iloc[1:top_n+1].index.tolist()
        for i, c in enumerate(recs, 1):
            st.write(f"{i}. {c}")

# 3. User-Based
elif method == "User-Based CF":
    st.subheader("👥 User-Based Recommendations")
    user_id = st.selectbox("Select User ID:", user_item_matrix.index.tolist())
    if st.button("Recommend"):
        user_sim = pd.DataFrame(cosine_similarity(user_item_matrix), index=user_item_matrix.index, columns=user_item_matrix.index)
        sim_users = user_sim[user_id].sort_values(ascending=False).iloc[1:6].index
        scores = user_item_matrix.loc[sim_users].mean().sort_values(ascending=False)
        taken = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
        recs = scores.drop(taken, errors='ignore').head(top_n).index.tolist()
        for i, c in enumerate(recs, 1):
            st.write(f"{i}. {c}")

# 4. Item-Based
elif method == "Item-Based CF":
    st.subheader("📦 Item-Based Recommendations")
    user_id = st.selectbox("Select User ID:", user_item_matrix.index.tolist())
    if st.button("Recommend"):
        item_sim = pd.DataFrame(cosine_similarity(user_item_matrix.T), index=user_item_matrix.columns, columns=user_item_matrix.columns)
        scores = item_sim.dot(user_item_matrix.loc[user_id]).sort_values(ascending=False)
        taken = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
        recs = scores.drop(taken, errors='ignore').head(top_n).index.tolist()
        for i, c in enumerate(recs, 1):
            st.write(f"{i}. {c}")

# 5. Hybrid
elif method == "Hybrid":
    st.subheader("🔀 Hybrid Recommendations")
    user_id = st.selectbox("Select User ID:", user_item_matrix.index.tolist())
    course_name = st.selectbox("Pick a course you liked:", content_sim_df.index.tolist())
    if st.button("Recommend"):
        content_scores = content_sim_df[course_name]
        user_sim = pd.DataFrame(cosine_similarity(user_item_matrix), index=user_item_matrix.index, columns=user_item_matrix.index)
        sim_users = user_sim[user_id].sort_values(ascending=False).iloc[1:6].index
        user_scores = user_item_matrix.loc[sim_users].mean()
        # Align both scores to same index
        common = content_scores.index.intersection(user_scores.index)
        hybrid = (content_scores[common] * 0.5) + (user_scores[common] * 0.5)
        taken = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
        recs = hybrid.drop(taken, errors='ignore').sort_values(ascending=False).head(top_n).index.tolist()
        for i, c in enumerate(recs, 1):
            st.write(f"{i}. {c}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
