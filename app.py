import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Configuration
DATA_FILE = "data/fashion_items.json"
INDEX_FILE = "vector_store/faiss_index.faiss"
KNOWN_ARTICLE_TYPES = [
    "Shirts", "Tshirts", "Jeans", "Hoodies", "Sweaters", "Jackets",
    "Dresses", "Skirts", "Shorts", "Tops", "Track Pants", "Watches", "Shoes"
]

st.set_page_config(
    page_title="AI Fashion Recommender",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    with open(DATA_FILE, "r") as f:
        items = json.load(f)
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        index = None
    return items, index

def extract_article_type(user_input, model, known_types):
    user_vec = model.encode([user_input], convert_to_numpy=True).astype("float32")
    type_vecs = model.encode(known_types, convert_to_numpy=True).astype("float32")

    similarities = np.dot(type_vecs, user_vec.T).flatten()
    
    # 💡 Debug log
    print("LLM Similarity with article types:")
    for i, sim in enumerate(similarities):
        print(f"{known_types[i]}: {sim:.4f}")

    best_idx = int(np.argmax(similarities))
    if similarities[best_idx] > 0.3:
        return known_types[best_idx]
    return None


def display_fashion_item(item, col):
    with col:
        st.image(item.get("image_link", "https://via.placeholder.com/200x200"), width=200)
        st.markdown(f"**{item['name']}**")
        st.caption(item.get("description", ""))
        st.caption(f"Type: {item.get('article_type', 'N/A')}, Color: {item.get('color', 'N/A')}")

def main():
    st.title("AI-Powered Fashion Style Recommender")

    model = load_model()
    items, index = load_data()

    user_input = st.text_input("🌈 Describe your fashion preference:",
        placeholder="e.g., I love oversized hoodies and streetwear style")

    if user_input:
        st.markdown("---")
        st.subheader("✨ Your Recommended Styles")

    # Extract article type using LLM
    article_type = extract_article_type(user_input, model, KNOWN_ARTICLE_TYPES)
    filtered_items = items  # default

    if article_type:
        filtered_items = [
            item for item in items if item["article_type"].lower() == article_type.lower()
        ]
        st.info(f"🎯 Matching article type: **{article_type}** ({len(filtered_items)} items found)")
    else:
        st.warning("No specific article type detected — showing general style matches.")

    if not filtered_items:
        st.warning("No matching items found for that article type.")
        return

    # Encode descriptions of filtered items
    descriptions = [item["description"] for item in filtered_items]
    vectors = model.encode(descriptions, convert_to_numpy=True).astype("float32")

    # Create temp index and search
    temp_index = faiss.IndexFlatL2(vectors.shape[1])
    temp_index.add(vectors)

    user_vec = model.encode([user_input], convert_to_numpy=True).astype("float32")
    D, I = temp_index.search(user_vec, min(6, len(filtered_items)))

    # Display recommendations
    cols = st.columns(3)
    for i, idx in enumerate(I[0]):
        if idx < len(filtered_items):
            display_fashion_item(filtered_items[idx], cols[i % 3])

    


if __name__ == "__main__":
    main()
