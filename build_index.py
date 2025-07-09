import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import os

DATA_FILE = "data/fashion_items.json"
INDEX_FILE = "vector_store/faiss_index.faiss"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open(DATA_FILE, "r") as f:
    items = json.load(f)

descriptions = [item["description"] for item in items]
vectors = model.encode(descriptions, convert_to_numpy=True).astype("float32")

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)
faiss.write_index(index, INDEX_FILE)
print(f"Indexed {len(descriptions)} items to {INDEX_FILE}")