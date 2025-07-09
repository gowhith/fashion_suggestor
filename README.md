# AI Fashion Recommender

An interactive, AI-powered fashion recommendation system that suggests clothing and accessories based on your style preferences. Built with Streamlit, FAISS, and state-of-the-art language models, this project demonstrates how machine learning can enhance personal style discovery.

---

## Features
- **Natural Language Input:** Describe your fashion preferences in plain English.
- **AI-Powered Recommendations:** Uses Sentence Transformers to understand your style and recommend matching items.
- **Large Fashion Dataset:** Draws from a curated dataset of 13,000+ fashion items, including images, categories, and descriptions.
- **Fast Vector Search:** Utilizes FAISS for efficient similarity search over item embeddings.
- **Interactive Web App:** User-friendly interface built with Streamlit.

---

## Demo
![Demo Screenshot](https://via.placeholder.com/800x400?text=Demo+Screenshot)

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/gowhith/fashion_suggestor.git
cd fashion_suggestor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the Data
- Ensure `data/fashion_items.json` is present (already included).
- The dataset contains fields like `id`, `name`, `description`, `category`, `subcategory`, `gender`, `article_type`, `color`, `season`, `year`, and `image_link`.

### 4. Build the Vector Index (First Time Only)
```bash
python build_index.py
```
This script encodes all item descriptions and builds a FAISS index for fast recommendations. The index is saved to `vector_store/faiss_index.faiss`.

### 5. Run the App
```bash
streamlit run app.py
```

---

## Project Structure
```
.
├── app.py                # Streamlit web app for recommendations
├── build_index.py        # Script to build the FAISS vector index
├── data/
│   └── fashion_items.json # Fashion items dataset (13,000+ items)
├── vector_store/
│   └── faiss_index.faiss  # Saved FAISS index (auto-generated)
├── requirements.txt      # Python dependencies
├── .gitignore            # Files and folders to ignore in git
├── LICENSE               # MIT License
├── README.md             # Project documentation
├── recommender.txt       # (Legacy/experimental) index builder script
├── test.txt              # Example/test fashion items (not used in app)
```

---

## Data
- **fashion_items.json:** Each entry contains:
  - `id`, `name`, `description`, `category`, `subcategory`, `gender`, `article_type`, `color`, `season`, `year`, `image_link`
- Example:
```json
{
  "id": "15970",
  "name": "Turtle Check Men Navy Blue Shirt",
  "description": "Casual",
  "category": "Apparel",
  "subcategory": "Topwear",
  "gender": "Men",
  "article_type": "Shirts",
  "color": "Navy Blue",
  "season": "Fall",
  "year": "2011.0",
  "image_link": "http://assets.myntassets.com/v1/images/style/properties/7a5b82d1372a7a5c6de67ae7a314fd91_images.jpg"
}
```

---

## How It Works
- The app loads the dataset and vector index.
- User describes their style (e.g., "I love oversized hoodies and streetwear").
- The app uses a Sentence Transformer to encode the input and find the closest matching items using FAISS.
- Top recommendations are displayed with images and details.

---

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

---

## Author
- [gowhith](https://github.com/gowhith) 
