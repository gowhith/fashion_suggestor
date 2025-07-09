# Fashion Suggestor

This project is a fashion item recommender system. It uses a dataset of fashion items and a vector store for recommendations.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the `data/fashion_items.json` file is present.
3. Run the application:
   ```bash
   python app.py
   ```

## Project Structure
- `app.py`: Main application file
- `build_index.py`: Script to build the vector index
- `data/`: Contains the fashion items dataset
- `vector_store/`: Stores the FAISS index (ignored in git)

## License
MIT 