from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from app.schema import EmbedRequest
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# Load pre-downloaded model
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/embed")
def embed(req: EmbedRequest):
    try:
        logging.info("Received request for embedding texts.")
        if not req.texts:
            logging.warning("Received empty list of texts.")
            return {"embeddings": []}
        vectors = model.encode(req.texts, batch_size=64)
        logging.info(f"Embedding output shape: {vectors.shape}")
        return {"embeddings": vectors.tolist()}
    except Exception as e:
        logging.error(f"Error during embedding: {e}")
        return {"error": str(e)}