import pandas as pd
import joblib
import requests
from fastapi import FastAPI
from app.schema import InferenceRequest
import logging

# initialize logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()
model = joblib.load("app/model.joblib")
EMBEDDING_API_URL = "http://host.docker.internal:5000/embed" # embedder API URL in local development

@app.post("/predict")
def predict(request: InferenceRequest):
    try:
        logging.info("Received prediction request.")
        if not request.data or not request.columns:
            logging.warning("Received empty data or columns.")
            return {"predictions": []}
        df = pd.DataFrame(request.data, columns=request.columns)
        texts = df["resume_text"].tolist()

        # call external embedder
        response = requests.post(
            EMBEDDING_API_URL,
            json={"texts": texts}
        )
        embeddings = response.json()["embeddings"]

        preds = model.predict(embeddings)
        logging.info(f"Predictions made: {preds}")
        return {"predictions": preds.tolist()}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"error": str(e)}