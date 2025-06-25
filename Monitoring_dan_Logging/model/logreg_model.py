""" 
This script different from 'Membangun_model' directory.
It trains a Logistic Regression model on resume data, embed the text using a pre-trained SentenceTransformer model separately.
It saves the trained model to a directory without including the embedder.
This approach allows for more efficient model deployment in a dockerized environment.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import joblib

# prepare data
df = pd.read_csv("cleaned_data/resume_data_cleaned-labeled.csv")
df_sampled = df.sample(n=500, random_state=42)
X = df_sampled["resume_text"]
y = df_sampled["label"].astype(int)

# embed only once
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_embeddings = embedder.encode(X.tolist(), batch_size=128, show_progress_bar=True)

# train classifier
clf = LogisticRegression(max_iter=100)
clf.fit(X_embeddings, y)

# save model only (no embedder)
joblib.dump(clf, "../app/model.joblib")
print("Logistic Regression model saved.")