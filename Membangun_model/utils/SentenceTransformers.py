from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import hashlib
import joblib
import os

class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2', batch_size=128, cache_dir='cache'):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory = joblib.Memory(location=self.cache_dir, verbose=0)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_list = X.tolist() if hasattr(X, 'tolist') else list(X)
        # hash from input as cache key
        input_hash = hashlib.md5("".join(X_list).encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{input_hash}.npy")

        if os.path.exists(cache_file):
            print('Loaded cached embeddings.')
            return np.load(cache_file)

        # if not cached, perform encoding
        start_time = time.time()
        embeddings = self.model.encode(X_list, batch_size=self.batch_size)
        print(f'Embeddings done in {time.time() - start_time:.2f}s')

        # save to cache
        np.save(cache_file, embeddings)
        return embeddings
