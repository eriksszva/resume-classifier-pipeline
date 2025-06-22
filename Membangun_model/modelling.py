from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils.SentenceTransformers import SentenceEmbeddingTransformer
import mlflow.sklearn
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# data
file_path = 'cleaned_data/resume_data_cleaned-labeled.csv'
df = pd.read_csv(file_path)
df_sampled = df.sample(n=500, random_state=42)
X = df_sampled['resume_text']
y = df_sampled['label'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# tracking setup
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('Resume Classification')
mlflow.sklearn.autolog()

with mlflow.start_run():
    pipeline = Pipeline([
        ('embedder', SentenceEmbeddingTransformer(batch_size=128)),
        ('clf', LogisticRegression(max_iter=100))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # log classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)
        else:
            mlflow.log_metric(f"{label}", metrics)
    
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {pipeline.score(X_test, y_test)}')

