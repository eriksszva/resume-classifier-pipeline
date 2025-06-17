from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc,
    accuracy_score, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score
)
from utils.SentenceTransformers import SentenceEmbeddingTransformer
from utils.PipelineWrapperModel import PipelineWrapperModel
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import mlflow.pyfunc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mlflow
import logging
import time
import os
import json


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# load dataset 
file_path = 'cleaned_data/resume_data_cleaned-labeled.csv'
df = pd.read_csv(file_path)
df_sampled = df.sample(n=500, random_state=42)
X = df_sampled['resume_text']
y = df_sampled['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow setup
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('Resume Classification')

# define pipeline
pipeline = Pipeline([
    ('embedder', SentenceEmbeddingTransformer(batch_size=128)),
    ('clf', LogisticRegression())
])

# define hyperparameter search space
param_distributions = {
    'clf__C': loguniform(1e-4, 1e4),
    'clf__penalty': ['l2'],
    'clf__solver': ['liblinear', 'saga'],
    'clf__max_iter': [200, 500, 1000]
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    cv=3,
    verbose=2,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42
)

with mlflow.start_run():

    # fit hyperparameter tuning
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    # predict & evaluate
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)
    y_train_proba = best_pipeline.predict_proba(X_train)
    y_test_proba = best_pipeline.predict_proba(X_test)

    # log parameters
    mlflow.log_params(search.best_params_)

    # log dataset info
    mlflow.log_param("dataset_size", len(df_sampled))
    mlflow.log_param("num_classes", len(np.unique(y)))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))

    # log training metrics
    mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_train_pred))
    mlflow.log_metric("train_f1", f1_score(y_train, y_train_pred, average='weighted'))
    mlflow.log_metric("train_log_loss", log_loss(y_train, y_train_proba))
    mlflow.log_metric("train_precision", precision_score(y_train, y_train_pred, average='weighted'))
    mlflow.log_metric("train_recall", recall_score(y_train, y_train_pred, average='weighted'))
    mlflow.log_metric("train_roc_auc", roc_auc_score(y_train, y_train_proba, multi_class='ovr'))

    # log testing metrics
    mlflow.log_metric("test_log_loss", log_loss(y_test, y_test_proba))
    mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_test_proba, multi_class='ovr'))
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)
        else:
            mlflow.log_metric(f"{label}", metrics)
    
    # === artifacts ===
    os.makedirs("artifacts", exist_ok=True)

    def save_plot(fig, path):
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    # confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues")
    plt.title('Training Confusion Matrix')
    save_plot(fig, 'artifacts/training_confusion_matrix.png')

    cm_test = confusion_matrix(y_test, y_test_pred)
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Oranges")
    plt.title('Testing Confusion Matrix')
    save_plot(fig, 'artifacts/testing_confusion_matrix.png')

    # precision-recall curves
    fig = plt.figure(figsize=(6, 4))
    for i in range(y_train_proba.shape[1]):
        precision, recall, _ = precision_recall_curve(y_train == i, y_train_proba[:, i])
        plt.plot(recall, precision, label=f"Class {i}")
    plt.title('Training Precision-Recall Curve')
    plt.legend()
    save_plot(fig, 'artifacts/training_precision_recall.png')

    fig = plt.figure(figsize=(6, 4))
    for i in range(y_test_proba.shape[1]):
        precision, recall, _ = precision_recall_curve(y_test == i, y_test_proba[:, i])
        plt.plot(recall, precision, label=f"Class {i}")
    plt.title('Testing Precision-Recall Curve')
    plt.legend()
    save_plot(fig, 'artifacts/testing_precision_recall.png')

    # ROC curves
    fig = plt.figure(figsize=(6, 4))
    for i in range(y_train_proba.shape[1]):
        fpr, tpr, _ = roc_curve(y_train == i, y_train_proba[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Training ROC Curve')
    plt.legend()
    save_plot(fig, 'artifacts/training_roc_curve.png')

    fig = plt.figure(figsize=(6, 4))
    for i in range(y_test_proba.shape[1]):
        fpr, tpr, _ = roc_curve(y_test == i, y_test_proba[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Testing ROC Curve')
    plt.legend()
    save_plot(fig, 'artifacts/testing_roc_curve.png')

    # save metrics to JSON
    metrics_summary = {
        "train": {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "f1": f1_score(y_train, y_train_pred, average='weighted'),
            "log_loss": log_loss(y_train, y_train_proba),
            "precision": precision_score(y_train, y_train_pred, average='weighted'),
            "recall": recall_score(y_train, y_train_pred, average='weighted'),
            "roc_auc": roc_auc_score(y_train, y_train_proba, multi_class='ovr')
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred, average='weighted'),
            "log_loss": log_loss(y_test, y_test_proba),
            "precision": precision_score(y_test, y_test_pred, average='weighted'),
            "recall": recall_score(y_test, y_test_pred, average='weighted'),
            "roc_auc": roc_auc_score(y_test, y_test_proba, multi_class='ovr')
        }
    }
    with open('artifacts/metric_info.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    # save pipeline HTML
    from sklearn import set_config
    set_config(display='diagram')
    with open('artifacts/estimator.html', 'w') as f:
        f.write(best_pipeline._repr_html_())

    # Upload artifacts to MLflow
    mlflow.log_artifact('artifacts/estimator.html')
    mlflow.log_artifact('artifacts/metric_info.json')
    mlflow.log_artifact('artifacts/training_confusion_matrix.png')
    mlflow.log_artifact('artifacts/testing_confusion_matrix.png')
    mlflow.log_artifact('artifacts/training_precision_recall.png')
    mlflow.log_artifact('artifacts/testing_precision_recall.png')
    mlflow.log_artifact('artifacts/training_roc_curve.png')
    mlflow.log_artifact('artifacts/testing_roc_curve.png')


    # log model (wrapped for custom transformer)
    wrapped_model = PipelineWrapperModel(best_pipeline)
    mlflow.pyfunc.log_model(
        artifact_path='custom_model',
        python_model=wrapped_model,
        input_example=pd.DataFrame(X_test.head(1)),
        conda_env=mlflow.sklearn.get_default_conda_env()
    )

    # print classification report (for inspection)
    print(classification_report(y_test, y_test_pred))
    print(f'Accuracy: {pipeline.score(X_test, y_test)}')