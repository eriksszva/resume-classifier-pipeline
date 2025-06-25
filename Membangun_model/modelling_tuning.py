from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc,
    accuracy_score, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score
)
from utils.SentenceTransformers import SentenceEmbeddingTransformer
from utils.PipelineWrapperModel import PipelineWrapperModel
from mlflow.models.signature import infer_signature
from mlflow.models import infer_pip_requirements
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import mlflow.pyfunc
import pandas as pd
import numpy as np
import seaborn as sns
import dagshub
import shutil
import mlflow
import logging
import time
import yaml
import os
import json

# setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# define a function to save ROC AUC score
def save_roc_auc_score(y_true, y_proba):
    if len(np.unique(y_true)) == 2:
        return roc_auc_score(y_true, y_proba[:, 1])
    else:
        return roc_auc_score(y_true, y_proba, multi_class='ovr')
  
# define a function to save plots  
def save_plot(fig, path):
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        
        
# prepare dataset 
file_path = 'cleaned_data/resume_data_cleaned-labeled.csv'
df = pd.read_csv(file_path)
df_sampled = df.sample(n=500, random_state=42)
X = df_sampled['resume_text']
y = df_sampled['label'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dagshub setup
load_dotenv()  # read .env file
dagshub.init(
    repo_owner='erikssssszz',
    repo_name='resume-classification-mlflow',
    mlflow=True
    )

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

with mlflow.start_run(run_name="Logistic Regression Tuning"):

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
    mlflow.log_metric("train_roc_auc", save_roc_auc_score(y_train, y_train_proba))

    # log testing metrics
    mlflow.log_metric("test_log_loss", log_loss(y_test, y_test_proba))
    mlflow.log_metric("test_roc_auc", save_roc_auc_score(y_test, y_test_proba))
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)
        else:
            mlflow.log_metric(f"{label}", metrics)
    
    # make Artifacts directory
    os.makedirs("Artifacts", exist_ok=True)

    # confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues")
    plt.title('Training Confusion Matrix')
    save_plot(fig, 'Artifacts/training_confusion_matrix.png')

    cm_test = confusion_matrix(y_test, y_test_pred)
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Oranges")
    plt.title('Testing Confusion Matrix')
    save_plot(fig, 'Artifacts/testing_confusion_matrix.png')

    # precision-recall curves
    fig = plt.figure(figsize=(6, 4))
    for i in range(y_train_proba.shape[1]):
        precision, recall, _ = precision_recall_curve(y_train == i, y_train_proba[:, i])
        plt.plot(recall, precision, label=f"Class {i}")
    plt.title('Training Precision-Recall Curve')
    plt.legend()
    save_plot(fig, 'Artifacts/training_precision_recall.png')

    fig = plt.figure(figsize=(6, 4))
    for i in range(y_test_proba.shape[1]):
        precision, recall, _ = precision_recall_curve(y_test == i, y_test_proba[:, i])
        plt.plot(recall, precision, label=f"Class {i}")
    plt.title('Testing Precision-Recall Curve')
    plt.legend()
    save_plot(fig, 'Artifacts/testing_precision_recall.png')

    # ROC curves
    fig = plt.figure(figsize=(6, 4))
    for i in range(y_train_proba.shape[1]):
        fpr, tpr, _ = roc_curve(y_train == i, y_train_proba[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Training ROC Curve')
    plt.legend()
    save_plot(fig, 'Artifacts/training_roc_curve.png')

    fig = plt.figure(figsize=(6, 4))
    for i in range(y_test_proba.shape[1]):
        fpr, tpr, _ = roc_curve(y_test == i, y_test_proba[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Testing ROC Curve')
    plt.legend()
    save_plot(fig, 'Artifacts/testing_roc_curve.png')

    # save metrics to JSON
    metrics_summary = {
        "train": {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "f1": f1_score(y_train, y_train_pred, average='weighted'),
            "log_loss": log_loss(y_train, y_train_proba),
            "precision": precision_score(y_train, y_train_pred, average='weighted'),
            "recall": recall_score(y_train, y_train_pred, average='weighted'),
            "roc_auc": save_roc_auc_score(y_train, y_train_proba)
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred, average='weighted'),
            "log_loss": log_loss(y_test, y_test_proba),
            "precision": precision_score(y_test, y_test_pred, average='weighted'),
            "recall": recall_score(y_test, y_test_pred, average='weighted'),
            "roc_auc": save_roc_auc_score(y_train, y_train_proba)
        }
    }
    with open('Artifacts/metric_info.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    # save pipeline HTML
    from sklearn import set_config
    set_config(display='diagram')
    with open('Artifacts/estimator.html', 'w', encoding='utf-8') as f:
        f.write(best_pipeline._repr_html_())

    # log Artifacts
    mlflow.log_artifacts('Artifacts')

    # log model (custom transformer)
    wrapped_model = PipelineWrapperModel(best_pipeline)
    input_example = pd.DataFrame(X_test.head())

    # signature
    signature = infer_signature(input_example, best_pipeline.predict(input_example))
    
    # load custom conda env from utils/env.yaml
    with open('utils/env.yaml', 'r') as f:
        conda_env = yaml.safe_load(f)
     
    # remove existing model directory so it wont conflict with new model   
    model_path = "Artifacts/custom_model"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=wrapped_model,
        input_example=input_example,
        signature=signature,
        conda_env=conda_env
    )

    # infer requirements.txt for reproducibility
    requirements = infer_pip_requirements(
        model_uri=model_path,
        flavor="python_function"
    )
    with open("Artifacts/custom_model/requirements.txt", "w") as f:
        f.write("\n".join(requirements))

    # log model dir as artifact
    mlflow.log_artifacts(model_path, artifact_path="custom_model")