import mlflow.pyfunc
import pandas as pd

class PipelineWrapperModel(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input: pd.DataFrame) -> list:
        # take first column as text input
        texts = model_input.iloc[:, 0]  
        return self.pipeline.predict(texts).tolist()