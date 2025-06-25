from pydantic import BaseModel
from typing import List, Any

class InferenceRequest(BaseModel):
    columns: List[str]
    data: List[List[Any]]