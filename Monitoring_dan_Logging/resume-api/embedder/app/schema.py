from pydantic import BaseModel
from typing import List

class EmbedRequest(BaseModel):
    texts: List[str]