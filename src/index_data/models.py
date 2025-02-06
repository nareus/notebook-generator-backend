from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentsResponse(BaseModel):
    documents: List[str]

class IndexPDFResponse(BaseModel):
    message: str