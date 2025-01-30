from pydantic import BaseModel, Field
from typing import List, Optional

class NotebookPage(BaseModel):
    title: str
    type: str = Field(
        ..., 
        description="Type of notebook page",
        pattern="^(text|code|markdown|chart)$"
    )
    placeholders: Optional[List[str]] = None
    content: Optional[str] = None

class NotebookSection(BaseModel):
    name: str
    pages: List[NotebookPage]

class NotebookStructure(BaseModel):
    notebook_name: str
    sections: List[NotebookSection]

    # Optional: Custom validation
    @classmethod
    def validate_page_type(cls, v):
        allowed_types = ['text', 'code', 'markdown', 'chart']
        if v not in allowed_types:
            raise ValueError(f"Page type must be one of {allowed_types}")
        return v

class StructureRequest(BaseModel):
    topic: str

class StructureResponse(BaseModel):
    structure: NotebookStructure

class TopicFeedbackRequest(BaseModel):
    topics: str
    feedback: str

class StructureFeedbackRequest(BaseModel):
    structure: str
    feedback: str

class TopicRequest(BaseModel):
    topic: str
    notebook_count: int

class TopicResponse(BaseModel):
    topics: List[str]

class NotebookRequest(BaseModel):
    structure: NotebookStructure

class NotebookResponse(BaseModel):
    cells: List[str]

class Cell(BaseModel):
    cell_type: str
    content: str