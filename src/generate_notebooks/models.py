from pydantic import BaseModel, Field
from typing import List, Optional

class NotebookPage(BaseModel):
    title: str
    type: str = Field(
        ..., 
        description="Type of notebook page",
        pattern="^(code|markdown)$"
    )
    placeholders: Optional[List[str]] = None
    content: Optional[str] = None

class NotebookSection(BaseModel):
    name: str
    pages: List[NotebookPage]

class Cell(BaseModel):
    type: str
    content: str

class NotebookStructure(BaseModel):
    notebook_name: str
    cells: List[Cell]

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

class CellRequest(BaseModel):
    topic: str
    prompt: str

class CellResponse(BaseModel):
    content: str