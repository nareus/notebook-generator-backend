from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal

CODE_CELL_TYPES = (
    "code_snippet",
    "code_with_output",
    "code_with_visualization",
)
CELL_TYPES = (
    "short_paragraph",
    "bullet_points",
    "blockquote",
    "multiple_paragraphs",
) + CODE_CELL_TYPES

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
    loading: Optional[bool] = None
    generated: Optional[bool] = None

    @field_validator('type')
    def validate_cell_type(cls, value):
        if value not in CELL_TYPES:
            raise ValueError(f"Invalid cell type: {value}. Must be one of {CELL_TYPES}.")
        return value

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
    type: str

    @field_validator('type')
    def validate_cell_type(cls, value):
        if value not in CELL_TYPES:
            raise ValueError(f"Invalid cell type: {value}. Must be one of {CELL_TYPES}.")
        return value

class CellResponse(BaseModel):
    content: str

class AllCellsResponse(BaseModel):
    structure: NotebookStructure