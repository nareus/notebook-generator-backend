import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from generate_notebooks.models import (
    NotebookRequest, NotebookResponse, StructureFeedbackRequest,
    StructureRequest, StructureResponse, TopicFeedbackRequest,
    TopicRequest, TopicResponse, CellRequest,
    CellResponse
)
from generate_notebooks.utils import retrieve_context, create_notebook
from openai import OpenAI
import nbformat


router = APIRouter()

@router.post("/generate_notebook", response_model=NotebookResponse)
async def generate_notebook(request: NotebookRequest):

    notebook = create_notebook(request.structure.cells)
    notebook_content = nbformat.writes(notebook)
    
    return JSONResponse(
        content={"notebook": notebook_content},
        headers={
            "Content-Disposition": "attachment; filename=generated_notebook.ipynb"
        }
    )

@router.post("/generate_cell_content", response_model=CellResponse)
async def generate_cell(request: CellRequest):
    # Retrieve context from Pinecone
    context = retrieve_context(request.topic)

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert in creating educational Jupyter notebooks for university level students.
                Generate a content of the cell based on the given topic, prompt and context. Make sure it is elaborate, clear and concise. 
                Only add the content of the cell, no other text. If it seems like a code or chart cell, return just the code.
                """
            },
            {
                "role": "user",
                "content": f"Topic: {request.topic}\n\nPrompt: {request.prompt}\n\nContext:\n{context}"
            }
        ]
    )
    cell_content = response.choices[0].message.content
    return CellResponse(content=cell_content)


@router.post("/generate_structure", response_model=StructureResponse)
async def generate_notebook_structure(request: StructureRequest):
    # Retrieve context from Pinecone
    context = retrieve_context(request.topic)

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system", 
                "content": """
                You are an expert in creating structured notebook outlines for educational purposes. 
                The Structure should include multiple cells with a prompt to generate the content using LLM.
                Generate a comprehensive JSON structure for a Jupyter notebook 
                based on the given topic and context. Follow these guidelines:

                1. Use the Topic given as the notebook name
                2. Create multiple cells that comprehensively cover the topic with specific attributes:
                   - type: One of 'code', 'markdown', or 'chart'
                   - content: The prompt to generate the content using LLM

                IMPORTANT: Your response must be a valid JSON string. Do not include any additional text or explanations outside the JSON structure.

                Example Structure:
                {
                    "notebook_name": "Advanced Python Programming",
                    "cells": [
                        {
                            "type": "text",
                            "content": "Generate a detailed explanation of Python's decorators."
                        },
                        {
                            "type": "markdown",
                            "content": "Create a markdown cell with a table summarizing the key features of Python."
                        }
                    ]
                }
                """
            },
            {
                "role": "user", 
                "content": f"Topic: {request.topic}\n\nContext:\n{context}"
            }
        ],
        response_format={ "type": "json_object" }
    )

    try:
        structure = json.loads(response.choices[0].message.content)
        
        # Validate structure matches expected interface
        validated_structure = {
            "notebook_name": structure.get("notebook_name", f"{request.topic} Notebook"),
            "cells": []
        }

        for cell in structure.get("cells", []):
            validated_cell = {
                "type": cell.get("type", "text"),
                "content": cell.get("content", "")
            }

            # Ensure type is one of the allowed values
            if validated_cell["type"] not in ['code', 'markdown', 'chart', 'image']:
                validated_cell["type"] = "markdown"

            validated_structure["cells"].append(validated_cell)

        return StructureResponse(structure=validated_structure)
    except json.JSONDecodeError:
        # Fallback to a default structure if JSON parsing fails
        default_structure = {
            "notebook_name": f"{request.topic} Notebook",
            "cells": [
                {
                    "type": "markdown",
                    "content": "Generate an introduction to the topic."
                }
            ]
        }
        return StructureResponse(structure=default_structure)
    
@router.post("/generate_feedback_structure", response_model=StructureResponse)
async def generate_feedback_notebook_structure(request: StructureFeedbackRequest):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system", 
                "content": """
                You are an expert in refining notebook structures based on feedback.
                Review the provided notebook structure and feedback, then generate 
                an improved JSON structure incorporating the suggested changes.
                Maintain the same format with:
                - notebook_name
                - sections (with name and pages)
                - pages (with title, type, placeholders, content)
                
                Page types must be: 'text', 'code', 'markdown', or 'chart'
                """
            },
            {
                "role": "user", 
                "content": f"Initial Structure:\n{request.structure}\n\nFeedback:\n{request.feedback}"
            }
        ],
    )

    structure = json.loads(response.choices[0].message.content)
    
    validated_structure = {
        "notebook_name": structure.get("notebook_name", "Revised Notebook"),
        "sections": []
    }

    for section in structure.get("sections", []):
        validated_section = {
            "name": section.get("name", "Unnamed Section"),
            "pages": []
        }

        for page in section.get("pages", []):
            validated_page = {
                "title": page.get("title", "Untitled Page"),
                "type": page.get("type", "text"),
                "placeholders": page.get("placeholders", []),
                "content": page.get("content", "")
            }

            if validated_page["type"] not in ['text', 'code', 'markdown', 'chart']:
                validated_page["type"] = "text"

            validated_section["pages"].append(validated_page)

        validated_structure["sections"].append(validated_section)

    return StructureResponse(structure=validated_structure)
    
@router.post("/generate_topics", response_model=TopicResponse)
async def generate_notebook_topics(request: TopicRequest):  # Specify the request model
    context = retrieve_context(request.topic)

    # Prepare prompt for generating notebook structure
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system", 
                "content": """
                You are an expert in creating topics of jupyter notebooks for educational purposes.
                Generate a JSON response containing a list of subtopics with the exact number of notebooks specified derived from the main topic.
                The response should break down the information into well-structured learning segments.

                Expected JSON format if notebook count is 3:
                {
                    "topics": ["topic1", "topic2", "topic3"]
                }
                """

            },
            {
                "role": "user", 
                "content": f"Topic: {request.topic}\n\nNotebook Count: {request.notebook_count}\n\nContext:\n{context}"
            }
        ],
    )
    
    # Parse the JSON response
    structure = json.loads(response.choices[0].message.content)
    return TopicResponse(topics=structure["topics"])

@router.post("/generate_feedback_topics", response_model=TopicResponse)
async def generate_feedback_notebook_topics(request: TopicFeedbackRequest):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system", 
                "content": """
                You are an expert in refining notebook topics based on feedback.
                Review the provided notebook topics and feedback, then generate 
                a valid JSON structure incorporating new notebook names based on the feedback.
                The response must be a valid JSON string in the format:
                {
                    "topics": ["topic1", "topic2", "topic3"]
                }
                """
            },
            {
                "role": "user", 
                "content": f"Initial Topics:\n{request.topics}\n\nChange according to this feedback:\n{request.feedback}"
            }
        ]
    )

    try:
        structure = json.loads(response.choices[0].message.content)
        if not isinstance(structure, dict) or "topics" not in structure:
            # Fallback structure if response is invalid
            structure = {"topics": [request.topics.split(",")[0]]}
    except json.JSONDecodeError:
        # Provide a default structure if JSON parsing fails
        structure = {"topics": [request.topics.split(",")[0]]}

    return TopicResponse(topics=structure["topics"])

__all__ = ['router']