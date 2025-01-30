import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from generate_notebooks.models import NotebookRequest, NotebookResponse, StructureFeedbackRequest, StructureRequest, StructureResponse, TopicFeedbackRequest, TopicRequest, TopicResponse, NotebookStructure
from generate_notebooks.utils import generate_with_context, retrieve_context, create_notebook
from openai import OpenAI
import nbformat

router = APIRouter()

@router.post("/generate_notebook", response_model=NotebookResponse)
async def generate_notebook(request: NotebookRequest):
    # Retrieve context from Pinecone
    # context = retrieve_context(request.structure.notebook_name)
    
    # structure : NotebookStructure = request.structure
    # print(structure)

    # client = OpenAI()
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": """
    #             You are an expert in creating educational Jupyter notebooks.
    #             Generate comprehensive notebook content based on the given topic, structure and context.
    #             For each section and page in the structure:
    #             1. Create detailed educational content following the specified type (text/code/markdown/chart)
    #             2. Include practical examples and explanations
    #             3. Ensure content flows logically between sections
    #             4. Use '---' as a cell separator
                
    #             Format the response with clear cell separations and appropriate content types.
    #             """
    #         },
    #         {
    #             "role": "user",
    #             "content": f"Topic: {structure.notebook_name}\n\nStructure:\n{str(structure.sections)}\n\nContext:\n{context}"
    #         }
    #     ]
    # )

    # notebook_content = response.choices[0].message.content
    notebook_content  = """Introduction to Python Programming
    ---
    # Getting Started with Python
    Python is a versatile programming language known for its readability and extensive library ecosystem.

    Key concepts we'll cover:
    - Variables and Data Types
    - Control Flow
    - Functions
    - Basic I/O
    ---
    # First Python Program
    Let's write our first Python program:
    ```python
    print("Hello, World!")
    name = "Python Learner"
    print(f"Welcome, {name}!")

    ---
    # Variables and Data Types
    Python has several built-in data types:

    Strings
    Numbers (int, float)
    Lists
    Dictionaries

    ---
    # Example of different data types
    text = "Python"
    number = 42
    decimal = 3.14
    my_list = [1, 2, 3]
    my_dict = {"name": "Python", "type": "Programming Language"}

    print(f"String: {text}")
    print(f"Integer: {number}")
    print(f"Float: {decimal}")
    print(f"List: {my_list}")
    print(f"Dictionary: {my_dict}")
    ---"""

    # Split content into cells based on "---" or other delimiter
    notebook_cells = []
    cells = [cell.strip() for cell in notebook_content.split("---") if cell.strip()]
    for cell in cells:
        # Detect cell type based on content patterns
        if "```python" in cell or "```" in cell:
            # Code cell - strip out the python markers
            cell_type = "code"
            cell_content = cell.replace("```python", "").replace("```", "").strip()
        elif cell.startswith("#"):
            # Markdown header
            cell_type = "markdown" 
            cell_content = cell
        elif any(marker in cell for marker in ["*", "-", "1.", "|"]):
            # Markdown with lists, bullet points or tables
            cell_type = "markdown"
            cell_content = cell
        else:
            # Default to markdown for text content
            cell_type = "markdown"
            cell_content = cell

        notebook_cells.append({
            "cell_type": cell_type,
            "content": cell_content.split("\n")
        })
    print(notebook_cells)

    notebook = create_notebook(notebook_cells)
    notebook_content = nbformat.writes(notebook)
    
    return JSONResponse(
        content={"notebook": notebook_content},
        headers={
            "Content-Disposition": "attachment; filename=generated_notebook.ipynb"
        }
    )
    
    # return NotebookResponse(notebook=notebook)

@router.post("/generate_structure", response_model=StructureResponse)
async def generate_notebook_structure(request: StructureRequest):
    # Retrieve context from Pinecone
    context = retrieve_context(request.topic)

    # Prepare prompt for generating notebook structure
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system", 
                "content": """
                You are an expert in creating structured notebook outlines for educational purposes. 
                Generate a comprehensive JSON structure for a Jupyter notebook 
                based on the given topic and context. Follow these guidelines:

                1. Choose a descriptive notebook name related to the topic
                2. Create multiple sections that comprehensively cover the topic
                3. For each section, include multiple pages with specific attributes:
                   - title: Clear, descriptive page title
                   - type: One of 'text', 'code', 'markdown', or 'chart'
                   - placeholders: Optional suggestions for content development
                   - content: Optional initial content (can be left empty)

                Example Structure:
                {
                    "notebook_name": "Advanced Python Programming",
                    "sections": [
                        {
                            "name": "Introduction to Advanced Concepts",
                            "pages": [
                                {
                                    "title": "Course Overview",
                                    "type": "text",
                                    "placeholders": ["Learning Objectives", "Prerequisites"]
                                },
                                {
                                    "title": "Python Ecosystem",
                                    "type": "markdown",
                                    "placeholders": ["Key Libraries", "Development Environment"]
                                }
                            ]
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
    )

    # Parse the JSON response
    try:
        structure = json.loads(response.choices[0].message.content)
        
        # Validate structure matches expected interface
        validated_structure = {
            "notebook_name": structure.get("notebook_name", f"{request.topic} Notebook"),
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

                # Ensure type is one of the allowed values
                if validated_page["type"] not in ['text', 'code', 'markdown', 'chart']:
                    validated_page["type"] = "text"

                validated_section["pages"].append(validated_page)

            validated_structure["sections"].append(validated_section)

        return StructureResponse(structure=validated_structure)
    except json.JSONDecodeError:
        # Fallback to a default structure if JSON parsing fails
        default_structure = {
            "notebook_name": f"{request.topic} Notebook",
            "sections": [
                {
                    "name": "Introduction",
                    "pages": [
                        {
                            "title": "Overview",
                            "type": "text",
                            "placeholders": ["Key Concepts", "Scope"],
                            "content": ""
                        }
                    ]
                }
            ]
        }
        return StructureResponse(structure=default_structure)
    
@router.post("/generate_feedback_structure", response_model=StructureResponse)
async def generate_feedback_notebook_structure(request: StructureFeedbackRequest):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
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
        model="gpt-4",
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