import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from generate_notebooks.models import (
    NotebookRequest, NotebookResponse, StructureFeedbackRequest,
    StructureRequest, StructureResponse, TopicFeedbackRequest,
    TopicRequest, TopicResponse, CellRequest, AllCellsResponse,
    CellResponse, CELL_TYPES
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

    context = retrieve_context(request.topic)
        # Choose the appropriate system prompt based on the cell type
    if request.type == "short_paragraph":
        system_prompt = (
            """
            You are an expert in creating educational Jupyter notebooks for university-level students. 
            Generate a short, engaging paragraph (2-5 sentences) introducing the given concept. 
            - Start with a real-world analogy or an intuitive explanation before introducing technical terms.
            - Avoid jargon initially, and introduce formulas or definitions only after setting the intuition.
            - The explanation should be clear, concise, and self-contained without extra commentary.
            """
        )
    elif request.type == "bullet_points":
        system_prompt = (
            """
            You are an expert in creating educational Jupyter notebooks for university students. 
            Generate a set of bullet points summarizing the concept based on the given topic. 
            - Keep each bullet clear, concise, and focused on one key idea.
            - If relevant, include real-world applications or examples to reinforce understanding.
            - Use a logical order, ensuring the points build on each other progressively.
            """
        )
    elif request.type == "numbered_list":
        system_prompt = (
            """
            You are an expert in creating structured, step-by-step educational content. 
            Generate a numbered list explaining the given concept in a progressive and logical order. 
            - Each step should be clear, self-contained, and build upon the previous one.
            - Avoid skipping intermediate steps—assume the reader is new to the topic.
            - If applicable, connect the steps to a real-world scenario for better comprehension.
            """
        )
    elif request.type == "code_snippet":
        system_prompt = (
            """
            You are an expert in creating educational Jupyter notebooks for university students. 
            Generate a concise Python code snippet that demonstrates the given concept. 
            - The code should be beginner-friendly with inline comments explaining each step.
            - Ensure all necessary imports are included for a fully self-contained example.
            - Use simple, clear logic rather than unnecessary complexity.
            - Avoid excessive print statements; use structured output when relevant.
            - only include code and not any other text
            - Remove the explicit "```python" directions.
            """     
        )
    elif request.type == "code_with_output":
        system_prompt = (
            """
            You are an expert in creating educational Jupyter notebooks for university students. 
            Generate a Python code snippet that produces visible output demonstrating the given concept. 
            - Ensure expected output is included (either as printed results or as comments).
            - Add inline comments explaining key operations.
            - Keep the example clear, simple, and easy to follow without unnecessary complexity.
            - only include code and not any other text
            - Remove the explicit "```python" directions.
            """
        )
    elif request.type == "code_with_visualization":
        system_prompt = (
            """
            You are an expert in creating educational Jupyter notebooks for university students. 
            Generate a Python code snippet that creates a visualization (e.g., a chart, plot, or graph) to illustrate the concept. 
            - Ensure all necessary imports are included (e.g., Matplotlib, Seaborn).
            - Generate the visualization step by step (first raw data, then any modifications like regression lines).
            - Include axis labels, titles, and legends for clarity.
            - Avoid using external utility functions—make the code self-contained.
            - only include code and not any other text
            - Remove the explicit "```python" directions.
            """
        )
    elif request.type == "multiple_paragraphs":
        system_prompt = ("""
            You are an expert in creating educational Jupyter notebooks for university-level students. 
            Generate detailed Markdown content (a few paragraphs) providing an in-depth explanation or description of the given concept. 

            Content Guidelines:
            - **Start with an intuitive explanation or real-world analogy** before introducing technical details.
            - **Use clear, structured paragraphs** to break down the concept logically.
            - **Introduce definitions and equations progressively**, ensuring a smooth transition between ideas.
            - If applicable, **explain real-world applications** of the concept.
            - **Use headings and subheadings where necessary** to enhance readability.
            - **Keep the explanation self-contained and beginner-friendly**, assuming the reader has no prior knowledge.

            Return only the Markdown-formatted content without any extra notes.
            """
        )
    else:
        system_prompt = (
            "You are an expert in creating educational Jupyter notebooks for university level students. "
            "Generate cell content based on the given topic, prompt, and context. Make sure it is clear, concise, and directly addresses the subject. "
            "Return only the cell content without extra text."
            " Add headings if needed"
        )

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Topic: {request.topic}\n\n"
                    f"Prompt: {request.prompt}\n\n"
                    f"Context:\n{context}"
                ),
            },
        ],
    )
    cell_content = response.choices[0].message.content
    return CellResponse(content=cell_content)

@router.post("/generate_all_cells", response_model=AllCellsResponse)
async def generate_all_cells(request: NotebookRequest):
    client = OpenAI()
    context = retrieve_context(request.structure.notebook_name)
    updated_notebook = request.structure

    for index, cell in enumerate(request.structure.cells):
        # Choose the appropriate system prompt based on the cell type
        if cell.type == "short_paragraph":
            system_prompt = (
                """
                You are an expert in creating educational Jupyter notebooks for university-level students. 
                Generate a short, engaging paragraph (2-5 sentences) introducing the given concept. 
                - Start with a real-world analogy or an intuitive explanation before introducing technical terms.
                - Avoid jargon initially, and introduce formulas or definitions only after setting the intuition.
                - The explanation should be clear, concise, and self-contained without extra commentary.
                """
            )
        elif cell.type == "bullet_points":
            system_prompt = (
                """
                You are an expert in creating educational Jupyter notebooks for university students. 
                Generate a set of bullet points summarizing the concept based on the given topic. 
                - Keep each bullet clear, concise, and focused on one key idea.
                - If relevant, include real-world applications or examples to reinforce understanding.
                - Use a logical order, ensuring the points build on each other progressively.
                """
            )
        elif cell.type == "numbered_list":
            system_prompt = (
                """
                You are an expert in creating structured, step-by-step educational content. 
                Generate a numbered list explaining the given concept in a progressive and logical order. 
                - Each step should be clear, self-contained, and build upon the previous one.
                - Avoid skipping intermediate steps—assume the reader is new to the topic.
                - If applicable, connect the steps to a real-world scenario for better comprehension.
                """
            )
        elif cell.type == "code_snippet":
            system_prompt = (
                """
                You are an expert in creating educational Jupyter notebooks for university students. 
                Generate a concise Python code snippet that demonstrates the given concept. 
                - The code should be beginner-friendly with inline comments explaining each step.
                - Ensure all necessary imports are included for a fully self-contained example.
                - Use simple, clear logic rather than unnecessary complexity.
                - Avoid excessive print statements; use structured output when relevant.
                - only include code and not any other text
                - Remove the explicit "```python" directions.
                """     
            )
        elif cell.type == "code_with_output":
            system_prompt = (
                """
                You are an expert in creating educational Jupyter notebooks for university students. 
                Generate a Python code snippet that produces visible output demonstrating the given concept. 
                - Ensure expected output is included (either as printed results or as comments).
                - Add inline comments explaining key operations.
                - Keep the example clear, simple, and easy to follow without unnecessary complexity.
                - only include code and not any other text
                - Remove the explicit "```python" directions.   
                """
            )
        elif cell.type == "code_with_visualization":
            system_prompt = (
                """
                You are an expert in creating educational Jupyter notebooks for university students. 
                Generate a Python code snippet that creates a visualization (e.g., a chart, plot, or graph) to illustrate the concept. 
                - Ensure all necessary imports are included (e.g., Matplotlib, Seaborn).
                - Generate the visualization step by step (first raw data, then any modifications like regression lines).
                - Include axis labels, titles, and legends for clarity.
                - only include code and not any other text
                - Remove the explicit "```python" directions.
                """
            )
        elif cell.type == "multiple_paragraphs":
            system_prompt = ("""
                You are an expert in creating educational Jupyter notebooks for university-level students. 
                Generate detailed Markdown content (a few paragraphs) providing an in-depth explanation or description of the given concept. 

                Content Guidelines:
                - **Start with an intuitive explanation or real-world analogy** before introducing technical details.
                - **Use clear, structured paragraphs** to break down the concept logically.
                - **Introduce definitions and equations progressively**, ensuring a smooth transition between ideas.
                - If applicable, **explain real-world applications** of the concept.
                - **Use headings and subheadings where necessary** to enhance readability.
                - **Keep the explanation self-contained and beginner-friendly**, assuming the reader has no prior knowledge.

                Return only the Markdown-formatted content without any extra notes.
                """

            )
        else:
            system_prompt = (
                "You are an expert in creating educational Jupyter notebooks for university level students. "
                "Generate cell content based on the given topic, prompt, and context. Make sure it is clear, concise, and directly addresses the subject. "
                "Return only the cell content without extra text."
                " Add headings if needed"
            )
          
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Topic: {request.structure.notebook_name}\n\nPrompt: {cell.content}\n\nContext:\n{context}",
                },
            ],
        )
        updated_notebook.cells[index].content = response.choices[0].message.content
        updated_notebook.cells[index].generated = True
    return AllCellsResponse(structure=updated_notebook)
    


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
                You are an expert in designing structured Jupyter notebooks for university-level students. 
                Generate a well-organized JSON structure for a Jupyter notebook that ensures concepts flow logically from basic intuition to advanced understanding. 
                The notebook should include multiple sections, each containing well-defined cells with specific attributes.

                General Guidelines:
                - The notebook name should match the given topic.
                - Ensure a natural progression of learning.
                - Notebooks can be elaborate in explanations if required, but should not be overly complex.
                - Include a mix of theory, code, and visualizations for a well-rounded educational experience.
                - Maintain a text-to-code ratio of approximately 3:1.
                - Provide different cell types if needed with appropriate content generation prompts, ensuring each concept is explained in a clear and engaging manner.

                Each cell should contain:
                - **type**: One of ['short_paragraph', 'bullet_points', 'numbered_list', 'code_snippet', 'code_with_output', 'code_with_visualization', 'multiple_paragraphs']
                - **content**: The prompt to generate content using an LLM.

                IMPORTANT: Your response must be a valid JSON string. Do not include any additional text or explanations outside the JSON structure.

                Simplified Example Structure:
                {
                    "notebook_name": "Advanced Python Programming",
                    "cells": [
                        {
                            "type": "long_paragraph",
                            "content": "Generate a detailed explanation of Python's decorators."
                        },
                        {
                            "type": "bullet_points",
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
                "type": cell.get("type", "short_paragraph"),
                "content": cell.get("content", "")
            }

            # Ensure type is one of the allowed values
            if validated_cell["type"] not in CELL_TYPES:
                validated_cell["type"] = "short_paragraph"

            validated_structure["cells"].append(validated_cell)
            
        return StructureResponse(structure=validated_structure)
    except json.JSONDecodeError:
        # Fallback to a default structure if JSON parsing fails
        default_structure = {
            "notebook_name": f"{request.topic} Notebook",
            "cells": [
                {
                    "type": "short_paragraph",
                    "content": "Generate an introduction to the topic."
                }
            ]
        }
        return StructureResponse(structure=default_structure)
    except Exception:
            # Fallback for any other error
            default_structure = {
                "notebook_name": f"{request.topic} Notebook",
                "cells": [
                    {
                        "type": "short_paragraph",
                        "content": "An error occurred while generating the structure. Please try again."
                    }
                ]
            }
            return StructureResponse(structure=default_structure)
    
@router.post("/generate_feedback_structure", response_model=StructureResponse)
async def generate_feedback_notebook_structure(request: StructureFeedbackRequest):
    # Optionally retrieve context based on the topic (if available in the request)
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert in refining structured Jupyter notebook designs for university-level students.
                Given the initial notebook structure and the provided feedback, generate an improved JSON structure for the notebook.
                
                Guidelines:
                - The notebook name should match the given topic.
                - The JSON should have a "notebook_name" field and a "cells" list.
                - Each cell must contain:
                  - "type": one of ['short_paragraph', 'bullet_points', 'numbered_list', 'code_snippet', 'code_with_output', 'code_with_visualization', 'multiple_paragraphs']
                  - "content": the prompt to generate cell content using an LLM.
                - Ensure the final structure reflects natural progression and incorporates the feedback.
                - IMPORTANT: Your response must be a valid JSON string. Do not include any extra text.
                """
            },
            {
                "role": "user", 
                "content": f"Initial Structure:\n{request.structure}\n\nFeedback:\n{request.feedback}"
            }
        ],
        response_format={ "type": "json_object" }
    )

    try:
        structure = json.loads(response.choices[0].message.content)
        
        # Validate structure matches expected interface using a similar approach to generate_notebook_structure
        validated_structure = {
            "notebook_name": structure.get("notebook_name", f"{request.topic} Notebook"),
            "cells": []
        }
        for cell in structure.get("cells", []):
            validated_cell = {
                "type": cell.get("type", "short_paragraph"),
                "content": cell.get("content", "")
            }
            if validated_cell["type"] not in CELL_TYPES:
                validated_cell["type"] = "short_paragraph"
            validated_structure["cells"].append(validated_cell)
            
        return StructureResponse(structure=validated_structure)
    except json.JSONDecodeError:
        # Fallback to a default structure if JSON parsing fails
        default_structure = {
            "notebook_name": f"{request.topic} Notebook",
            "cells": [
                {
                    "type": "short_paragraph",
                    "content": "Generate an improved introduction based on the feedback."
                }
            ]
        }
        return StructureResponse(structure=default_structure)
    except Exception:
        # Fallback for any other error
        default_structure = {
            "notebook_name": f"{request.topic} Notebook",
            "cells": [
                {
                    "type": "short_paragraph",
                    "content": "An error occurred while refining the structure. Please try again."
                }
            ]
        }
        return StructureResponse(structure=default_structure)
    
def validate_structure(structure: dict) -> bool:
    try:
        # Basic structure validation
        if not isinstance(structure, dict):
            return False
        if "topics" not in structure or not isinstance(structure["topics"], list):
            return False
        if not all(isinstance(topic, str) for topic in structure["topics"]):
            return False
        return True
    except Exception:
        return False

@router.post("/generate_topics", response_model=TopicResponse)
async def generate_notebook_topics(request: TopicRequest):

    context = retrieve_context(request.topic)
    client = OpenAI()
    
    max_retries = 3
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """
                    You are an expert in designing structured educational curricula for university-level students. 
                    Generate a JSON response containing a list of well-structured subtopics for Jupyter notebooks based on the given main topic. 

                    Guidelines:
                    - **Ensure a logical progression** from fundamental concepts to more advanced topics.
                    - **Subtopics should be distinct** but collectively provide a **comprehensive understanding** of the main topic.
                    - If applicable, include **both theoretical and practical aspects**.
                    - The topics should be **engaging, relevant, and applicable to real-world scenarios**.

                    JSON Format (If Notebook Count is 3):
                    {
                        "topics": ["Introduction to <main_topic>", "Intermediate Concepts in <main_topic>", "Advanced Applications of <main_topic>"]
                    }

                    Generate **only the JSON response** without any additional commentary.
                    """

                },
                {
                    "role": "user", 
                    "content": f"Topic: {request.topic}\n\nNotebook Count: {request.notebook_count}\n\nContext:\n{context}"
                }
            ],
            response_format={ "type": "json_object" }
        )
        
        try:
            structure = json.loads(response.choices[0].message.content)
            if validate_structure(structure):
                return TopicResponse(topics=structure["topics"])
        except json.JSONDecodeError:
            continue
            
    # Fallback structure if all retries fail
    default_structure = {
        "topics": [f"{request.topic} Part {i+1}" for i in range(request.notebook_count)]
    }
    return TopicResponse(topics=default_structure["topics"])

@router.post("/generate_feedback_topics", response_model=TopicResponse)
async def generate_feedback_notebook_topics(request: TopicFeedbackRequest):
    client = OpenAI()
    max_retries = 3
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "system",
                "content": """
                    You are an expert in refining notebook topics based on feedback.

                    Review the provided notebook topics and feedback, and generate a new JSON response containing a list of revised subtopics. The revised subtopics should:

                    - Reflect the feedback given.
                    - Ensure a logical progression from fundamental concepts to more advanced topics.
                    - Maintain distinct subtopics while offering a comprehensive understanding of the main topic.
                    - Include both theoretical and practical aspects if applicable.
                    - Ensure that the topics are engaging, relevant, and applicable to real-world scenarios.
                    
                    The response should be in the following format:
                    
                    {
                    "topics": ["topic1", "topic2", "topic3"]
                    }
                    """
                },
                {
                "role": "user",
                "content": "Initial Topics:\n{request.topics}\n\nChange according to this feedback:\n{request.feedback}"
                }
            ]
        )

        try:
            structure = json.loads(response.choices[0].message.content)
            if validate_structure(structure):
                return TopicResponse(topics=structure["topics"])
        except json.JSONDecodeError:
            continue
            
    # Fallback structure if all retries fail
    default_structure = {
        "topics": [f"{request.topic} Part {i+1}" for i in range(request.notebook_count)]
    }
    return TopicResponse(topics=default_structure["topics"])

__all__ = ['router']