import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from config import Config
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Pinecone Initialization
pc = Pinecone(api_key=Config.PINECONE_API_KEY)
index = pc.Index(host="https://fyp-context-0mqoalz.svc.aped-4627-b74a.pinecone.io")

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

class NotebookRequest(BaseModel):
    topic: str

class NotebookResponse(BaseModel):
    cells: List[str]
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

    # Ensure arbitrary types are allowed
    model_config = {
        "arbitrary_types_allowed": True
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def retrieve_context(topic: str, top_k: int = 3):
    # Use Pinecone to find the most relevant documents for the given topic
    query_vector = embed_topic(topic)  # Function to embed the topic, defined below
    response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    # Extract relevant information from the retrieved documents
    context = "\n\n".join([match['metadata']['text'] for match in response['matches']])
    return context

def embed_topic(topic: str):
    # Use OpenAI or any embedding model to generate an embedding vector for the topic
    return model.encode(topic).tolist()

def generate_with_context(prompt: str, context: str):
    # Combine context and prompt to generate a response
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system", 
                "content": "You need to generate jupyter notebook cells for university students based on the given topic and context. Make it elaborate and keep a 3 to 1 ratio of explanation and code. Seperate the cells using '---'"
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\n{prompt}"
            }
        ],
        response_format={ "type": "text" }
    )
    return response.choices[0].message.content

@app.post("/generate_notebook", response_model=NotebookResponse)
async def generate_notebook(request: NotebookRequest):
    # Retrieve context from Pinecone
    context = retrieve_context(request.topic)

    # Construct prompt with topic and context
    prompt = f"Generate structured content on the topic '{request.topic}' with each section as a notebook cell."

    # Generate notebook content with the contextual prompt
    notebook_content = generate_with_context(prompt, context)

    # Split content into cells based on "---" or other delimiter
    cells = [cell.strip() for cell in notebook_content.split("---") if cell.strip()]

    return NotebookResponse(cells=cells)

@app.post("/generate_structure", response_model=StructureResponse)
async def generate_notebook_structure(request: StructureRequest):
    # Retrieve context from Pinecone
    context = retrieve_context(request.topic)

    # Prepare prompt for generating notebook structure
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
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
        response_format={ "type": "json_object" }
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    