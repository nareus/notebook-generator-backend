from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from config import Config

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
                "content": "You are a helpful assistant that generates jupyter notebook cells for university students based on the given topic and context. Make it elaborate and keep a 3 to 1 ratio of explanation and code. Seperate the cells using '---'"
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)