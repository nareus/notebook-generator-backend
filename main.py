from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
from openai import OpenAI
import json
from config import Config

app = FastAPI()

class NotebookRequest(BaseModel):
    topic: str
    num_cells: int

class NotebookResponse(BaseModel):
    cells: List[str]

def generate(prompt: str):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant that generates notebook cells based on the given topic."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        response_format={ "type": "text" }  # Changed to text format
    )
    return response.choices[0].message.content

@app.post("/generate_notebook", response_model=NotebookResponse)
async def generate_notebook(request: NotebookRequest):
    cells = []
    
    for _ in range(request.num_cells):
        # Construct prompt
        prompt = f"Generate a notebook cell about the topic: {request.topic}"
        
        # Generate cell content
        cell_content = generate(prompt)
        
        cells.append(cell_content)
    
    return NotebookResponse(cells=cells)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)