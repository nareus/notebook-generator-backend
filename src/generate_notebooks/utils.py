from openai import OpenAI
from pinecone import Pinecone
from config import Config
from sentence_transformers import SentenceTransformer
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from generate_notebooks.models import Cell
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Pinecone Initialization
pc = Pinecone(api_key=Config.PINECONE_API_KEY)
index = pc.Index(host="https://fyp-context-0mqoalz.svc.aped-4627-b74a.pinecone.io")


def retrieve_context(topic: str, top_k: int = 3):
    client = MongoClient(Config.MONGODB_URI, server_api=ServerApi('1'))
    documents = client['fyp']['documents']
    selected_documents = documents.find({"selected": True}) 
    selected_doc_names = [doc["name"] for doc in selected_documents] 
    print("Slected Doc Names", selected_doc_names) 

    query_vector = embed_topic(topic) 
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"filename": {"$in": selected_doc_names}}
    )

    context = "\n\n".join([match['metadata']['text'] for match in response['matches']])
    return context

def embed_topic(topic: str):
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Use OpenAI or any embedding model to generate an embedding vector for the topic
    return model.encode(topic).tolist()

def create_notebook(cells: list[Cell]):
    nb = new_notebook()
    
    for cell in cells:
        if cell.type == 'markdown' or cell.type == 'image':
            new_cell = new_markdown_cell(cell.content)
        elif cell.type == 'code' or cell.type == 'chart':
            new_cell = new_code_cell(cell.content)
        else:
            raise ValueError(f"Invalid cell type: {cell.type}")
        
        nb.cells.append(new_cell)
    
    return nb