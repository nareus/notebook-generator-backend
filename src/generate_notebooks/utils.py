from openai import OpenAI
from pinecone import Pinecone
from config import Config
from sentence_transformers import SentenceTransformer
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from generate_notebooks.models import Cell
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from generate_notebooks.models import CODE_CELL_TYPES

# Global Initialization for faster performance
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
CLIENT = MongoClient(Config.MONGODB_URI, server_api=ServerApi('1'))

# Pinecone Initialization
pc = Pinecone(api_key=Config.PINECONE_API_KEY)
index = pc.Index(host="https://fyp-context-0mqoalz.svc.aped-4627-b74a.pinecone.io")


def retrieve_context(topic: str, top_k: int = 3):
    documents = CLIENT['fyp']['documents']
    selected_documents = documents.find({"selected": True})
    selected_doc_names = [doc["name"] for doc in selected_documents]

    query_vector = embed_topic(topic)
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"filename": {"$in": selected_doc_names}}
    )
    if not response['matches']:
        return 'None'

    context = "\n\n".join([match['metadata']['text'] for match in response['matches']])
    return context

def embed_topic(topic: str):
    # Reuse the pre-loaded model.
    return MODEL.encode(topic).tolist()

def create_notebook(cells: list[Cell]):
    nb = new_notebook()

    for cell in cells:
        if cell.type in CODE_CELL_TYPES:
            new_cell = new_code_cell(cell.content)
        else:
            new_cell = new_markdown_cell(cell.content)
        nb.cells.append(new_cell)

    return nb