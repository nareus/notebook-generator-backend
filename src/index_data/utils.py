import os
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import textwrap
import uuid
from config import Config
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from fastapi import UploadFile
from io import BytesIO

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key=Config.PINECONE_API_KEY)
client = MongoClient(Config.MONGODB_URI, server_api=ServerApi('1'))
documents = client['fyp']['documents']

def extract_text_from_pdf(file : UploadFile):
    """
    Extract text from a PDF file.
    """
    file_content = file.file.read()
    pdf_file = BytesIO(file_content)
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=100):
    """
    Split the text into overlapping chunks.
    """
    chunks = textwrap.wrap(text, chunk_size)
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped_chunks.append(chunk)
        else:
            overlapped_chunks.append(chunks[i-1][-overlap:] + chunk)
    return overlapped_chunks

def embed_text(text):
    """
    Create an embedding for the given text.
    """
    return model.encode(text).tolist()

# def index_pdf_directory(directory_path, index_name):
#     """
#     Index all PDF files in a directory into Pinecone.
#     """
#     existing_pdfs = get_list_of_pdfs()
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".pdf"):
#             pdf_path = os.path.join(directory_path, filename)
#             if pdf_path not in existing_pdfs:
#                 print(f"Indexing {pdf_path}")
#                 documents.insert_one({'name': filename})
#                 index_pdf(pdf_path, index_name)

def get_list_of_pdfs():
    client = MongoClient(Config.MONGODB_URI, server_api=ServerApi('1'))
    documents = client['fyp']['documents']
    return [doc['name'] for doc in documents.find({}, {'name': 1})]
