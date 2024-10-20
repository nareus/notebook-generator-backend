import os
import PyPDF2
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import textwrap
import uuid
from config import Config

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
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

def index_pdf(pdf_path, index_name):
    """
    Index a PDF file into Pinecone.
    """
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Chunk the text
    chunks = chunk_text(text)
    
    # Initialize Pinecone
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    
    # Create or connect to the index
    if index_name not in pc.list_indexes():
        pc.create_index(index_name, dimension=384)
    index = pc.Index(index_name)
    
    # Embed and index each chunk
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        index.upsert([(str(uuid.uuid4()), embedding, {"text": chunk, "source": pdf_path, "chunk_id": i})])
    
    print(f"Indexed {len(chunks)} chunks from {pdf_path}")

def index_pdf_directory(directory_path, index_name):
    """
    Index all PDF files in a directory into Pinecone.
    """
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            index_pdf(pdf_path, index_name)