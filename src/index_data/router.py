from pymongo.mongo_client import MongoClient
from config import Config
from pymongo.server_api import ServerApi
from fastapi import APIRouter, Query
from .models import DocumentsResponse, IndexPDFResponse, DeletePDFRequest, SelectPDFsRequest
from .utils import extract_text_from_pdf, chunk_text, embed_text, get_list_of_pdfs
from fastapi import UploadFile, File, HTTPException
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import uuid
from config import Config
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

router = APIRouter()

@router.get("/get_documents", response_model=DocumentsResponse)
async def get_documents():
    client = MongoClient(Config.MONGODB_URI, server_api=ServerApi('1'))
    documents = client['fyp']['documents']

    return DocumentsResponse(documents=[doc['name'] for doc in documents.find({}, {'name': 1})])

@router.post("/index_pdf", response_model=IndexPDFResponse)
async def index_pdf(file: UploadFile = File(...)):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    # Validate file is PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail='File must be a PDF')
    
    if file.filename in get_list_of_pdfs():
        return IndexPDFResponse(message="File already indexed")
    
    file_content = extract_text_from_pdf(file)
    
    # Chunk the text
    chunks = chunk_text(file_content)
    
    # Create or connect to the index
    if 'fyp-context' not in [index['name'] for index in pc.list_indexes()]:
        pc.create_index(
            name='fyp-context',
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    index = pc.Index('fyp-context')

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        index.upsert([(str(uuid.uuid4()), embedding, {"text": chunk, "filename": file.filename, "chunk_id": i})])
    
    print(f"Indexed {len(chunks)} chunks from {file.filename}")
    # Store the file name in MongoDB
    client = MongoClient(Config.MONGODB_URI, server_api=ServerApi('1'))
    documents = client['fyp']['documents']
    documents.insert_one({'name': file.filename}, {'selected': False})
    
    return IndexPDFResponse(message=f"Indexed {len(chunks)} chunks from {file.filename}")


@router.post('/delete_pdf')
async def delete_pdf(request: DeletePDFRequest):
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    index = pc.Index('fyp-context')

    query_response = index.query(
        vector=[0] * 384,  # dummy vector for querying
        filter={"filename": request.filename},
        top_k=10000,
        include_metadata=True
    )

    vector_ids = [match.id for match in query_response.matches]
    if vector_ids:
        index.delete(ids=vector_ids)

    # Delete from MongoDB
    client = MongoClient(Config.MONGODB_URI, server_api=ServerApi('1'))
    documents = client['fyp']['documents']
    result = documents.delete_one({'name': request.filename})
    
    return {"message": f"Deleted {request.filename}", "deleted_count": result.deleted_count}

@router.post('/select_pdfs')
async def select_pdfs(request: SelectPDFsRequest):
    client = MongoClient(Config.MONGODB_URI, server_api=ServerApi('1'))
    documents = client['fyp']['documents']
    documents.update_many(
        {'name': {'$nin': request.filenames}},
        {'$set': {'selected': False}}
    )
    documents.update_many(
    {'name': {'$in': request.filenames}}, 
    {'$set': {'selected': True}}
)
