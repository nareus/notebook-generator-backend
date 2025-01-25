from openai import OpenAI
from pinecone import Pinecone
from config import Config
from sentence_transformers import SentenceTransformer

# Pinecone Initialization
pc = Pinecone(api_key=Config.PINECONE_API_KEY)
index = pc.Index(host="https://fyp-context-0mqoalz.svc.aped-4627-b74a.pinecone.io")


def retrieve_context(topic: str, top_k: int = 3):
    # Use Pinecone to find the most relevant documents for the given topic
    query_vector = embed_topic(topic)  # Function to embed the topic, defined below
    response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    # Extract relevant information from the retrieved documents
    context = "\n\n".join([match['metadata']['text'] for match in response['matches']])
    return context

def embed_topic(topic: str):
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Use OpenAI or any embedding model to generate an embedding vector for the topic
    return model.encode(topic).tolist()

def generate_with_context(prompt: str, context: str):
    # Combine context and prompt to generate a response
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
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