import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MONGODB_URI = os.getenv("MONGODB_URI")

    @classmethod
    def validate(cls):
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and value is None:
                raise ValueError(f"Environment variable {key} is not set. Please check your .env file.")

Config.validate()