import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Add any other configuration variables here

    @classmethod
    def validate(cls):
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and value is None:
                raise ValueError(f"Environment variable {key} is not set. Please check your .env file.")

# Validate the configuration when this module is imported
Config.validate()