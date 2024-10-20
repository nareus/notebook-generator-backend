import uvicorn
from config import Config
from main import app

def main():
    # Config.validate() will be called when config is imported
    # If any required environment variable is missing, it will raise an error here

    # Run the Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()