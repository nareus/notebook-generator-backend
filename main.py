
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from generate_notebooks.router import router as generate_notebook_router
from src.index_data.router import router as index_data_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_notebook_router)
app.include_router(index_data_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    