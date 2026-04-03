from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# from .routers import *
# from .utils.logger import get_logger, log_execution

# logger = get_logger(__name__)

app = FastAPI(
    title = "EcoTrack API",
    description = "API for EcoTrack interaction"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # would be replaced with web url if hosted publicly 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def get_root():
    return {"message": "Hello from main"}

@app.post("/analyze")
def post_analyze():
    pass

def start_server():
    """
    Launch API server with uvicorn
    """
    uvicorn.run("app.server:app", host="localhost", port=8000, reload=True)
