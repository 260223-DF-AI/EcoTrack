from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from model.check_model import load_model, get_classification

# from .routers import *
# from .utils.logger import get_logger, log_execution

# logger = get_logger(__name__)

MODEL_PATH = 'model/weights/best50_84p_validacc.pth' 

bird_classifier = load_model(MODEL_PATH)

app = FastAPI(
    title = "EcoTrack API",
    description = "API for EcoTrack interaction"
)

app.mount('/app/static', StaticFiles(directory='app/static'), name='static')

templates = Jinja2Templates(directory='app/templates')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # would be replaced with web url if hosted publicly 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def get_root(request: Request):
    return templates.TemplateResponse(request=request, name='home.html', context={"message": "Hello from main"})

@app.post("/classify_bird")
async def post_classify_bird(request: Request, img_file: UploadFile):
    # print(img_file)
    img_extensions = ['jpeg', 'jpg', 'png', 'heic']
    ext_start_idx = img_file.filename.rfind('.')
    if(img_file.filename[ext_start_idx + 1:] in img_extensions):
        img_content = img_file.file
        species, endangered_status, multi, confidence = get_classification(bird_classifier, img_content)
        await img_file.close()
    else:
        await img_file.close()
        # Raise some error here, probably also want to send an error back to the site as an HTTP status code
        raise HTTPException(status_code=415, detail="Needs to be an image file type with extensions 'jpeg', 'jpg', 'png', or 'heic'")
    result = {
        'species' : species,
        'endangered_status': endangered_status,
        'multi': multi,
        'confidence': confidence
    }
    return templates.TemplateResponse(request=request, name='classify_birds.html', context={'result': result})

@app.post("/analyze")
def post_analyze():
    pass

def start_server():
    """
    Launch API server with uvicorn
    """
    uvicorn.run("app.server:app", host="localhost", port=8000, reload=True)
