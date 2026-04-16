from fastapi import FastAPI, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from SageMaker.llm import animal_loc_analysis
from SageMaker.check_model import load_model, get_classification

# from .routers import *
# from .utils.logger import get_logger, log_execution

# logger = get_logger(__name__)

MODEL_PATH = 'model/weights/model.pth' 

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

@app.post("/classify_animal")
async def post_classify_animal(request: Request, img_file: UploadFile, additional_info: str=Form()):
    """
    Classify an uploaded animal image and analyze its location if endangered.

    This endpoint accepts an image file upload and optional additional information.
    It uses a pre-trained model to classify the animal species, determine its
    endangered status, and perform location analysis if the species is critically
    endangered (CR) or endangered (EN).

    Args:
        request (Request): The FastAPI request object.
        img_file (UploadFile): The uploaded image file (must be jpeg, jpg, png, or heic).
        additional_info (str): Optional additional context or location information.

    Returns:
        TemplateResponse - Renders the classify_animal.html template with classification results.

    Raises:
        HTTPException - If the uploaded file is not a supported image type (status 415).

    The response includes:
    - species: The predicted animal species.
    - endangered_status: The conservation status of the species.
    - multi: Additional classification metadata.
    - confidence: Confidence score of the classification.
    - unusual_location: (Only if endangered) Analysis of whether the location is unusual.
    """
    print(additional_info)
    img_extensions = ['jpeg', 'jpg', 'png', 'heic']
    ext_start_idx = img_file.filename.rfind('.')
    if(img_file.filename[ext_start_idx + 1:] in img_extensions):
        img_content = img_file.file
        species, endangered_status, multi, confidence = get_classification(bird_classifier, img_content)
        await img_file.close()
    else:
        await img_file.close()
        # Raise an error with status code 415
        raise HTTPException(status_code=415, detail="Needs to be an image file type with extensions 'jpeg', 'jpg', 'png', or 'heic'")
    
    result = {
        'species' : species,
        'endangered_status': endangered_status,
        'multi': multi,
        'confidence': confidence
    }

    print(endangered_status)
    # get most critical status of an animal
    endangered_status = endangered_status[-1]
    # map it to words
    match endangered_status:
        case 'NE': 
            endangered_status = 'NOT EVALUATED'
        case 'DD': 
            endangered_status = 'DATA DEFICIENT'
        case 'LC': 
            endangered_status = 'LEAST CONCERN'
        case 'NT': 
            endangered_status = 'NEAR THREATENED'
        case 'VU': 
            endangered_status = 'VULNERABLE'
        case 'EN': 
            endangered_status = 'ENDANGERED'
        case 'CR': 
            endangered_status = 'CRITICALLY ENDANGERED'
        case 'EW': 
            endangered_status = 'EXTINCT IN THE WILD'
        case 'EX': 
            endangered_status = 'EXTINCT'
        
    result['endangered_status'] = endangered_status
            
    if endangered_status == 'ENDANGERED' or endangered_status == 'CRITICALLY ENDANGERED':
        print("If statement entered")
        evalutation = animal_loc_analysis(result, additional_info=additional_info)
        result['unusual_location'] = evalutation['unusual_location']

    return templates.TemplateResponse(request=request, name='classify_animal.html', context={'result': result})

@app.post("/analyze")
def post_analyze(request: Request, img_result, additional_info: str):
    pass

def start_server():
    """
    Launch API server with uvicorn
    """
    uvicorn.run("app.server:app", host="localhost", port=8000, reload=True)

