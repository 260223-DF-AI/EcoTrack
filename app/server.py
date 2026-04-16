from fastapi import FastAPI, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from SageMaker import upload, deploy, predict, shutdown, animal_loc_analysis, SpeciesStatuses
from PIL import Image
from torchvision import transforms

# from .routers import *
# from .utils.logger import get_logger, log_execution

# logger = get_logger(__name__)

app = FastAPI(
    title = "EcoTrack API",
    description = "API for EcoTrack interaction"
)

species_statuses = SpeciesStatuses()

std_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

app.state.predictor = None

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

@app.post("/upload")
def post_upload():
    upload()
    return JSONResponse(content={"message": "Model package uploaded."})

@app.post("/deploy")
def post_deploy():
    # Replace an existing endpoint reference with a fresh deployment when requested.
    if app.state.predictor is not None:
        shutdown(app.state.predictor)
    app.state.predictor = deploy()
    return JSONResponse(content={"message": "Model deployed.", "endpoint": app.state.predictor.endpoint_name})


@app.on_event("startup")
def on_startup():
    # Create one predictor for the app process and reuse it across requests.
    if app.state.predictor is None:
        app.state.predictor = deploy()


@app.on_event("shutdown")
def on_shutdown():
    if app.state.predictor is not None:
        shutdown(app.state.predictor)
        app.state.predictor = None



@app.post("/analyze")
async def post_classify_animal(request: Request, img_file: UploadFile, additional_info: str=Form()):
    """Classify an uploaded image and return the rendered result page."""
    img_extensions = ['jpeg', 'jpg', 'png', 'heic']
    ext_start_idx = img_file.filename.rfind('.')
    if ext_start_idx == -1 or img_file.filename[ext_start_idx + 1:].lower() not in img_extensions:
        await img_file.close()
        raise HTTPException(status_code=415, detail="Needs to be an image file type with extensions 'jpeg', 'jpg', 'png', or 'heic'")

    if app.state.predictor is None:
        await img_file.close()
        raise HTTPException(status_code=503, detail="Predictor is not initialized. Deploy the model first.")

    try:
        img = Image.open(img_file.file).convert("RGB")
        input_data = std_transform(img).unsqueeze(0)
        confidence, label = predict(app.state.predictor, input_data)

        species, all_statuses, endangered_status = species_statuses[label]
        await img_file.close()
    except Exception as e:
        await img_file.close()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    result = {
        'species' : species,
        'endangered_status': endangered_status,
        'all statuses': all_statuses,
        'confidence': confidence
    }

    if endangered_status in ['ENDANGERED', 'CRITICALLY ENDANGERED', 'REGIONALLY']:
        evalutation = animal_loc_analysis(result, additional_info=additional_info)
        result['unusual_location'] = evalutation['unusual_location']
    return templates.TemplateResponse(request=request, name='classify_animal.html', context={'result': result})

def start_server():
    """
    Launch API server with uvicorn
    """
    uvicorn.run("app.server:app", host="localhost", port=8000, reload=True)
