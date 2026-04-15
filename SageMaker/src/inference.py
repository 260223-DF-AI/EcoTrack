import os
import json
import torch
from SageMaker.src.AnimalResNet import AnimalResNet

def model_fn(model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnimalResNet(num_classes=90)
    model_path = os.path.join(model_dir, 'model.pth')

    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    
    return model.to(device)

def input_fn(request_body, request_content_type):
    #TODO: How do we get images passed here?
    if request_content_type == 'application/json':
        body = request_body.decode('utf-8') if isinstance(request_body, bytes) else request_body
        data = json.loads(body)
        return torch.tensor(data, dtype=torch.float32).view(-1,1)
    
    raise ValueError(f"Unsupported formatting")

def predict_fn(input_data, model):
    """Pass in as tensor, returns float and int"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_data = input_data.to(device)
    
        model.eval()
        with torch.no_grad():
            logits = model(input_data)
        probabilities = torch.softmax(logits, dim=1)
        confidence, output = torch.max(probabilities, dim=1)
        confidence = float(confidence.item()) * 100
        return confidence, output.item()
    
    except Exception as e:
        print(f"Exception occured: {e}")
        raise e

def output_fn(prediction, content_type):
    return json.dumps(prediction.tolist())