import os
import torch
from PIL import Image
from torchvision import transforms

from BirdResNet import BirdResNet

MODEL_PATH = "model/weights/model101_93p_evalacc.pth" # this one is good for nighthawks and black footed albatros

# Transformations
std_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( #these are provided hardcoded values
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) 
    ])

model = BirdResNet(200)
if os.path.exists(MODEL_PATH):
        best_model = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model["model_state_dict"])

output = None
img_path = input("Enter the path to your image: ")
if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")

        model.eval()
        with torch.no_grad():
                out = model(std_transform(img).unsqueeze(0))
                output = torch.argmax(out, dim=1)
                print(output.item() + 1)
