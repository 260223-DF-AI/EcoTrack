import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
import time

# set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda')

# Opening the image using PIL
img = Image.open(Path('bird.jpg').absolute())

# Loading the model and preprocessor from HuggingFace
preprocessor = EfficientNetImageProcessor.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")
model = EfficientNetForImageClassification.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")
#model = model.to(device)

# Preprocessing the input
inputs = preprocessor(img, return_tensors="pt")
#inputs = inputs.to(device)

# Running the inference
with torch.no_grad():
    logits = model(**inputs).logits

# Getting the predicted label
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label]) # type: ignore


# Followed code from these notes: c248-advanced-cv-and-transfer-learning.md
class PreTrainedResNet34(nn.Module):
    def __init__(self):
        super(PreTrainedResNet34, self).__init__()
        # load model
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # freeze layers
        for param in model.parameters():
            param.requires_grad = False

        # adjust final fc layer with what we need
        num_classes = 200
        prev_in_features = model.fc.in_features
        self.model.fc = nn.Linear(prev_in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
