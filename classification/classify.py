import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
import os
import tensorflow
import time

MODEL_PATH = "bird_species_model.pth"

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
class BirdClassifier(nn.Module):
    def __init__(self):
        super(BirdClassifier, self).__init__()
        # load model
        self.model = models.resnet34(weights=models.ResNet50_Weights.DEFAULT)

        # freeze layers
        for param in model.parameters():
            param.requires_grad = False

        # adjust final fc layer with what we need
        num_classes = 200
        prev_in_features = model.fc.in_features
        self.model.fc = nn.Linear(prev_in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
# referenced W6 Demo doge.py for the train, evaluate, and duner main methods 
def train_bird_classification_model(dataloader, model, criterion, optimizer, best_loss, epoch, writer):
    for batch, (x, y) in enumerate(dataloader):
        # zero gradient
        optimizer.zero_grad()

        # forward pass
        prediction = model(x)

        # evaluate and backward pass
        loss = criterion(prediction, y)
        loss.backward()

        # update weights
        optimizer.step()

        if(loss < best_loss):
            best_loss = loss

            print("New best model found! Loss: ", loss.item(), " Saving...")

            torch.save({'epoch': epoch, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            })
        
        if(batch % 100 == 0):
            print(f"Batch {batch}: Loss = {loss.item():>7f}")
    
    # can print epoch time here if we want
    return model, best_loss

def evaluate_bird_classification_model(dataloader, model, criterion, epochs, writer):
    test_loss, correct, total = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            prediction = model(x)

            total += len(y)
            test_loss += criterion(prediction, y).item()
            correct += int((prediction.argmax(1) == y).type(torch.float).sum().item())
    # can print all the stats (test_loss, total, and correct) here if we want

if __name__ == "__main__":
    lr = 0.001
    momentum = 0.9
    best_loss = float('inf')
    epochs = None # TODO: change

    model = BirdClassifier()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if(torch.cuda.is_available()):
        device = torch.device('cuda')

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, momentum=momentum)

    # load best model if one exists
    if(os.path.exists(MODEL_PATH)):
        best_model = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model['model_state_dict'])
        optimizer.load_state_dict(best_model['optimizer_state_dict'])
        best_loss = best_model['best_loss']

    for epoch in range(epochs):
        # TODO: Fill in the parameters for the two methods below
        model, best_loss = train_bird_classification_model()
        evaluate_bird_classification_model()

