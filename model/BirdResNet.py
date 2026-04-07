import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.models as models

from PIL import Image

from warnings import deprecated

# Global Variables
DATA_ROOT = "data/CUB_200_2011/images"
LOG_DIR = "runs/bird_logs"
MODEL_PATH = "birds.pth"
NUM_EPOCHS = 1
LEARNING_RATE = 0.001


@deprecated("Model trains from scratch, use BirdResNet class instead for fine-tuning ResNet model")
class BirdModel(nn.Module):
    """
    ignore this class for now, finetuning RESNET model seems to be loading a model definied for us
    Keeping for the moment in case we end up training from scratch for whatever reason. 
    """
    def __init__(self, num_classes: int = 200):
        """
        
        """
        super(BirdResNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Linear(in_features=32, out_features=num_classes)
        )

    def forward(self, x):
        """
        Forward pass of data through our sequential layers
        """
        return self.layers(x)
    

class BirdResNet(nn.Module):
    """
    Bird species classification model. Built on pretrained ResNet model.
    """
    def __init__(self, num_classes):
        super(BirdResNet, self).__init__()

        # Transfer Learning based on ResNet model
        # Options are 18, 34, 50, 101, and 151
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze ResNet params
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace final fully-connected linear layer with our own to fine-tune
        # Allows us to set our number of output classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class BirdDataset(Dataset):
    """
    Bird image dataset
    """
    def __init__(self, image_paths: list[str], labels: list[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = f"{DATA_ROOT}/{self.image_paths[idx]}"
        label = self.labels[idx]

        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_loop(dataloader, model, loss_fn, optimizer, epoch, best_loss, writer, device):
    """
    Train for one epoch
    """
    print()
    print(f"\n--- Training Epoch {epoch+1} ---")

    model.train()
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.item(), batch_idx)

        if loss < best_loss:
            best_loss = loss

            print(f"Saving new best model: Loss = {loss.item()}")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }, MODEL_PATH)

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():>7f}")
    
    end_time = time.time()
    print(f"Epoch {epoch + 1} completed: {batch_idx} batches processed")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return model, best_loss

def evaluate(dataloader, model, loss_fn, writer, device):
    """
    Evaluate after one epoch
    """
    print()
    print("--- Eval Model ---")

    test_loss, correct, total = 0, 0, 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader, 1):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total += len(y)
            test_loss += loss_fn(pred, y).item()
            correct += int((pred.argmax(1) == y).type(torch.float).sum().item())
            if batch_idx == 10: break
        
    writer.add_scalar("Loss/test", test_loss / total)
    print("Total Samples: ", total)
    print("Correct Predictions: ", correct)
    print(f"Test Loss: {test_loss / total:.4f}")
    print(f"Evaluation: Accuracy = {(100 * correct / total):.2f}%")

def load_data():
    """
    Create our train and test dataloaders
    """
    # Set up train and test datasets and dataloaders
    train_test_image_map = []

    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []

    with open("./data/CUB_200_2011/train_test_split.txt", 'r', encoding='utf-8') as f:
        for line in f:
            train_test_image_map.append(int(line[-1]))

    with open("./data/CUB_200_2011/images.txt", 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            _, path = line.split()
            label = int(path[:3])
            if train_test_image_map[idx] == 1:
                train_labels.append(label)
                train_paths.append(path)
            else:
                test_labels.append(label)
                test_paths.append(path)
                
    # Standardize our image sizes for the model, while applying random transformations to strengthen model accuracy
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(45),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( #these are provided hardcoded values
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create dataset objects
    train_data = BirdDataset(train_paths, train_labels, transform=transform)
    test_data = BirdDataset(test_paths, test_labels, transform=transform)
    # valid_data = BirdDataset(valid_paths, valid_labels, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    # valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
    
    return train_data, test_data, train_loader, test_loader


def main():

    # Identify best device to train on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda:0?
    if(torch.backends.mps.is_available()):
        device = torch.device('mps')

    print()
    print("--- Tensorboard Setup ---")
    writer = SummaryWriter(LOG_DIR)

    print()
    print("--- Create DataLoaders ---")
    train_data, test_data, train_loader, test_loader = load_data()

    print()
    print("--- Instantiate Model ---")
    model = BirdResNet(len(train_data.classes))
    best_loss = float("inf")

    print("Adding graph to tensorboard...")
    writer.add_graph(model, )

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    print("--- Load Best Model ---")
    if os.path.exists(MODEL_PATH):
        best_model = torch.loa(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model["model_state_dict"])
        optimizer.load_state_dict(best_model["optimizer_state_dict"])
        best_loss = best_model["loss"]
        print(f"Loaded best model from {MODEL_PATH}")

    for epoch in range(NUM_EPOCHS):
        model, best_loss = train_loop(train_loader, model, criterion, optimizer, epoch, best_loss, writer, device)
        evaluate(test_loader, model, criterion, writer, device)

if __name__ == "__main__":
    main()