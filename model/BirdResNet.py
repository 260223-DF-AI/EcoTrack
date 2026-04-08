import os
import sys
import time
import ssl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter # tensorboard --logdir=./runs/bird_logs
from torchvision import transforms
import torchvision.models as models

from PIL import Image

from warnings import deprecated

# Global Variables
DATA_ROOT = "data/CUB_200_2011/images"
LOG_DIR = "runs/bird_logs"
MODEL_PATH = "model/weights/birds.pth"
NUM_EPOCHS = 5
LEARNING_RATE = 0.01
PATIENCE = 100
BATCH_SIZE = 32

ssl._create_default_https_context = ssl._create_unverified_context


class BirdResNet(nn.Module):
    """
    Bird species classification model. Built on pretrained ResNet model.
    """
    def __init__(self, num_classes):
        super(BirdResNet, self).__init__()

        # Transfer Learning based on ResNet model
        # Options are 18, 34, 50, 101, and 151
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        # Freeze ResNet params
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer4

        # Replace final fully-connected linear layer with our own to fine-tune
        # Allows us to set our number of output classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        for name, param in self.model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True

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

def train_loop(dataloader, model, loss_fn, optimizer, epoch, writer, device, early_stop):
    """
    Train for one epoch
    """
    print()
    print(f"\n--- Training Epoch {epoch+1} ---")

    model.train()
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        x, y = x.to(device), y.to(device)
        # print(x.device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.item(), batch_idx)
        
        should_stop, improved = early_stop(loss.item())

        if improved:

            print(f"New best model at batch {batch_idx}: Loss = {loss.item():.4f}")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }, MODEL_PATH)

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}")

        if should_stop:
            return model, True
    
    end_time = time.time()
    print(f"Epoch {epoch + 1} completed: {batch_idx} batches processed")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return model, False

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
            train_test_image_map.append(int(line.replace('\n', '').split()[-1]))

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
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    # valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_data, test_data, train_loader, test_loader

class EarlyStopping:
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0 # number of batches w/o improvement
        self.early_stop = False

    def __call__(self, loss: float):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop, False

def load_model(model, optimizer, early_stop):
        """
        Load best model weights, optimizer state dict, and loss
        """
        best_model = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model["model_state_dict"])
        optimizer.load_state_dict(best_model["optimizer_state_dict"])
        early_stop.best_loss = best_model["loss"]
        print(f"Loaded best model from {MODEL_PATH}")
        return model, optimizer, early_stop


def main():

    # Identify best device to train on
    print(f"Cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda:0?
    if(torch.backends.mps.is_available()):
        device = torch.device('mps')

    print()
    print("--- Tensorboard Setup ---")
    writer = SummaryWriter(LOG_DIR)

    print()
    print("--- Create DataLoaders ---")
    train_data, test_data, train_loader, test_loader = load_data()

    # Test to make sure dataloaders working
    """
    # Random training example
    rand_idx = torch.randint(0, len(train_data), (1,)).item()
    sample_img, sample_label = train_data[rand_idx]
    sample_path = train_data.image_paths[rand_idx]
    # Log random sample to TensorBoard
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    sample_img_tb = (sample_img * std + mean).clamp(0, 1)

    writer.add_image("Samples/RandomTrain/Image", sample_img_tb, 0)
    writer.add_text("Samples/RandomTrain/Path", sample_path, 0)
    writer.add_text("Samples/RandomTrain/Label", str(sample_label), 0)
    writer.flush()

    print()
    print("--- Random Train Sample ---")
    print(f"Index: {rand_idx}")
    print(f"Path: {sample_path}")
    print(f"Label: {sample_label}")
    print(f"Image tensor shape: {tuple(sample_img.shape)}")
    """

    print()
    print("--- Instantiate Model ---")
    model = BirdResNet(len(train_data))
    model = model.to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    early_stop = EarlyStopping(PATIENCE)

    print("--- Load Best Model ---")
    if os.path.exists(MODEL_PATH):
        model, optimizer, early_stop = load_model(model, optimizer, early_stop)

    for epoch in range(NUM_EPOCHS):
        model, early_stopped = train_loop(train_loader, model, criterion, optimizer, epoch, writer, device, early_stop)
        if early_stopped:
            print("Broke early, loading best weights for eval")
            model, optimizer, early_stop = load_model(model, optimizer, early_stop)
        
        evaluate(test_loader, model, criterion, writer, device)

        if early_stopped:
            break

if __name__ == "__main__":
    main()