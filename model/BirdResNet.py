import os
import time
import ssl
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter # tensorboard --logdir=./runs/bird_logs
from torch.amp import autocast, GradScaler
from torchvision import transforms
import torchvision.models as models

from PIL import Image

# Global Variables
DATA_ROOT = "data/CUB_200_2011/images"
LOG_DIR = "runs/bird_logs"
MODEL_PATH = "model/weights/model.pth"
BEST_MODEL_PATH = "model/weights/best.pth"
NUM_EPOCHS = 5
LEARNING_RATE = 0.01
PATIENCE = 2
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
        # self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        # Freeze ResNet params
        for param in self.model.parameters():
            param.requires_grad = False

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
        self.classes = len(set(labels))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = f"{DATA_ROOT}/{self.image_paths[idx]}"
        label = self.labels[idx]

        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label

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

def load_data():
    """
    Create our train and test dataloaders
    """
    # Set up train and test datasets and dataloaders
    # train_test_image_map = []
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    valid_paths, valid_labels = [], []

    # Read all images and group by folder/class
    class_images = {}
    with open("./data/CUB_200_2011/images.txt", 'r', encoding='utf-8') as f:
        for line in f:
            _, path = line.replace('\n', '').split()
            folder = path.split('/')[0]
            label = int(folder[:3]) - 1
            
            if label not in class_images:
                class_images[label] = []
            class_images[label].append(path)
    
    # Split each class 80/10/10
    for label in sorted(class_images.keys()):
        paths = class_images[label]
        random.shuffle(paths)
        
        train_count = math.ceil(len(paths) * 0.8)
        test_count = math.ceil(len(paths) * 0.1)
        
        train_paths.extend(paths[:train_count])
        train_labels.extend([label] * train_count)
        
        test_paths.extend(paths[train_count:train_count + test_count])
        test_labels.extend([label] * test_count)
        
        valid_paths.extend(paths[train_count + test_count:])
        valid_labels.extend([label] * (len(paths) - train_count - test_count))

                
    # Standardize our image sizes for the model, while applying random transformations to strengthen model accuracy
    train_transform = transforms.Compose([
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

    std_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( #these are provided hardcoded values
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) 
    ])

    # Create dataset objects
    train_data = BirdDataset(train_paths, train_labels, transform=train_transform)
    test_data = BirdDataset(test_paths, test_labels, transform=std_transform)
    valid_data = BirdDataset(valid_paths, valid_labels, transform=std_transform)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False) #batch size won't matter if running through all data
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train data count: {len(train_data)}; Classes: {train_data.classes}")
    print(f"Test data count: {len(test_data)}; Classes: {test_data.classes}")
    print(f"Valid data count: {len(valid_data)}; Classes: {valid_data.classes}")
    
    return train_data, train_loader, test_loader, valid_loader

def load_model(model, optimizer, early_stop):
        """
        Load best model weights, optimizer state dict, and loss
        """
        best_model = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model["model_state_dict"])
        optimizer.load_state_dict(best_model["optimizer_state_dict"])
        early_stop.best_loss = best_model["loss"]
        # early_stop_train.best_loss = best_model["train_loss"]
        # early_stop_test.best_loss = best_model["test_loss"]
        print(f"Loaded best model from {MODEL_PATH}")
        # return model, optimizer, early_stop_train, early_stop_test
        return model, optimizer, early_stop

def train_loop(dataloader, model, loss_fn, best_loss, optimizer, scaler, writer, device, device_type):
    """
    Train for one epoch
    """

    model.train()
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        # print(f"Batch {batch_idx}")
        x, y = x.to(device), y.to(device)
        # print(x.device)
        optimizer.zero_grad()
        # print("Cast")
        with autocast(device_type):
            # print("Predict")
            pred = model(x)
            # print("Loss")
            loss = loss_fn(pred, y)
        # print("Scale")
        scaler.scale(loss).backward()
        # print("Unscale")
        scaler.unscale_(optimizer)
        # print("Clip")
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # print("Step")
        scaler.step(optimizer)
        # print("Update")
        scaler.update()

        writer.add_scalar("Loss/train", loss.item(), batch_idx)

        if loss.item() < best_loss: #leaving early stop in here for now but it should be initiated by eval first
            best_loss = loss.item()
            print(f"New best training loss at batch {batch_idx}: Loss = {loss.item():.4f}")

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }, MODEL_PATH)
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}")
    
    end_time = time.time()
    print(f"Epoch completed: {batch_idx} batches processed")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return model, optimizer, best_loss

def evaluate(dataloader, model, loss_fn, writer, device, early_stop):
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
            batch_size = y.size(0)
            total += batch_size
            test_loss += loss_fn(pred, y).item() * batch_size
            correct += int((pred.argmax(1) == y).type(torch.float).sum().item())
            # if batch_idx == 100:
                # print(f"Batch {batch_idx}")
    
    test_loss /= total

    should_stop, improved = early_stop(test_loss)

    if improved: #leaving early stop in here for now but it should be initiated by eval first
        print(f"New best model: Loss = {test_loss:.4f}")
        
    writer.add_scalar("Loss/test", test_loss)
    print("Total Samples: ", total)
    print("Correct Predictions: ", correct)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Evaluation: Accuracy = {(100 * correct / total):.2f}%")

    return should_stop, test_loss

def validate(dataloader, model, loss_fn, writer, device):
    """
    Evaluate after one epoch
    """
    print()
    print("--- Final Validation ---")

    valid_loss, correct, total = 0, 0, 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader, 1):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            batch_size = y.size(0)
            total += batch_size
            valid_loss += loss_fn(pred, y).item() * batch_size
            correct += int((pred.argmax(1) == y).type(torch.float).sum().item())
            # if batch_idx == 100:
                # print(f"Batch {batch_idx}")
    
    valid_loss /= total
    
    writer.add_scalar("Loss/valid", valid_loss)
    print("Total Samples: ", total)
    print("Correct Predictions: ", correct)
    print(f"Valid Loss: {valid_loss:.4f}")
    print(f"Validation: Accuracy = {(100 * correct / total):.2f}%")

def main():

    # Manually set random seed for reproducibility   
    # torch.manual_seed(327)
    # torch.backends.cudnn.deterministic = True

    # Identify best device to train on
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if(torch.backends.mps.is_available()):
        device_type = "mps"
    device = torch.device(device_type) # cuda:0?
    print(f"Device: {device_type}")

    scaler = GradScaler(device_type)

    print()
    print("--- Tensorboard Setup ---")
    writer = SummaryWriter(LOG_DIR)

    print()
    print("--- Create DataLoaders ---")
    train_data, train_loader, test_loader, valid_loader = load_data()

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
    model = BirdResNet(train_data.classes)
    model = model.to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    early_stop = EarlyStopping(PATIENCE)

    print()
    print("--- Load Best Model ---")
    if os.path.exists(BEST_MODEL_PATH):
        model, optimizer, early_stop = load_model(model, optimizer, early_stop)

    print()
    print(f"Batches per epoch: {math.ceil(len(train_data) / BATCH_SIZE)}")
    for epoch in range(1, NUM_EPOCHS+1):
        print()
        print(f"\n--- Training Epoch {epoch} ---")
        model, optimizer, best_loss = train_loop(train_loader, model, criterion, best_loss, optimizer, scaler, writer, device, device_type)
        
        early_stopped, test_loss = evaluate(test_loader, model, criterion, writer, device, early_stop)

        if early_stopped:
            print("Broke early, saving best weights")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": test_loss,
            }, BEST_MODEL_PATH)
            break

    validate(valid_loader, model, criterion, writer, device)

if __name__ == "__main__":
    main()