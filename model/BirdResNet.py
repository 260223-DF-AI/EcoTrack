import os
import io
import ssl
import math
import random
import time
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter # tensorboard --logdir=./runs/bird_logs
from torch.amp import autocast, GradScaler
from torchvision import transforms
import torchvision.models as models

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from species_status import SpeciesStatuses

# Global Variables
DATA_ROOT = "data/CUB_200_2011/images"
ANIMALS_ROOT = "data/animals"
LOG_DIR = "runs/bird_logs"
MODEL_PATH = "model/weights/model.pth"
BEST_MODEL_PATH = "model/weights/best.pth"
NUM_EPOCHS = 50
# LEARNING_RATE_4 = 0.001
LEARNING_RATE_4 = 0.00001
# LEARNING_RATE_4 = 0.0000001
# LEARNING_RATE_FC = 0.01
LEARNING_RATE_FC = 0.0001
# LEARNING_RATE_FC = 0.000001
PATIENCE = 15

ssl._create_default_https_context = ssl._create_unverified_context


class BirdResNet(nn.Module):
    """
    Bird species classification model. Built on pretrained ResNet model.
    """
    def __init__(self, num_classes):
        super(BirdResNet, self).__init__()

        # Transfer Learning based on ResNet model
        # Options are 18, 34, 50, 101, and 152
        # self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        # Freeze ResNet params
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        # for param in self.model.layer3.parameters():
        #     param.requires_grad = True

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
        self.classes = len(set(labels))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
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

def load_data(BATCH_SIZE: int = 32):
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
            class_images[label].append(os.path.join(DATA_ROOT, path))

    # Load all animal images for testing
    class_images[len(class_images)] = [] # should be 200
    # animal_images = []
    if os.path.exists(ANIMALS_ROOT):
        for root, dirs, files in os.walk(ANIMALS_ROOT):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_images[200].append(os.path.join(root, file))

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

def load_model(model, optimizer, early_stop, device_type):
        """
        Load best model weights, optimizer state dict, and loss
        """
        best_model = torch.load(BEST_MODEL_PATH, weights_only=True, map_location=device_type)
        model.load_state_dict(best_model["model_state_dict"])
        optimizer.load_state_dict(best_model["optimizer_state_dict"])
        early_stop.best_loss = best_model["loss"]
        # early_stop_train.best_loss = best_model["train_loss"]
        # early_stop_test.best_loss = best_model["test_loss"]
        print(f"Loaded best model from {BEST_MODEL_PATH}")
        # return model, optimizer, early_stop_train, early_stop_test
        return model, optimizer, early_stop

def train_loop(dataloader, model, loss_fn, best_loss, optimizer, scaler, writer, device, device_type, amp: bool = False):
    """
    Train for one epoch
    """

    print(f"Using AMP: {amp}")

    model.train()
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        # print(f"Batch {batch_idx}")
        x, y = x.to(device), y.to(device)
        # print(x.device)
        optimizer.zero_grad()
        # print("Cast")
        if amp:
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
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        
    test_loss /= total

    should_stop, improved = early_stop(test_loss)

    if improved: #leaving early stop in here for now but it should be initiated by eval first
        print(f"New best model: Loss = {test_loss:.4f}")
        
    writer.add_scalar("Loss/test", test_loss)
    print("Total Samples: ", total)
    print("Correct Predictions: ", correct)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Evaluation: Accuracy = {(100 * correct / total):.2f}%")

    return improved, should_stop, test_loss

def validate(dataloader, model, loss_fn, writer, device, classes):
    """
    Evaluate after one epoch
    """
    print()
    print("--- Final Validation ---")

    species_statuses = SpeciesStatuses()
    status_counts = {}

    for value in species_statuses.statuses.values():
        status_counts[value] = {
            "correct" : 0,
            "incorrect" : 0,
            "total" : 0,
            "confidence" : 0.0
        }

    # initialize variables for confusion matrix
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    all_y_pred = []
    all_y_true = []

    valid_loss, correct, total = 0, 0, 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader, 1):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            probabilities = nn.functional.softmax(pred, dim=1)
            max_probs = torch.max(probabilities, dim=1)[0]
            _, predictions = torch.max(pred, 1) # for confusion matrix
            batch_size = y.size(0)
            total += batch_size
            valid_loss += loss_fn(pred, y).item() * batch_size
            correct += int((pred.argmax(1) == y).type(torch.float).sum().item())

            # if batch_idx == 100:
                # print(f"Batch {batch_idx}")

            # for confusion matrix
            all_y_pred.extend(predictions.cpu().numpy())
            all_y_true.extend(y.cpu().numpy())
            for label, prediction, prob in zip(y, predictions, max_probs):
                status = species_statuses[label.item()+1][1]
                status_p = species_statuses[prediction.item()+1][1]
                if label == prediction:
                    correct_pred[classes[label.item()]] += 1
                    status_counts[status]["correct"] +=1
                else:
                    status_counts[status_p]["incorrect"] +=1
                    status_counts[status_p]["confidence"] += prob

                total_pred[classes[label.item()]] += 1
                status_counts[status]["total"] +=1
    incorrect = total - correct

    for key, value in status_counts.items():
        print()
        try:
            accuracy = 100*status_counts[key]["correct"] / status_counts[key]["total"]
        except:
            accuracy = -1
        try:
            incorrect_p = 100*status_counts[key]["incorrect"] / incorrect
        except:
            incorrect_p = -1
        try:
            incorrect_c = 100*status_counts[key]["confidence"] / status_counts[key]["incorrect"]
        except:
            incorrect_c = -1
        status_counts[key]["accuracy"] = accuracy
        print(f"{key}: {accuracy:.2f}% accruacy")
        print(f"{key}: {incorrect_p:.2f}% of incorrect predictions fell under this category")
        print(f"{key}: {incorrect_c:.2f}% average confidence of predictions incorrectly identifying this category")

    # create the confusion matrix and store it in a dataframe
    cf_matrix = confusion_matrix(all_y_true, all_y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    
    valid_loss /= total
    
    writer.add_scalar("Loss/valid", valid_loss)
    print("Total Samples: ", total)
    print("Correct Predictions: ", correct)
    print(f"Valid Loss: {valid_loss:.4f}")
    print(f"Validation: Accuracy = {(100 * correct / total):.2f}%")

    return df_cm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batchsize', type=int, default=32)

    args, _ = parser.parse_known_args()
    BATCH_SIZE = args.batchsize


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
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"runs/bird_logs/{run_name}")

    df_cm = pd.DataFrame()

    print()
    print("--- Create DataLoaders ---")
    train_data, train_loader, test_loader, valid_loader = load_data(BATCH_SIZE)

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

    optimizer = optim.Adam([
        # {'params': filter(lambda p: p.requires_grad, model.model.layer3.parameters()),
        # 'lr': 0},
        {'params': filter(lambda p: p.requires_grad, model.model.layer4.parameters()),
        'lr': LEARNING_RATE_4},
        {'params': filter(lambda p: p.requires_grad, model.model.fc.parameters()),
        'lr': LEARNING_RATE_FC}
    ])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.075)
    best_loss = float('inf')
    early_stop = EarlyStopping(PATIENCE)

    print()
    print("--- Load Best Model ---")
    if os.path.exists(BEST_MODEL_PATH):
        model, optimizer, early_stop = load_model(model, optimizer, early_stop, device_type)

    print()
    print(f"Batches per epoch: {math.ceil(len(train_data) / BATCH_SIZE)}")
    for epoch in range(1, NUM_EPOCHS+1):
        break # uncomment this to just run validation
        print()
        print(f"\n--- Training Epoch {epoch} ---")
        model, optimizer, best_loss = train_loop(train_loader, model, criterion, best_loss, optimizer, scaler, writer, device, device_type)
        
        improved, early_stopped, test_loss = evaluate(test_loader, model, criterion, writer, device, early_stop)
        # scheduler.step(test_loss)
        if improved:
            print("New best, saving best weights")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": test_loss,
            }, BEST_MODEL_PATH)

        if early_stopped:
            print(f"No improvements in {PATIENCE} epochs. Ending training early.")
            break

    df_cm = validate(valid_loader, model, criterion, writer, device,list(set(train_data.labels)))

    print("Now displaying confusion matrix for last executed epoch")
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.savefig('bird_confusion_matrix.png')
    # Save to TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.show()
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = transforms.ToTensor()(image)
    writer.add_image('Confusion Matrix', image_tensor)
    plt.close()


if __name__ == "__main__":
    main()