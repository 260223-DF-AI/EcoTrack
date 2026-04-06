import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from pathlib import Path

class BirdResNet(nn.Module):
    """
    TODO: ignore this class for now, finetuning RESNET model seems to be loading a model definied for us
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
    

class BirdDataset(Dataset):
    def __init__(self, image_paths: list[str], labels: list[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label


def main():

    lr = 0.001
    momentum = 0.9

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
                

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
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
    
    # 50-layer deep residual network with predefined weights. other options from 18, 34, 101, and 152
    model = resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(train_data.classes)) #reshaping output to match our number of classes

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, momentum=momentum)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda:0?
    if(torch.backends.mps.is_available()):
        device = torch.device('mps')

    model = model.to(device)

    #train

if __name__ == "__main__":
    main()