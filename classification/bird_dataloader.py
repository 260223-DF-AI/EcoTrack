import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path

class BirdDataset(Dataset):
    def __init__(self, transforms=None):
        pass
    
    def __len__(self):
        pass

    def __getitem__(self):
        pass

class BirdDataloader:
    pass

if __name__ == "__main__":
    bird_image_transform = transforms.Compose([
        # set shape to (3 x H x W), where H and W are expected to be at least 224
        transforms.Resize(224), # I am not sure what my mac can handle, but I think the smaller the better

        # load in to a range of [0, 1], ToTensor() automatically does this    
        transforms.ToTensor(),    

        # normalize to have a mean of [0.485, 0.456, 0.406] and std of [0.229, 0.224, 0.225]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
