import os
import json
import torch
import torch.nn as nn
import torchvision.models as models


class AnimalResNet(nn.Module):
    """
    Animal species classification model. Built on pretrained ResNet model.
    """
    def __init__(self, num_classes, pretrained: bool = True):
        super(AnimalResNet, self).__init__()

        # Transfer Learning based on ResNet model
        # Options are 18, 34, 50, 101, and 152
        # self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        if pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.model = models.resnet50(weights=None)
        # self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        # Freeze ResNet params
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace final fully-connected linear layer with our own to fine-tune
        # Allows us to set our number of output classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def model_fn(model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(model_dir, 'model.pth')

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format in model.pth")

    # Infer class count from the checkpoint so the inference head matches training.
    fc_weight = state_dict.get("model.fc.weight")
    print(f"FC Weight: {fc_weight}")
    num_classes = int(fc_weight.shape[0]) if fc_weight is not None else 90
    print(num_classes)

    model = AnimalResNet(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        body = request_body.decode('utf-8') if isinstance(request_body, bytes) else request_body
        data = json.loads(body)

        # Allow either a raw nested list or a dict wrapper such as {"inputs": ...}.
        if isinstance(data, dict):
            data = data.get('inputs', data.get('instances'))

        if data is None:
            raise ValueError("JSON payload must include image tensor data.")

        input_tensor = torch.tensor(data, dtype=torch.float32)
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        if input_tensor.ndim != 4:
            raise ValueError(f"Expected input shape [N, C, H, W], got {tuple(input_tensor.shape)}")

        return input_tensor
    
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
        return {
            "confidence": float(confidence.item()) * 100.0,
            "label": int(output.item())
        }
    
    except Exception as e:
        print(f"Exception occured: {e}")
        raise e

def output_fn(prediction, content_type):
    return json.dumps(prediction)