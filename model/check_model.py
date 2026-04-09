import os
import torch
import torchvision.models as models
import torch.optim as optim

MODEL_PATH = "model/weights/best91p_validacc.pth"

model = models.resnet101()
optimizer = optim.Adam()
if os.path.exists(MODEL_PATH):
        best_model = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model["model_state_dict"])
        optimizer.load_state_dict(best_model["optimizer_state_dict"])