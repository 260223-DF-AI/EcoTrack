import os
import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision import transforms
from model.species_status import SpeciesStatuses
from model.BirdResNet import BirdResNet

MODEL_PATH = "model/weights/model50_90p_evalacc.pth" 
__classes: SpeciesStatuses = SpeciesStatuses()

# Transformations
std_transform = transforms.Compose([
	transforms.Resize((256, 256)),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize( #these are provided hardcoded values
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	) 
])
	
def load_model(model_path: str) -> BirdResNet:
	"""Loads BirdResNet model or raises an 
	args:
		model_path: str - the path of the BirdResNet model to load
	returns:
		a BirdResNet model if the path existed and the weights in the path aligns with what is in the model called"""
	# instantiate model
	model = BirdResNet(200)

	# pull specific model from path
	if os.path.exists(model_path):
		best_model = torch.load(model_path, weights_only=True)
		model.load_state_dict(best_model["model_state_dict"])

		# set to eval mode so it doesn't train
		model.eval()
	else:
		raise Exception(f"Model path does not exist or you're using a different ResNet version in your bird model than you saved your weights on.")
	return model

def get_classification(model: BirdResNet, img_content):
	img = Image.open(img_content).convert("RGB")

	with torch.no_grad():
		out = model(std_transform(img).unsqueeze(0))
		probabilities = functional.softmax(out, dim=1)
		confidence, output = torch.max(probabilities, dim=1)
		pred_species, endangered_status, multi = __classes[output.item() + 1]
		confidence = float(confidence.item()) * 100

		print(f"{pred_species} is {endangered_status}. Model was {confidence:.2f}% confident.")
		return pred_species, endangered_status, confidence

if __name__ == "__main__":
	model = load_model(MODEL_PATH)
	img_path = input("Enter the path to your image: ")
	if os.path.exists(img_path):
		with open(img_path, 'rb') as img_content:
			x, y, z = get_classification(model, img_path)