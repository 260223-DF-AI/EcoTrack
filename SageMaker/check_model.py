import os
import torch
import torch.nn.functional as functional
import numpy as np
from PIL import Image
from torchvision import transforms
from SageMaker.species_status import SpeciesStatuses
from SageMaker.src.AnimalResNet import AnimalResNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2


MODEL_PATH = "SageMaker/local_model/model.pth" 

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

def visualize_class_features(model: AnimalResNet, img_content):
    img = Image.open(img_content).convert("RGB")
    img_tensor = std_transform(img).unsqueeze(0)
	
    # for param in model.model.parameters():
    #     param.requires_grad = True
    target_layer = model.model.layer4[-1]

    outputs = model(img_tensor)
    pred_class = outputs.argmax(dim=1).item()

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor)[0]

    img_resized = img.resize((224, 224))
    rgb_img = np.array(img_resized).astype(np.float32) / 255.0

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    cv2.imshow(f"Grad-CAM (pred: {pred_class})", visualization)
    cv2.waitKey(0) # You need to click on the window with the image and press "0" to let your code move on
    cv2.destroyAllWindows()
	
def load_model(model_path: str) -> AnimalResNet:
	"""Loads AnimalResNet model or raises an exception
	args:
		model_path: str - the path of the AnimalResNet model to load
	returns:
		a AnimalResNet model if the path existed and the weights in the path aligns with what is in the model called"""
	# instantiate model
	model = AnimalResNet(90)

	device_type = "cuda" if torch.cuda.is_available() else "cpu"
	if(torch.backends.mps.is_available()):
		device_type = "mps"

	# pull specific model from path
	if os.path.exists(model_path):
		best_model = torch.load(model_path, weights_only=True, map_location=device_type)
		model.load_state_dict(best_model["model_state_dict"])

		# set to eval mode so it doesn't train
		model.eval()
	else:
		raise Exception(f"Model path does not exist or you're using a different ResNet version in your animal model than you saved your weights on.")
	return model

def get_classification(model: AnimalResNet, img_content):
	"""
	Uses the provided model to classify animal species and their endangered status from an input image
	args:
		model: AnimalResNet - model to communicate with (to?)
		img_content: can be a file path or bytes of the image
	returns:
		a tuple containing the species identified/predicted, the endangered status, multi, and the model's confidence for its output
	"""
	# get the image
	img = Image.open(img_content).convert("RGB")


	with torch.no_grad():
		out = model(std_transform(img).unsqueeze(0))
		probabilities = functional.softmax(out, dim=1) # scale probability distribution 
		confidence, output = torch.max(probabilities, dim=1) # get the highest probability and the label associated with it
		# print(f"Output: {output.item()}")
		label, all_statuses, endangered_status = __classes[output.item()] 
		confidence = float(confidence.item()) * 100

		print(f"Output was {output.item()}, {label} is {endangered_status}. Model was {confidence:.2f}% confident.")
		return label, all_statuses, endangered_status, confidence

if __name__ == "__main__":
	model = load_model(MODEL_PATH)
	i = 0
	# for root, dirs, files in os.walk("animals"):
		# if root == "animals": continue
	while True:
		img_path = input("Enter the path to your image: ")
		# img_path = os.path.join(root, files[2])
		if os.path.exists(img_path):
			with open(img_path, 'rb') as img_content:
				model.eval()
				x, y, z, a = get_classification(model, img_path)
				result = {
					'species' : x,
					'endangered_status': y,
					'multi': z,
					'confidence': a
				}
				# true_label = os.path.basename(root)
				# if x != true_label:
					# print(f"ERROR: {true_label} WAS PREDICTED AS {x} WITH {a:.2f}% CONFIDENCE")
				# visualize_class_features(model, img_path) # uncomment if you want to see where/what the model is focusing on
		else:
			continue