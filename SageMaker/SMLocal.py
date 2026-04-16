import os
import shutil
import sagemaker
import tarfile
from dotenv import load_dotenv
# import torch
# import torch.nn as nn
# import torch.optim as optim
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
# from src.AnimalResNet import AnimalResNet
# import argparse
from species_status import SpeciesStatuses
from PIL import Image
from torchvision import transforms


def upload(use_gpu: bool = False):
    print("Initial Setup...")
    load_dotenv() # ARN from .env
    # USE_GPU = False
    # TRAIN_DEVICE = 'ml.g4dn.xlarge' if use_gpu else 'ml.m5.large'
    LOCAL_MODEL_DIR = 'SageMaker/local_model'
    TAR_NAME = 'model.tar.gz'
    # print(f"Training on {TRAIN_DEVICE}")

    # This comment is where we would call training if we were offloading it to SageMaker
    # Since we're using the weights we've already trained locally we can skip to deploying as a tar.gz

    # Model.pth weights file location
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(LOCAL_MODEL_DIR, 'model.pth')

    code_dir = os.path.join(LOCAL_MODEL_DIR, 'code') #prep the path for the package
    os.makedirs(code_dir, exist_ok=True)

    src_dir = 'SageMaker/src'
    if os.path.isdir(src_dir):
        # Package all Python modules used by inference.py to avoid import-time worker crashes.
        for filename in os.listdir(src_dir):
            if filename.endswith('.py'):
                shutil.copy(os.path.join(src_dir, filename), os.path.join(code_dir, filename))
    
    with tarfile.open(f"SageMaker/{TAR_NAME}", "w:gz") as tar: #create the archive
        tar.add(model_path, arcname='model.pth')
        tar.add(code_dir, arcname='code')

    print(f"Saved model to {TAR_NAME}")


    print("Uploading to S3...")
    try:
        session = sagemaker.Session()
        bucket = session.default_bucket()
        print(f"Bucket: {bucket}")
    except Exception as e:
        print(e)
        exit(1)

    s3_prefix = 'EcoTrack'
    s3_model_path = session.upload_data(path=f"SageMaker/{TAR_NAME}", bucket=bucket, key_prefix=s3_prefix)

    print(f"Uploaded model to {s3_model_path}")

def deploy():
    """Deploy model from S3 bucket"""
    load_dotenv() # ARN from .env
    try:
        session = sagemaker.Session()
        try:
            role = sagemaker.get_execution_role()
        except (ValueError, RuntimeError): #if not running on SageMaker
            role = os.getenv('ARN')
        bucket = session.default_bucket()
    except Exception as e:
        print(e)
        exit(1)

    s3_prefix = 'EcoTrack'
    TAR_NAME = 'model.tar.gz'
    s3_model_path = f"s3://{bucket}/{s3_prefix}/{TAR_NAME}" #TODO find path of model after upload?

    print("Deploying...")
    #Instantiate PyTorch model
    pytorch_model = PyTorchModel(
        model_data=s3_model_path,
        role=role,
        framework_version='2.0.0',
        py_version='py310',
        entry_point='inference.py',
        sagemaker_session=session
    )
    print("Deployed!")

    DEPLOY_DEVICE = 'ml.m5.large'

    print("Creating new predictor...")
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type=DEPLOY_DEVICE,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )
    print(f"Endpoint Name: {predictor.endpoint_name}")

    return predictor

def predict(predictor, input_data):
    print("Predicting...")

    # JSONSerializer cannot handle torch tensors directly.
    payload = input_data.detach().cpu().tolist()
    response = predictor.predict(payload)

    if isinstance(response, dict):
        confidence = float(response["confidence"])
        label = int(response["label"])
    elif isinstance(response, (list, tuple)) and len(response) >= 2:
        confidence = float(response[0])
        label = int(response[1])
    else:
        raise ValueError(f"Unexpected prediction response format: {response}")

    # print("Result:\n" + response)
    return confidence, label #feed label to a speciesstatus object to get information, dont put in here or else we'll have to make a new object constantly


def shutdown(predictor):
    # TODO: include this in a "finally" block in a try except satatement when predicting
    """Shut down endpoint after being done with predictions to avoid incurring too many costs"""
    print("Shutting down endpoint...")
    predictor.delete_endpoint()
    print("Endpoint shut down.")


if __name__ == "__main__":

    # upload()

    species_statuses = SpeciesStatuses()

    std_transform = transforms.Compose([
    	transforms.Resize((256, 256)),
    	transforms.CenterCrop(224),
    	transforms.ToTensor(),
    	transforms.Normalize( #these are provided hardcoded values
    		mean=[0.485, 0.456, 0.406],
    		std=[0.229, 0.224, 0.225]
    	) 
    ])

    predictor = deploy()

    try:
        while True:
            img_path = input("Enter the path to your image: ")
            if img_path.lower() in ['n', 'no', 'exit', 'q', 'quit']:
                break
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
            else:
                print("Image does not exist at that location")
                continue

            input_data = std_transform(img).unsqueeze(0)
            confidence, label = predict(predictor, input_data)
            print(f"Class: {species_statuses[label]}, %{confidence:.2f} confidence.")
    except Exception as e:
        print(e)
    finally:
        shutdown(predictor)