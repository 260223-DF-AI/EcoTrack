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


def upload(use_gpu: bool = False):
    print("Initial Setup...")
    load_dotenv() # ARN from .env
    # USE_GPU = False
    # TRAIN_DEVICE = 'ml.g4dn.xlarge' if use_gpu else 'ml.m5.large'
    LOCAL_MODEL_DIR = 'local_model'
    TAR_NAME = 'model.tar.gz'
    # print(f"Training on {TRAIN_DEVICE}")

    # This comment is where we would call training if we were offloading it to SageMaker
    # Since we're using the weights we've already trained locally we can skip to deploying as a tar.gz

    # Model.pth weights file location
    os.makedirs(LOCAL_MODEL_DIR, exists_ok=True)
    model_path = os.path.join(LOCAL_MODEL_DIR, 'model.pth')

    code_dir = os.path.join(LOCAL_MODEL_DIR, 'code') #prep the path for the package
    os.makedirs(code_dir, exist_ok=True)

    if os.path.exists('src/inference.py'):
        shutil.copy('src/inference.py', os.path.join(code_dir, 'inference.py')) #copy the inference script

    with tarfile.open(TAR_NAME, "w:gz") as tar: #create the archive
        tar.add(model_path, arcname='model.pth')
        tar.add(code_dir, arcname='code')

    print(f"Saved model to {TAR_NAME}")


    print("Uploading to S3...")
    try:
        session = sagemaker.Session()
        try:
            role = sagemaker.get_execution_role()
        except (ValueError, RuntimeError): #if not running on SageMaker
            role = ARN
        bucket = session.default_bucket()
        print(f"Bucket: {bucket}")
    except Exception as e:
        print(e)
        exit(1)

    s3_prefix = 'EcoTrack'
    s3_model_path = session.upload_data(path=TAR_NAME, bucket=bucket, key_prefix=s3_prefix)

    print(f"Uploaded model to {s3_model_path}")

def deploy():
    """Deploy model from S3 bucket"""
    load_dotenv() # ARN from .env
    try:
        session = sagemaker.Session()
        try:
            role = sagemaker.get_execution_role()
        except (ValueError, RuntimeError): #if not running on SageMaker
            role = ARN
        bucket = session.default_bucket
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
    return pytorch_model

def predict(pytorch_model: PyTorchModel, input):

    DEPLOY_DEVICE = 'ml.m5.large'

    print("Creating new predictor...")
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type=DEPLOY_DEVICE,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )
    print(f"Endpoint Name: {predictor.endpoint_name}")


    print("Predicting...")

    confidence, label = predictor.predict(input) #TODO: make sure the input is formatted correctly (taking in an image)
    # print("Result:\n" + response)
    return confidence, label #feed label to a speciesstatus object to get information, dont put in here or else we'll have to make a new object constantly


def shutdown(predictor):
    # TODO: include this in a "finally" block in a try except satatement when predicting
    """Shut down endpoint after being done with predictions to avoid incurring too many costs"""
    print("Shutting down endpoint...")
    predictor.delete_endpoint()
    print("Endpoint shut down.")


if __name__ == "__main__":
    upload()
    # deploy()
    # predict()