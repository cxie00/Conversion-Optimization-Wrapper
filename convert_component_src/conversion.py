import argparse
import os
from datetime import datetime

from hummingbird.ml import convert as hb_convert
import mlflow
import numpy as np
import torch

import yaml 
from yaml.loader import SafeLoader


def read_load_model(model_folder):
    """
    returns the instance of the input MLflow model by reading its flavors.
    params:
        model_folder: string path to the MLmodel file of the MLflow model. Must be like "path\to\run\artifacts\model_folder" 
    """
    mlmodel_file = model_folder + "/MLmodel"

    with open(mlmodel_file) as f:
        data = yaml.load(f, Loader=SafeLoader)
        for flavor in data["flavors"]:
            if (flavor == "sklearn"):
                print("found sklearn")
                return mlflow.sklearn.load_model(model_folder)
            elif (flavor == "pytorch"):
                return mlflow.pytorch.load_model(model_folder)
            elif (flavor == "onnx"):
                return mlflow.onnx.load_model(model_folder)
            
        print("No skl, pytorch, or onnx model found.")
        return

def convert(premodel, target, input_data=0):
    """
    Converts an MLflow model to a target framework and returns an MLflow model in the target framework.
    
    params:
        premodel: Model with Sklearn or Pytorch backend. 
        target: str of target backend to be converted to, such as ONNX and Pytorch.
            "torch" for Pytorch.
            "onnx" for ONNX. 
        test_input: The *onnx* backend requires either a test_input of a the initial types set through the exta_config parameter.

    returns:
        an MLflow model in the target framework.
    """
    assert premodel is not None
    assert (target == 'onnx' or target == 'torch')
   
    if (target == "onnx"):
        model = hb_convert(premodel, 'onnx', test_input=input_data)
        pred = model.predict(input_data)
        sig = mlflow.models.infer_signature(input_data, pred)
        mlflow.onnx.log_model(model.model, 'onnx_model', input_example=input_data, signature=sig)
    elif (target == "torch" or target == "pytorch"):
        model = hb_convert(premodel, 'torch')
        # model.to('cuda')
        pred = model.predict(input_data)
        sig = mlflow.models.infer_signature(input_data, pred)
        mlflow.pytorch.log_model(model.model, 'torch_model', input_example=input_data, signature=sig)

    return model

# input must be a link to the folder of the logged model for this to work
parser = argparse.ArgumentParser()
parser.add_argument("--conversion_component_input", type=str)
parser.add_argument("--conversion_target_input", type=str)
parser.add_argument("--conversion_component_output", type=str)

args = parser.parse_args()

print("conversion_component_input path: %s" % args.conversion_component_input)
print("conversion_target_input: %s" % args.conversion_target_input)
print("conversion_component_output path: %s" % args.conversion_component_output)

premodel = read_load_model(args.conversion_component_input)

if (args.conversion_target_input == "onnx"):
    # load the test inputs for this dataset with  mlflow
    ml_model = mlflow.models.Model.load(path=args.conversion_component_input)
    input_example =  ml_model.load_input_example(path=args.conversion_component_input)

    model = convert(premodel, args.conversion_target_input, input_example)
else: 
    model = convert(premodel, args.conversion_target_input)
