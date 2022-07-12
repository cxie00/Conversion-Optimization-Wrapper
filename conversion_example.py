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
    Returns the instance of the input MLflow model by reading its flavors.
    
    params:
        model_folder: string path to the MLmodel file of the MLflow model. Must be like "path\to\run\artifacts\flavor\MLmodel" 
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

def convert_model(premodel, target, input_data= 0):
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
        mlflow.onnx.log_model(model.model, 'onnx_model', input_example=input_data)
    elif (target == "torch" or target == "pytorch"):
        model = hb_convert(premodel, 'torch')
        # model.to('cuda')
        mlflow.pytorch.log_model(model.model, 'torch_model')

    return model


"""
Example:
Make sure you pip install hummingbird-ml, torch, onnx, onnxruntime==1.9.0, and protobuf==3.20 and whatever else import error you get.
Then in terminal, run: python conversion_example.py
"""

# # load the mlflow input model. this is a skl RandomForestClassifier model
# premodel = read_load_model("conversionAPI\mlruns\\0\c59e68a72e5745659fa156a6c1836428\\artifacts\skl")

# # load the test inputs for this dataset with  mlflow
# ml_model = mlflow.models.Model.load("conversionAPI\mlruns\\0\c59e68a72e5745659fa156a6c1836428\\artifacts\skl")
# input_example =  ml_model.load_input_example("conversionAPI\mlruns\\0\c59e68a72e5745659fa156a6c1836428\\artifacts\skl")
# sig = mlflow.models.infer_signature()
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from hummingbird.ml import convert
from sklearn.metrics import accuracy_score
import mlflow
import time

# convert the mlflow input model
X, y = load_breast_cancer(return_X_y=True)
skl_model = RandomForestClassifier(n_estimators=500, max_depth=7)
skl_model.fit(X, y)
pred = skl_model.predict(X)
sig = mlflow.models.infer_signature(X, pred)
mlflow.sklearn.log_model(skl_model, 'pre_skl_model', input_example=X, signature=sig)

model = convert_model(skl_model, "torch", X)

