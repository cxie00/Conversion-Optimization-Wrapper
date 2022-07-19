import argparse
import os
from datetime import datetime

from hummingbird.ml import convert as hb_convert
import mlflow
import onnx
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

def convert_model(premodel, target, input_data=0):
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



"""
Example:
Make sure you pip install hummingbird-ml, torch, onnx, onnxruntime==1.9.0, and protobuf==3.20 and whatever else import error you get.
Then in terminal, run: python conversion_example.py
"""


# convert the mlflow input model
# X, y = load_breast_cancer(return_X_y=True)
# skl_model = RandomForestClassifier(n_estimators=500, max_depth=7)
# skl_model.fit(X, y)
# pred = skl_model.predict(X)
# sig = mlflow.models.infer_signature(X, pred)
# mlflow.sklearn.log_model(skl_model, 'pre_skl_model', input_example=X, signature=sig)

# model = convert_model(skl_model, "torch", X)
model = mlflow.sklearn.load_model("mlruns\\0\\b81d20537a9345cf9126d7c8bdc7a2c6\\artifacts\pre_skl_model")
ml_model = mlflow.models.Model.load("mlruns\\0\\b81d20537a9345cf9126d7c8bdc7a2c6\\artifacts\pre_skl_model")
input_example =  ml_model.load_input_example("mlruns\\0\\b81d20537a9345cf9126d7c8bdc7a2c6\\artifacts\pre_skl_model")
print(type(model))
pred = model.predict(input_example)


model = mlflow.onnx.load_model("mlruns\\0\\b81d20537a9345cf9126d7c8bdc7a2c6\\artifacts\onnx_model")
ml_model = mlflow.models.Model.load("mlruns\\0\\b81d20537a9345cf9126d7c8bdc7a2c6\\artifacts\onnx_model")
input_example =  ml_model.load_input_example("mlruns\\0\\b81d20537a9345cf9126d7c8bdc7a2c6\\artifacts\onnx_model")
print(type(input_example))
# print(type(model))
pred = model.predict(input_example)

