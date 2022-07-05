import argparse
import os
from datetime import datetime

from hummingbird.ml import convert as hb_convert
import mlflow
from sklearn.preprocessing import Normalizer
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

import yaml 
from yaml.loader import SafeLoader


def read_load_model(model_folder):
    """
    returns the instance of the input MLflow model by reading its flavors.
    params:
        model_folder: string path to the MLmodel file of the MLflow model. Must be like "path\to\run\artifacts\flavor\MLmodel" 
    """
    mlmodel_file = model_folder + "/MLmodel"
    print(model_folder)
    # mlmodel_file = model_folder
    with open(mlmodel_file) as f:
        data = yaml.load(f, Loader=SafeLoader)
        print(data)
        for flavor in data["flavors"]:
            print(data["flavors"])
            if (flavor == "sklearn"):
                print("found sklearn")
                # return mlflow.sklearn.load_model(model_folder + "/artifacts/skl") # lob off the \MLmodel
                return mlflow.sklearn.load_model(model_folder)
            elif (flavor == "pytorch"):
                # return mlflow.pytorch.load_model(model_folder + "/artifacts/torch")
                # lob off the \MLmodel
                return mlflow.sklearn.load_model(model_folder)
            elif (flavor == "onnx"):
                return mlflow.onnx.load_model(model_folder)
            
        print("No skl, pytorch, or onnx model found.")
        return

def convert(premodel, target, test_input = 0):
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
        model = hb_convert(premodel, 'onnx', test_input)
        mlflow.onnx.log_model(model.model, 'onnx_model')
    elif (target == "torch" or target == "pytorch"):
        model = hb_convert(premodel, 'torch')
        model.to('cuda')
        mlflow.pytorch.log_model(model.model, 'torch_model')

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

# print("files in input path: ")
# arr = os.listdir(args.conversion_component_input)
# print(arr)
# print()
premodel = read_load_model(args.conversion_component_input)
model = convert(premodel, args.conversion_target_input)

# skl onnx test
# np.random.seed(0)
# data = np.random.rand(100, 200) * 1000
# data = np.array(data, dtype=np.float32)
# data_tensor = torch.from_numpy(data)
# norm = "l1"
# model = Normalizer(norm=norm)
# model.fit(data)
# hb_model = convert(model, 'onnx', data)

# skl pytorch test 
# X, y = load_breast_cancer(return_X_y=True)
# skl_model = RandomForestClassifier(n_estimators=500, max_depth=7)
# skl_model.fit(X, y)

# model = convert(skl_model, 'torch')

# lgbm onnx test
# import numpy as np
# import lightgbm as lgb

# import time 
# # Create some random data for binary classification.
# num_classes = 2
# X = np.array(np.random.rand(10000, 28), dtype=np.float32)
# y = np.random.randint(num_classes, size=10000)

# # Create and train a model (LightGBM in this case).
# model = lgb.LGBMClassifier()
# model.fit(X, y)
# onnx_model = convert(model, "onnx", X)
