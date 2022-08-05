import argparse
import os
from datetime import datetime

from hummingbird.ml import convert as hb_convert
import mlflow
import numpy as np
import torch

import yaml 
from yaml.loader import SafeLoader
import json
from sklearn.metrics import accuracy_score, precision_score, f1_score
import time

import os
print(os.environ)
def read_load_model(model_folder):
    """
    returns the instance of the input MLflow model by reading its flavors.
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

def load_input_data(model_folder):
    """
        Returns training and test data of input_data set.
        params:
            input_json (str): path to input_example.json in MLflow 
    """
    with open(model_folder + "/input_example.json") as d:
        dictData = json.load(d)

        X = dictData["inputs"]["train"]
        X = np.array(X)
        
        y = dictData["inputs"]["test"]
        y = np.array(y)
        
    return (X, y)

def evaluate(model, premodel, X, y, threshold, metrics_list):
    """
    Scores the new model on the input data and compares performance results.
    Fails self if perf is worse.
    Returns scored dataset
    """
    
    n = 105 # TODO: how many runs? make it a parameter?

    # pre-conversion: time predict() with RandomForestClassifier and get accuracy score
    pre_conv = []
    for i in range(0, n):
        start = time.perf_counter()
        pred = premodel.predict(X)
        end = time.perf_counter()
        if (n > 4): # omit first few runs, according to Klein
            pre_conv.append(end - start)
    pre_avg_ms = sum(pre_conv) / n * 1000

    # pre-converion: log metrics
    mlflow.log_metric('Pre-conversion-prediction-time (ms)', pre_avg_ms) 

    # post conversion: time predict() with new model and get accuracy score
    post_conv = []
    post_acc = []
    for i in range(0, n):
        start = time.perf_counter()
        pred_cpu_hb = model.predict(X)
        end = time.perf_counter()
        if (n > 4): # omit first few runs, according to Klein
            post_conv.append(end - start)
    post_avg_ms = sum(post_conv) / n * 1000

    # post converion: log metrics
    mlflow.log_metric('Post-conversion prediction time (ms)', post_avg_ms)

    if (post_avg_ms > (pre_avg_ms * threshold)):
        raise Exception("Performance of post-conversion model exceeded threshold. Failing job.")
    
    metric_evaluation(metrics_list, y, pred, pred_cpu_hb)

    return pred_cpu_hb

def metric_evaluation(metrics_list, y_true, y_pred, y_pred_cpu_hb):
    metrics_list = metrics_list.split(",")
    print(metrics_list)
        # go through list of metrics and evaluate
    for metric in metrics_list:
        if (metric == "accuracy"):
            accuracy = accuracy_score(y_true, y_pred)
            mlflow.log_metric('Pre-conversion-accuracy-score', accuracy)
            post_accuracy = accuracy_score(y_true, y_pred_cpu_hb)
            mlflow.log_metric('Post-conversion accuracy score', post_accuracy)
        elif (metric == "precision"):
            precision = precision_score(y_true, y_pred)
            mlflow.log_metric('Pre-conversion-precision-score', precision)
            post_precision = precision_score(y_true, y_pred_cpu_hb)
            mlflow.log_metric('Post-conversion-precision-score', post_precision)
        elif (metric == "f1"):
            f1 = f1_score(y_true, y_pred)
            mlflow.log_metric('Pre-conversion-f1-score', f1)
            post_f1 = f1_score(y_true, y_pred_cpu_hb)
            mlflow.log_metric('Post-conversion-f1-score', post-f1)
    return

def convert(model_folder, target, threshold, metrics_list):
    """
    Converts an MLflow model to a target framework and returns an MLflow model in the target framework.
    
    params:
        premodel: Model with Sklearn or Pytorch backend. 
        target: str of target backend to be converted to, such as ONNX and Pytorch.
            "torch" for Pytorch.
            "onnx" for ONNX. 
        input_data: The *onnx* backend requires either a test_input of a the initial types set through the exta_config parameter.

    returns:
        an MLflow model in the target framework.
    """
    premodel = read_load_model(model_folder)
    X, y = load_input_data(model_folder)
    assert premodel is not None
    assert (target == 'onnx' or target == 'torch')
   
    if (target == "onnx"):
        model = hb_convert(premodel, 'onnx', test_input=X)
        pred = model.predict(X)
        sig = mlflow.models.infer_signature(X, pred)

        scored_dataset = evaluate(model, premodel, X, y, threshold, metrics_list)
        mlflow.onnx.log_model(model.model, 'onnx_model', input_example={"train": X, "test": y}, signature=sig)
    elif (target == "torch" or target == "pytorch"):
        model = hb_convert(premodel, 'torch')
        model.to('cuda')
        pred = model.predict(X)
        sig = mlflow.models.infer_signature(X, pred)

        scored_dataset = evaluate(model, premodel, X, y, metrics_list)
        mlflow.pytorch.log_model(model.model, 'torch_model', input_example={"train": X, "test": y}, signature=sig)

    return (model, scored_dataset)


# input must be a link to the folder of the logged model for this to work
parser = argparse.ArgumentParser()
parser.add_argument("--conversion_component_input", type=str)
parser.add_argument("--conversion_target_input", type=str)
parser.add_argument("--conversion_threshold_input", type=int)
parser.add_argument("--conversion_metrics_input", type=str)
parser.add_argument("--conversion_component_output", type=str)

args = parser.parse_args()

print("conversion_component_input path: %s" % args.conversion_component_input)
print("conversion_target_input: %s" % args.conversion_target_input)
print("conversion_component_output path: %s" % args.conversion_component_output)


model = convert(args.conversion_component_input, args.conversion_target_input, args.conversion_threshold_input, args.conversion_metrics_input)



