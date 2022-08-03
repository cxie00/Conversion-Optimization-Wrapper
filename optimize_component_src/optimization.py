import numpy as np
import onnx
import glob

import tvm
from tvm import te
import tvm.relay as relay

import mlflow
from onnx import numpy_helper
import onnxruntime

import argparse
import yaml 
from yaml.loader import SafeLoader

def read_load_model(model_folder):
    """
    returns the instance of the input MLflow model by reading its flavors.
    params:
        model_folder: string path to the MLmodel file of the MLflow model. Must be like "path/to/run/artifacts/flavor" 
    """
    mlmodel_file = model_folder + "/MLmodel"

    with open(mlmodel_file) as f:
        data = yaml.load(f, Loader=SafeLoader)
        for flavor in data["flavors"]:
            if (flavor == "pytorch"):
                return (mlflow.pytorch.load_model(model_folder), 'pytorch')
            elif (flavor == "onnx"):
                return (mlflow.onnx.load_model(model_folder), 'onnx')
            
        print("No Pytorch or ONNX model found.")
        return

def optimize_onnx(model, inputs, shape_dict):
    """
    Returns a TVM-optimized model of an ONNX model.
    params:
        model: instance of ONNX model
        inputs: inputs to the ONNX model. 
        shape_dict (dict of str to tuple): dictionary of string input_name of model to tuple of the dimensions of the inputs. 
            User may need to manually input shape if np.shape does not work.
    """
    dtype = inputs.dtype
    mod, params = relay.frontend.from_onnx(model, shape_dict, dtype=dtype)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=1):
        executor = relay.build_module.create_executor(
            "graph", mod, tvm.cpu(0), target, params
        ).evaluate()
        
    tvm_output = executor(tvm.nd.array(inputs.astype(dtype))).numpy()
    return tvm_output

def optimize(model_folder, shape):
    """
    Returns a TVM-optimized model of an ONNX (or, in the future, a Pytorch) model. 
    params: 
        model_folder: string path to the MLmodel file of the MLflow model. Must be like "path/to/run/artifacts/flavor".
        shape (tuple): shape dimensions of inputs to model.
    """
    model, flavor = read_load_model(model_folder)
    if (flavor == 'onnx'):
        # Get input name to ONNX model
        session = onnxruntime.InferenceSession(model_folder + "/model.onnx")
        session.get_modelmeta()
        input_name = session.get_inputs()[0].name
        
        # Load inputs of ONNX model
        ml_model = mlflow.models.Model.load(model_folder)
        inputs =  ml_model.load_input_example(model_folder)
        shape_dict = {input_name: shape}
        
        # Optimize the ONNX model
        output = optimize_onnx(model, inputs, shape_dict)
        return output
    elif (flavor == 'pytorch'):
        # TODO: write optimize_pytorch
        # output = optimize_pytorch(model, shape)
        # return output
        return
    
# Input Preprocessing
parser = argparse.ArgumentParser()
parser.add_argument("--optimization_component_model", type=str)
parser.add_argument("--optimization_shape_input", type=str)
parser.add_argument("--optimization_component_output", type=str)

args = parser.parse_args()
print("optimization_component_model path: %s" % args.optimization_component_model)
print("optimization_shape_input: %s" % args.optimization_shape_input)
print("optimization_component_output: %s" % args.optimization_component_output)

shape = tuple([int(i) for i in args.optimization_shape_input.split(",")])
print(shape)
# Optimize the model
tvm_model = optimize(args.optimization_component_model, shape)
