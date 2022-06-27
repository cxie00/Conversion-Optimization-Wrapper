from msilib.schema import Error
from hummingbird.ml import convert as hb_convert
import mlflow


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
    print("Gamer time line 32")
    if (target == "onnx"):
        model = hb_convert(premodel, 'onnx', test_input)
        mlflow.onnx.log_model(model, 'onnx_model')
    elif (target == "torch"):
        model = hb_convert(premodel, 'torch')
        model.to('cuda')
        mlflow.pytorch.log_model(model.model, 'torch_model')

    return model





