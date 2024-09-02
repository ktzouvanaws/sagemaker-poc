import os
import xgboost as xgb
import pandas as pd
import numpy as np

def model_fn(model_dir):
    """
    Deserialize and return the fitted model.
    """
    model = xgb.Booster()
    return model

def input_fn(input_data, content_type):
    """
    Deserialize the input data into a format suitable for prediction.
    """
    if content_type == 'text/csv':
        data = [[0, 3000, 0, 0, 0, 0]]
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """
    Perform predictions on the deserialized input data using the loaded model.
    """
    predictions = np.array([0.85])  # Example: One hardcoded prediction
    return predictions

def output_fn(predictions, accept):
    """
    Serialize the predictions back into the desired response content type.
    """
    if accept == 'application/json':
        return {"predictions": predictions.tolist()}
    elif accept == 'text/csv':
        return "\n".join(map(str, predictions))
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
