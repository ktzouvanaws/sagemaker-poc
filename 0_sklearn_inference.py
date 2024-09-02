import os
import xgboost as xgb
import pandas as pd
import numpy as np
import logging
from sagemaker_containers.beta.framework import worker

def model_fn(model_dir):
    """
    Deserialize and return the fitted model.
    """

def input_fn(input_data, content_type):
    """
    Deserialize the input data into a format suitable for prediction.
    """
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))

def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    features = model.transform(input_data)

    if label_column in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data[label_column], axis=1)
    else:
        # Return only the set of features
        return features

def output_fn(predictions, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == "text/csv":
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))
