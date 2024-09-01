import os
import json
import logging
import pathlib
import pickle
import tarfile
import argparse

import joblib
import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def _parse_args():

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_path', type=str, default= '')
    parser.add_argument('--model_file', type=str, default= '0_model.tar.gz')
    parser.add_argument('--data_path', type=str, default= 'splits')
    parser.add_argument('--output_path', type=str, default= '')
    return parser.parse_known_args()
    
if __name__ == "__main__":

    # parse arguments
    args, _ = _parse_args()
    model_path = args.model_path
    model_file = args.model_file
    data_path = args.data_path
    output_path = args.output_path

    # extract gz file
    model_full_path_name = os.path.join(model_path, model_file)

    if not os.path.exists('temp'):
        os.makedirs('temp', exist_ok=True)
        print('creating temp folder')
    else:
        print('temp folder exists')

    print('loading model:' + model_full_path_name)
    with tarfile.open(model_full_path_name) as tar:
        tar.extractall(path="temp")
        
    model = xgboost.Booster()
    model.load_model("temp/xgboost-model")

    test_x_path = os.path.join(data_path, 'test_x.csv')
    test_x_df = pd.read_csv(test_x_path, header=None)
    x_test = xgboost.DMatrix(test_x_df.values)

    test_y_path = os.path.join(data_path, 'test_y.csv')
    test_y_df = pd.read_csv(test_y_path, header=None)
    y_test = test_y_df.iloc[:, 0].to_numpy()
    
    X_test = xgboost.DMatrix(test_x_df.values)
    predictions = model.predict(x_test)

    acc = accuracy_score(y_test, predictions.round())
    auc = roc_auc_score(y_test, predictions.round())
    
    # The metrics reported can change based on the model used, 
    # but it must be a specific name per 
    # (https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {
                "value": acc,
                "standard_deviation": "NaN",
            },
            "auc": {"value": auc, "standard_deviation": "NaN"},
        },
    }

    print(report_dict)

    #evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    evaluation_output_path = os.path.join(output_path, "0_evaluation.json")
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))