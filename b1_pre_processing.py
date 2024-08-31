import pandas
import numpy
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _parse_args():

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--input_path', type=str, default= '')
    parser.add_argument('--input_file', type=str, default= '0_bank_additional_full.csv')
    parser.add_argument('--output_path', type=str, default= 'splits')

    return parser.parse_known_args()

if __name__== "__main__":

    # parse arguments
    args, _ = _parse_args()
    dataset_full_path_name = os.path.join(args.input_path, args.input_file)
    logger.info(dataset_full_path_name)
    
    outputpath = args.output_path
    logger.info(outputpath)

    # Read data
    df = pandas.read_csv(dataset_full_path_name, sep=';')
    
    # Replace values
    df = df.replace(regex=r'\.', value='_')
    df = df.replace(regex=r'\_$', value='')
    
    # Add two new features
    df["no_previous_contact"] = (df["pdays"] == 999).astype(int)
    df["not_working"] = df["job"].isin(["student", "retired", "unemployed"]).astype(int)
    
    # Drop not need columns
    df = df.drop(['job', 'marital','education','default','housing','loan','contact','month','day_of_week','pdays','poutcome','emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], axis=1)
        
    # Encode categorical features
    df = pandas.get_dummies(df)
    
    # Train, test, validation split
    suffled_df = df.sample(frac=1, random_state=42)
    train_data_split, validation_data_split, test_data_split = numpy.split(suffled_df, [int(0.7 * len(df)), int(0.9 * len(df))])
    
    # clean up categorical encoding of Y
    train_data_df = pandas.concat([train_data_split['y_yes'], train_data_split.drop(['y_yes','y_no'], axis=1)], axis=1)
    validation_data_df = pandas.concat([validation_data_split['y_yes'], validation_data_split.drop(['y_yes','y_no'], axis=1)], axis=1)

    # create local folder if it does not exist
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    
    # Write data
    train_data_df.to_csv(os.path.join(outputpath, 'train.csv'), index=False, header=False)
    validation_data_df.to_csv(os.path.join(outputpath, 'validate.csv'), index=False, header=False)
    
    test_data_split['y_yes'].to_csv(os.path.join(outputpath, 'test_y.csv'), index=False, header=False)
    test_data_split.drop(['y_yes','y_no'], axis=1).to_csv(os.path.join(outputpath, 'test_x.csv'), index=False, header=False)

    print('pre-processing complete')