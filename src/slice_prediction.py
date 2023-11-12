
import os
import yaml
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# data directory
PROCESS_DATA = 'data/clean_census.csv'
MODEL_ARTIFACT = 'model'

logging.info("Getting Data Schema...")
schema_file_path = 'config/schema.yaml'

with open(schema_file_path, 'r') as fm:
    data_config = yaml.safe_load(fm)

cat_features = data_config['categorical_columns']
target = data_config['target'][0]


def predict_on_slices():
    """ 
    Make prediction on each categories in the dataset 
    """
    logging.info("Loading the data...")
    df = pd.read_csv(PROCESS_DATA)
    _, test = train_test_split(df, test_size=0.20)

    logging.info("Loading model artifacts...")
    rf_model = joblib.load(os.path.join(MODEL_ARTIFACT, 'model.pkl'))

    logging.info("Loading encoder and label binarizer...")
    encoder = joblib.load(os.path.join(MODEL_ARTIFACT, 'encoder.pkl'))
    lb = joblib.load(os.path.join(MODEL_ARTIFACT, 'lb.pkl'))

    slice_metrics = []

    for feature in cat_features:
        # make slice prediction on the test data
        for cls in test[feature].unique():
            df_cls = test[test[feature] == cls]

            X_test, y_test, _, _ = process_data(
                df_cls,
                cat_features,
                label=target,
                encoder=encoder,
                lb=lb,
                training=False)

            logging.info(f"Making prediction on slice {feature} - {cls}...")
            y_pred = rf_model.predict(X_test)

            logging.info("Computing model metrics...")
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            row = f"{feature} - {cls} :: Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}"
            slice_metrics.append(row)

            os.makedirs('slice_performance', exist_ok=True)
            with open('slice_performance/slice_prediction.txt', 'w') as file:
                for row in slice_metrics:
                    file.write(row + '\n')

    logging.info(f"Slice performance saved...")


if __name__ == '__main__':
    predict_on_slices()
