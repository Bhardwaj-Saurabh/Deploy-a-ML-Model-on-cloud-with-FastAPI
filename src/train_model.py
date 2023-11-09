# Script to train machine learning model.
import logging
import pandas as pd
import yaml
import joblib
from ml.data import process_data
from sklearn.model_selection import train_test_split
from ml.model import train_model, compute_model_metrics, inference
from ml.basic_cleaning import data_cleaning

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# data directory
data_dir = 'data/census.csv'
output_file = 'data/clean_census.csv'

# Add code to load in the data.
logging.info("Loading the data...")
data = pd.read_csv(data_dir)

logging.info("Getting Data Schema...")
schema_file_path = 'config/schema.yaml'

with open(schema_file_path, 'r') as fm:
    data_config = yaml.safe_load(fm)

cat_features = data_config['categorical_columns']
target = data_config['target'][0]

clean_data = data_cleaning(data, output_file)

logging.info("Create Train and Test dataset split...")
train, test = train_test_split(clean_data, test_size=0.2, random_state=42)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features,
    label=target, training=True
)

logging.info("Preprocessing the test data...")
# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features,
    label=target, training=False, encoder=encoder, lb=lb
)

# Add the necessary imports for the starter code.
# Train model
logging.info("Training RF model...")
model = train_model(X_train, y_train)

# Get the score
logging.info("Getting Scores on test dataset...")
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)


logging.info(f"Precision: {precision: .2f} \
             Recall: {recall: .2f} \
             Fbeta: {fbeta: .2f}")

# Save artifacts
logging.info("Saving model artifacts...")
joblib.dump(model, 'model/model.pkl')
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(lb, 'model/lb.pkl')
