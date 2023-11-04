# Script to train machine learning model.

# Add the necessary imports for the starter code.
import logging
from sklearn.model_selection import train_test_split
from src.main_utils import process_data, get_classification_metrics
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Add code to load in the data.
logging.info("Loading the data...")
data = pd.read_csv("data/census.csv")

logging.info("Getting Data Schema...")
schema_file_path = 'config/schema.yaml'

with open(schema_file_path) as f:
    data_config = yaml.load(f)

cat_features = data_config['categorical_columns']
target = data_config['target']  

logging.info("Create Traina and Test dataset split...")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.2)
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=target, training=True
)

logging.info("Preprocessing the test data...")  
# Proces the test data with the process_data function.
X_test = encoder.transform(test.drop(columns=target, axis=1))
y_test = lb.transform(test[target])

# Train and save a model.
logging.info("Training the model...")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

logging.info("Evaluating the model...")
F1_score, precision, recall = get_classification_metrics(rf, X_test, y_test)    
logging.info(f"F1 score: {F1_score}, Precision: {precision}, Recall: {recall}")

# Save artifacts
logging.info("Saving the model artifacts & preprocessor")
joblib.dump(rf, 'model/model.pkl')
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(lb, 'model/lb.pkl')