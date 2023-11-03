# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from src.main_utils import process_data
import pandas as pd
import yaml

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("data/census.csv")

schema_file_path = 'config/schema.yaml'

with open(schema_file_path) as f:
    data_config = yaml.load(f)

cat_features = data_config['categorical_columns']
target = data_config['target']  

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.2)
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=target, training=True
)

# Proces the test data with the process_data function.
X_test = encoder.transform(test.drop(columns=target, axis=1))
y_train = lb.transform(test[target])

# Train and save a model.
