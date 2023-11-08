"""
This module tests unit for the ML model
"""

import yaml
import pandas as pd
import pytest
import joblib
from ml.data import process_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = 'data/clean_census.csv'
MODEL_DIR = 'model/model.pkl'

schema_file_path = 'config/schema.yaml'

with open(schema_file_path, 'r') as fm:
    data_config = yaml.safe_load(fm)

cat_features = data_config['categorical_columns']
target = data_config['target'][0]


@pytest.fixture(name='data')
def data():
    """
    Fixture for creating dataset for tests
    """

    return pd.read_csv(DATA_DIR)


def test_load_data(data):
    """ 
    Check data is not empty
    """

    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_model():
    """ 
    Check if the train model exists
    """

    model = joblib.load(MODEL_DIR)
    assert isinstance(model, RandomForestClassifier)


def test_process_data(data):
    """ 
    Test the data split 
    """

    train, test = train_test_split(data, test_size=0.20)
    X, y, _, _ = process_data(train, cat_features, label='salary')
    assert len(X) == len(y)
