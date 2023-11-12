"""
This module tests unit for the ML model
"""

from pandas.core.frame import DataFrame
import yaml
import pandas as pd
import pytest
import joblib
from ml.data import process_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fastapi.testclient import TestClient

from main import app

DATA_DIR = 'data/clean_census.csv'
MODEL_DIR = 'model/model.pkl'

# get schema file path
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


def test_load_data(data: DataFrame):
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


def test_process_data(data: DataFrame):
    """ 
    Test the data split 
    """

    train, _ = train_test_split(data, test_size=0.20, random_state=42)
    X, y, _, _ = process_data(train, cat_features, label='salary')
    assert len(X) == len(y)


# tests for FAST API
CLIENT = TestClient(app)


def test_greet():
    """ 
    Test the root page get a succesful response
    """
    response = CLIENT.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Use this app to predict weather income is more than $50K per year."}


def test_post_predict_up():
    """ 
    Test an example when income is less than 50K 
    """
    response = CLIENT.post("/make_prediction", json={
        "age": 50,
        "workclass": "Private",
        "fnlgt": 287372,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    })

    assert response.status_code == 200
    assert response.json() == {"Income prediction": " >50K"}


def test_post_predict_down():
    """ 
    Test an example when income is higher than 50K 
    """
    response = CLIENT.post("/make_prediction", json={
        "age": 28,
        "workclass": "Private",
        "fnlgt": 183175,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })

    assert response.status_code == 200
    assert response.json() == {"Income prediction": " <=50K"}
