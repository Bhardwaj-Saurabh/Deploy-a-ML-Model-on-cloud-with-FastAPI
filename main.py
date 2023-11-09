# Put the code for your API here.

from fastapi import FastAPI
from pandas import DataFrame
import numpy as np
import joblib
import uvicorn
import yaml
from pydantic import BaseModel, Field
from src.ml.model import inference
from src.ml.data import process_data


# initiate fast API
app = FastAPI()

# load model artifacts
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")

# get data schema
schema_file_path = 'config/schema.yaml'
with open(schema_file_path, 'r') as fm:
    data_config = yaml.safe_load(fm)

cat_features = data_config['categorical_columns']
target = data_config['target'][0]

# define model input class


class ModelInputData(BaseModel):
    age: int = Field(example="39")
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example="77516")
    education: str = Field(example="Bachelors")
    education_num: int = Field(alias="education-num", example="13")
    marital_status: str = Field(
        alias="marital-status", example="Never-married")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(alias="capital-gain", example="2174")
    capital_loss: int = Field(alias="capital-loss", example="0")
    hours_per_week: int = Field(alias="hours-per-week", example="40")
    native_country: str = Field(
        alias="native-country", example="United-States")

# define root path


@app.get("/")
async def root():
    return {"message": "Use this app to predict weather income is more than $50K per year."}


# define inference path
@app.post("/make_prediction")
async def prediction(data: ModelInputData):
    input_data = np.array([
        [data.age,
         data.workclass,
         data.fnlgt,
         data.education,
         data.education_num,
         data.marital_status,
         data.occupation,
         data.relationship,
         data.race,
         data.sex,
         data.capital_gain,
         data.capital_loss,
         data.hours_per_week,
         data.native_country]
    ])

    df = DataFrame(data=input_data,
                   columns=[
                       "age",
                       "workclass",
                       "fnlwgt",
                       "education",
                       "education_num",
                       "marital-status",
                       "occupation",
                       "relationship",
                       "race",
                       "sex",
                       "capital_gain",
                       "capital_loss",
                       "hours-per-week",
                       "native-country"])

    X, _, _, _ = process_data(
        df, categorical_features=cat_features, encoder=encoder,
        lb=lb, training=False)

    y = inference(model, X)
    pred = lb.inverse_transform(y)[0]

    return {"Income prediction": pred}
