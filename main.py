# Put the code for your API here.

from fastapi import FastAPI
from typing import Literal
from pandas import DataFrame
import numpy as np
import joblib
import uvicorn
import yaml
from pydantic import BaseModel
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
    age: int
    workclass: Literal['State-gov',
                       'Self-emp-not-inc',
                       'Private',
                       'Federal-gov',
                       'Local-gov',
                       'Self-emp-inc',
                       'Without-pay']
    fnlgt: int
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    education_num: int
    marital_status: Literal["Never-married",
                            "Married-civ-spouse",
                            "Divorced",
                            "Married-spouse-absent",
                            "Separated",
                            "Married-AF-spouse",
                            "Widowed"]
    occupation: Literal["Tech-support",
                        "Craft-repair",
                        "Other-service",
                        "Sales",
                        "Exec-managerial",
                        "Prof-specialty",
                        "Handlers-cleaners",
                        "Machine-op-inspct",
                        "Adm-clerical",
                        "Farming-fishing",
                        "Transport-moving",
                        "Priv-house-serv",
                        "Protective-serv",
                        "Armed-Forces"]
    relationship: Literal["Wife", "Own-child", "Husband",
                          "Not-in-family", "Other-relative", "Unmarried"]
    race: Literal["White", "Asian-Pac-Islander",
                  "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Female", "Male"]
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']

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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
