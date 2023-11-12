
import requests
import json

# Render App URL Address
url = "https://census-prediction-deploye.onrender.com/make_prediction"

# explicit the sample to perform inference on
query_sample = {'age': 50,
                'workclass': "Private",
                'fnlgt': 234721,
                'education': "Doctorate",
                'education_num': 16,
                'marital_status': "Separated",
                'occupation': "Exec-managerial",
                'relationship': "Not-in-family",
                'race': "Black",
                'sex': "Female",
                'capital_gain': 0,
                'capital_loss': 0,
                'hours_per_week': 50,
                'native_country': "United-States"
                }

# define data
df = json.dumps(query_sample)

# post to API
query_response = requests.post(url, data=df)

# Putput the query response
print("Query_Response status", query_response.status_code)
print("Query_Response content:")
print(query_response.json())
