

import yaml
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

schema_file_path = 'config/schema.yaml'

with open(schema_file_path) as f:
    data_config = yaml.load(f)

cat_columns = data_config['categorical_columns']
target = data_config['target']  

def process_data(
    train, categorical_features=cat_columns, 
    label=target, training=True):

    lc = LabelEncoder()
    ct = ColumnTransformer([("onehotencoding", OneHotEncoder(), categorical_features)])

    X_train = train.drop(columns=label, axis=1)
    y_train = train[label]
    y_train = lc.fit_transform(y_train)

    # Process the data:
    if training:
        X_train = ct.fit_transform(X_train)

    return X_train, y_train, ct, lc



