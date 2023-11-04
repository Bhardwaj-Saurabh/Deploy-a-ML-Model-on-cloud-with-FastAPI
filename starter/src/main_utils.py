

import yaml
import logging  
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

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

def get_classification_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred)
    logging.info(class_report)  
    F1_score = f1_score(y_test, y_pred) 
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return F1_score, precision, recall



