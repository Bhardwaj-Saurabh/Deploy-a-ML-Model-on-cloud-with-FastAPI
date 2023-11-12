from sklearn.metrics import fbeta_score, precision_score, recall_score
import logging
from sklearn.ensemble import RandomForestClassifier


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Train and save a model.
    logging.info("Training the model...")
    rf = RandomForestClassifier(n_estimators=150,
                                min_samples_leaf=3)
    rf.fit(X_train, y_train)

    return rf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    logging.info("Evaluating the model...")
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    logging.info(
        f"F1 score: {fbeta}, Precision: {precision}, Recall: {recall}")
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : saved model format
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
