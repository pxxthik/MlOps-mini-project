import numpy as np
import pandas as pd

import mlflow
import dagshub

import os
import pickle
import json

import src.utils as utils

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# logging configure
logger = utils.configure_logger(__name__, log_file="predict_model.log")

# load model
def load_model():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# load data
def load_test_data() -> pd.DataFrame:
    try:
        logger.debug("Loading testing Data")
        data_path = os.path.join("data", "features")
        test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
        return test_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


if __name__ == "__main__":

    mlflow.set_tracking_uri('https://dagshub.com/pxxthik/MlOps-mini-project.mlflow')
    dagshub.init(repo_owner='pxxthik', repo_name='MlOps-mini-project', mlflow=True)

    mlflow.set_experiment("DVC Pipeline")
    with mlflow.start_run() as run:

        # load params
        params = utils.load_params("params.yaml", section="all", logger=logger)

        # log params
        for _, value in params.items():
            mlflow.log_params(value)

        # load model
        model = load_model()
        # log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model info
        save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

        # load data
        test_data = load_test_data()

        if test_data.empty:
            raise ValueError("Data is empty")
        
        # Split X and y
        X_test = test_data.drop("sentiment", axis=1)
        y_test = test_data["sentiment"]

        # predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=1)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=1)
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")

        # log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)

        # print results
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"ROC AUC: {roc_auc}")

        # log file
        mlflow.log_artifact(__file__)
        # Log the model info file to MLflow
        mlflow.log_artifact('reports/experiment_info.json')

        # Log the evaluation errors log file to MLflow
        mlflow.log_artifact('predict_model.log')
