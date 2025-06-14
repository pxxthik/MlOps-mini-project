import numpy as np
import pandas as pd

import os
import ast

import src.utils as utils

from sklearn.feature_extraction.text import CountVectorizer

# logging configure
logger = utils.configure_logger(name=__name__, log_file="feature_engineering.log")

# loading params
params = utils.load_params("params.yaml", section="feature_engineering", logger=logger)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.debug("Loading Data")
        data_path = os.path.join("data", "interim")
        train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
        test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()


def build_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.debug("Building Features")
        vectorizer = CountVectorizer(
            max_features=params["max_features"],
            ngram_range=ast.literal_eval(params["ngram_range"])
        )
        train_features = vectorizer.fit_transform(X_train["content"])
        test_features = vectorizer.transform(X_test["content"])
        return pd.DataFrame(train_features.toarray()), pd.DataFrame(
            test_features.toarray()
        )
    except Exception as e:
        logger.error(f"Error building features: {e}")
        return pd.DataFrame(), pd.DataFrame()


if __name__ == "__main__":

    # load data
    train_data, test_data = load_data()

    if train_data.empty or test_data.empty:
        raise ValueError("Data is empty")

    # Extracting X and y
    X_train = train_data.drop("sentiment", axis=1)
    y_train = train_data["sentiment"]

    X_test = test_data.drop("sentiment", axis=1)
    y_test = test_data["sentiment"]

    # build features
    train_bow, test_bow = build_features(X_train, X_test)
    logger.info("Features built successfully")

    # add labels
    train_bow["sentiment"] = y_train
    test_bow["sentiment"] = y_test

    # save data
    data_path = os.path.join("data", "features")
    utils.save_data(train_bow, test_bow, data_path, logger=logger)
