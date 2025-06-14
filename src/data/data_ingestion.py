import numpy as np
import pandas as pd

import os

import src.utils as utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# logging configure
logger = utils.configure_logger(__name__, log_file="data_ingestion.log")


def load_data(url: str) -> pd.DataFrame:
    try:
        logger.debug("Loading Data")
        return pd.read_csv(url)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def preliminary_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Preliminary Preprocessing")
        if "tweet_id" in df.columns:
            df.drop("tweet_id", axis=1, inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error in preliminary preprocessing: {e}")
        return pd.DataFrame()


def encode_labels(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.debug("Encoding Labels")

        if (
            "sentiment" not in train_data.columns
            or "sentiment" not in test_data.columns
        ):
            raise ValueError("Both train and test data must have 'sentiment' column")
        le = LabelEncoder()
        train_data["sentiment"] = le.fit_transform(train_data["sentiment"])
        test_data["sentiment"] = le.transform(test_data["sentiment"])
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error encoding labels: {e}")
        return pd.DataFrame(), pd.DataFrame()


if __name__ == "__main__":

    try:
        # load params
        params = utils.load_params(
            "params.yaml", section="data_ingestion", logger=logger
        )
        if not params:
            raise ValueError("Params not found")

        # load data
        url = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/refs/heads/main/tweet_emotions.csv"
        data = load_data(url)

        # preliminary preprocessing
        data = preliminary_preprocessing(data)
        if data.empty:
            raise ValueError("Data is empty")

        # split data
        train_data, test_data = train_test_split(
            data, test_size=params["test_size"], random_state=params["random_state"]
        )

        # encode labels
        train_data, test_data = encode_labels(train_data, test_data)

        # save data
        data_path = os.path.join("data", "raw")
        utils.save_data(train_data, test_data, data_path, logger=logger)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
