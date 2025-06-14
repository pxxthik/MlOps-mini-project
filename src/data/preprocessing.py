import numpy as np
import pandas as pd

import os

import src.utils as utils

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# logging configure
logger = utils.configure_logger(__name__, log_file="preprocessing.log")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.debug("Loading Data")
        data_path = os.path.join("data", "raw")
        train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
        test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()


# Preprocessing Class
class TextPreprocessor:
    def __init__(self, data: pd.DataFrame):
        try:
            self.data = data.copy()

            nltk.download("wordnet", quiet=True)
            nltk.download("stopwords", quiet=True)
            self.stop_words = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()

        except Exception as e:
            logger.error(f"Error initializing TextPreprocessor: {e}")

    def apply_preprocessing(self):
        try:
            self.lower_case()
            self.remove_stop_words()
            self.remove_numbers()
            self.remove_punctuations()
            self.remove_urls()
            self.lemmatization()
            self.remove_small_sentences()
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")

    def lower_case(self):
        try:
            logger.debug("Lowering Case")
            self.data["content"] = self.data["content"].str.lower()
        except Exception as e:
            logger.error(f"Error in lower_case: {e}")

    def remove_stop_words(self):
        try:
            logger.debug("Removing Stop Words")
            self.data["content"] = self.data["content"].apply(
                lambda x: " ".join(
                    word for word in str(x).split() if word not in self.stop_words
                )
            )
        except Exception as e:
            logger.error(f"Error in remove_stop_words: {e}")

    def remove_numbers(self):
        try:
            logger.debug("Removing Numbers")
            self.data["content"] = self.data["content"].str.replace(
                r"\d+", "", regex=True
            )
        except Exception as e:
            logger.error(f"Error in remove_numbers: {e}")

    def remove_punctuations(self):
        try:
            logger.debug("Removing Punctuations")
            self.data["content"] = self.data["content"].apply(
                lambda x: x.translate(str.maketrans("", "", string.punctuation))
            )
        except Exception as e:
            logger.error(f"Error in remove_punctuations: {e}")

    def remove_urls(self):
        try:
            logger.debug("Removing URLs")
            self.data["content"] = self.data["content"].str.replace(
                r"http\S+", "", regex=True
            )
        except Exception as e:
            logger.error(f"Error in remove_urls: {e}")

    def lemmatization(self):
        try:
            logger.debug("Lemmatizing")
            self.data["content"] = self.data["content"].apply(
                lambda x: " ".join(
                    self.lemmatizer.lemmatize(word) for word in str(x).split()
                )
            )
        except Exception as e:
            logger.error(f"Error in lemmatization: {e}")

    def remove_small_sentences(self):
        try:
            logger.debug("Removing Small Sentences")
            self.data.loc[self.data["content"].str.split().str.len() < 3, "content"] = (
                np.nan
            )
            self.data.dropna(inplace=True)
        except Exception as e:
            logger.error(f"Error in remove_small_sentences: {e}")

    def get_data(self):
        return self.data


if __name__ == "__main__":

    train_data, test_data = load_data()
    if train_data.empty or test_data.empty:
        raise ValueError("Data is empty")

    train_data_preprocessing = TextPreprocessor(data=train_data)
    test_data_preprocessing = TextPreprocessor(data=test_data)

    train_data_preprocessing.apply_preprocessing()
    test_data_preprocessing.apply_preprocessing()

    train_data = train_data_preprocessing.get_data()
    test_data = test_data_preprocessing.get_data()

    data_path = os.path.join("data", "interim")
    utils.save_data(train_data, test_data, data_path, logger=logger)
