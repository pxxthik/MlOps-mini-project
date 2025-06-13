import dagshub
import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import os

mlflow.set_tracking_uri('https://dagshub.com/pxxthik/MlOps-mini-project.mlflow')
dagshub.init(repo_owner='pxxthik', repo_name='MlOps-mini-project', mlflow=True)

mlflow.set_experiment("BoW vs TFIDF")


df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv").drop(columns=["tweet_id"])

x = df["sentiment"].isin(["sadness", "happiness"])
df = df[x]

# data preprocessing

# Define text preprocessing functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

df = normalize_text(df)
df["sentiment"].replace({"sadness": 0, "happiness": 1}, inplace=True)

# Feature extraction methods
vectorizers = {
    "BoW": CountVectorizer(),
    "TFIDF": TfidfVectorizer()
}

# algorithms
algorithms = {
    "Logistic Regression": LogisticRegression(),
    "Multinomial NB": MultinomialNB(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# start parent run
with mlflow.start_run() as parent_run:
    # loop over feature extraction methods
    for vectorizer_name, vectorizer in vectorizers.items():
        # loop over algorithms
        for algorithm_name, algorithm in algorithms.items():
            with mlflow.start_run(run_name=f"{vectorizer_name} with {algorithm_name}", nested=True) as child_run:
                # split data into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(df["content"], df["sentiment"], test_size=0.2, random_state=42)

                # vectorize text data
                X_train = vectorizer.fit_transform(X_train)
                X_test = vectorizer.transform(X_test)

                # log parameters
                mlflow.log_param("vectorizer", vectorizer_name)
                mlflow.log_param("algorithm", algorithm_name)

                # train model
                algorithm.fit(X_train, y_train)

                # log model parameter
                if algorithm_name == "Logistic Regression":
                    mlflow.log_param("C", algorithm.C)
                elif algorithm_name == "Multinomial NB":
                    mlflow.log_param("alpha", algorithm.alpha)
                elif algorithm_name == "Random Forest":
                    mlflow.log_param("n_estimators", algorithm.n_estimators)
                    mlflow.log_param("max_depth", algorithm.max_depth)
                elif algorithm_name == "Gradient Boosting":
                    mlflow.log_param("n_estimators", algorithm.n_estimators)
                    mlflow.log_param("learning_rate", algorithm.learning_rate)

                # log model
                mlflow.sklearn.log_model(algorithm, "model")

                # evaluate model
                y_pred = algorithm.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)

                # log model
                mlflow.sklearn.log_model(algorithm, "model")

                # log file
                mlflow.log_artifact(__file__)

                # print results
                print(f"{vectorizer_name} with {algorithm_name}")
                print(f"Accuracy: {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1: {f1}")
                print()
                print("-" * 50)
                print()
