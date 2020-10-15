import re
import csv
import joblib
import pickle
import warnings
import tqdm
import os
import pandas as pd
import numpy as np
import sys
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer


if __name__ == '__main__':
    # 1. Load data
    train_sentence, train_label = load_data('data/ratings_train.txt')
    test_sentence, test_label = load_data('data/ratings_test.txt')

    # 2. Generate tokens

    # 3. Generate 5 features

    # 4. Train classifiers with those features
    classifiers = [
        GaussianNB(),

    ]

    """
    Modeling
    """
    X_train, X_test, y_train, y_test = train_test_split(X_train_np, y_train_np)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    predictions_ = classifier.predict(X_test).tolist()
    print('Accuracy: %.10f' % accuracy_score(y_test, predictions_))

    estimator = SVC()
    n_estimators = 10
    n_jobs = -1

    model = BaggingClassifier(
        base_estimator=estimator,
        n_estimators=n_estimators,
        max_samples=(1 / n_estimators),
        max_features=1,
        n_jobs=n_jobs
    )

    model.fit(X_train[:10000], y_train[:10000])
    accuracy_score(model.predict(X_test), y_test)

    # 5. Evaluate and print out result
    for cf in classifiers:
        cf.fit(train_x, train_y)
        predictions = cf.predict(test_x)
        accuracy = accuracy_score(test_y, predictions)
        print('Classifier: {} | Accuracy: {}'.format(cf, accuracy))

    # 6. Generate output file with final test data (opens in 10:00 AM)
