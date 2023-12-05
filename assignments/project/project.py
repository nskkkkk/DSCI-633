import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


def clean(X):
    X['description'] = X['description'].str.lower()
    X['description'] = X['description'].str.replace('[^a-zA-Z]', ' ')
    X['description'] = X['description'].str.replace('!', '')
    return X


class my_model:
    def __init__(self):
        self.preprocessor = CountVectorizer(stop_words='english', max_df=.7)
        self.clf=RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X, y):
        # do not exceed 29 mins
        X=clean(X)
        X=X['description']
        XX = self.preprocessor.fit_transform(X)
        X_final = TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False, sublinear_tf=True).fit_transform(XX)
        self.clf.fit(X_final, y)
        return

    def predict(self, X):
        X=clean(X)
        X=X['description']
        XX = self.preprocessor.transform(X)
        X_final = TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False, sublinear_tf=True).fit_transform(XX)
        predictionsOfModel = self.clf.predict(X_final)
        return predictionsOfModel


