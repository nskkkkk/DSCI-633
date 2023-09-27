import pandas as pd
import numpy as np
from collections import Counter

class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        # Calculate P(yj) and P(xi|yj)        
        # make sure to use self.alpha in the __init__() function as the smoothing factor when calculating P(xi|yj)
        # write your code below
        self.P_y = Counter(y)
        self.P = {}

        for labels in self.classes_:
            self.P[labels] = {}
            for features in X.columns:
                self.P[labels][features] = {}

                unique_feature_values = X[features].unique()
                number_of_values = len(unique_feature_values)

                for x in unique_feature_values:
                    count_of_xi = np.sum((X[features] == x) & (y == labels))
                    self.P[labels][features][x] = (count_of_xi + self.alpha) /(np.sum(y == labels) + number_of_values * self.alpha)

        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # write your code below
        probabilities = self.predict_proba(X)
        predictions = [row.idxmax() for _, row in probabilities.iterrows()]

        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)                
        # P(yj|x) = P(x|yj)P(yj)/P(x)
        # write your code below

        probabilities = {label: [] for label in self.classes_}

        for index, row in X.iterrows():
            for label in self.classes_:
                probability = np.log(self.P_y[label])
                for feature in X.columns:
                    value = row[feature]
                    if value in self.P[label][feature]:
                        probability += np.log(self.P[label][feature][value])
                probabilities[label].append(probability)

        probabilities_df = pd.DataFrame(probabilities, columns=self.classes_)
        probabilities_df = np.exp(probabilities_df)
        probabilities_df = probabilities_df.div(probabilities_df.sum(axis=1), axis=0)
        
        return probabilities_df
        

        


