import pandas as pd
import numpy as np
from copy import deepcopy
from pdb import set_trace
import math
                                                 #NO HINT FILE USED
class my_AdaBoost:

    def __init__(self, base_estimator = None, n_estimators = 50, learning_rate=1):
        # Multi-class Adaboost algorithm (SAMME)
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator rounds
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]
        self.learning_rate = learning_rate

    def fit(self, X, y):
        # Fit the AdaBoost model to the training data.
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array, or pd.Series, dependent variables, int or str

        # Get the unique classes and the number of classes (k).
        self.classes_ = list(set(y))
        k = len(self.classes_)
        n = len(y)

        # Initialize sample weights uniformly.
        w = np.array([1.0 / n] * n)
        labels = np.array(y)
        self.alpha = []

        for i in range(self.n_estimators):
            # Sample with replacement from X, using the current sample weights.
            sample_indices = np.random.choice(n, n, p=w)

            # Train the base classifier with the sampled training data.
            sampled_data = X.iloc[sample_indices]
            sampled_data.index = range(len(sample_indices))
            self.estimators[i].fit(sampled_data, labels[sample_indices])

            # Predict using the current base estimator.
            predictions = self.estimators[i].predict(X)
            misclassified = np.array(predictions) != y

            # Compute the error rate for the current estimator.
            error = np.sum(misclassified * w)

            while error >= (1 - 1.0 / k):
                # Reset sample weights to uniform if error is too high.
                w = np.array([1.0 / n] * n)
                sample_indices = np.random.choice(n, n, p=w)
                sampled_data = X.iloc[sample_indices]
                sampled_data.index = range(len(sample_indices))
                self.estimators[i].fit(sampled_data, labels[sample_indices])
                predictions = self.estimators[i].predict(X)
                misclassified = np.array(predictions) != y
                error = np.sum(misclassified * w)

            # Calculate alpha value 
            if error <= 0.5:
                alpha = self.learning_rate * math.log((1.0 - error) / error) + np.log(k - 1)
            else:
                alpha = self.learning_rate * math.log((1.0 - error) / error)
            self.alpha.append(alpha)

            
            updated_sample_weights = [w * np.exp(alpha) if isError else w for w, isError in zip(w, misclassified)]
            w = np.array(updated_sample_weights)
            w /= np.sum(w)

        
        sum_alpha = np.sum(self.alpha)
        self.alpha = [a / sum_alpha for a in self.alpha]

        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
     

        
        list_predictions = []         # Initialize an empty list 

        for j in range(self.n_estimators):                
                                                            # Predict using the j-th base estimator and append to the list.
            predictions = self.estimators[j].predict(X)
            list_predictions.append(predictions)

        
        dflist_predictions = pd.DataFrame(list_predictions)    # Create a DataFrame to store the predictions

        
        probs = []             # Initialize a list to store the class probabilities.

        for col in range(dflist_predictions.shape[1]):
                                                                                
            pClass = {name: 0 for name in self.classes_}

            for row in range(dflist_predictions.shape[0]):
                                                                               
                classname = dflist_predictions.iloc[row, col]
                                                                                
                pClass[classname] += self.alpha[row]

                                                                                
            probs.append({name: pClass[name] for name in self.classes_})

                                                                                
        probs = pd.DataFrame(probs, columns=self.classes_)

        return probs



