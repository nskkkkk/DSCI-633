import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1 - cosine(x, y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        self.X_train = X
        self.y_train = y
        self.classes_ = list(set(y))

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list

       
        predictions = [] #creating an empty list to store predictations

        for i in range(len(X)):
           
            length = []      #empty list to store distances and class labels

            for j in range(len(self.X_train)):
                if self.metric == "minkowski":       #calculating the Minkowski length
                
                    dis = np.linalg.norm(X.iloc[i] - self.X_train.iloc[j], ord=self.p)
                elif self.metric == "euclidean":     # calculating the Euclidean length
                   
                    dis = np.linalg.norm(X.iloc[i] - self.X_train.iloc[j])
                elif self.metric == "manhattan":      #calculating the Manhattan length
                   
                    dis = np.sum(np.abs(X.iloc[i] - self.X_train.iloc[j]))
                elif self.metric == "cosine":         # calculating cosine length
                   
                    product = np.dot(X.iloc[i], self.X_train.iloc[j])
                    norm_X = np.linalg.norm(X.iloc[i])
                    norm_train = np.linalg.norm(self.X_train.iloc[j])
                    dis = 1 - (product / (norm_X * norm_train))

                length.append((dis, self.y_train.iloc[j]))


            knearest = sorted(length)[:self.n_neighbors]

           
            nearestclasses = [neighbor[1] for neighbor in knearest]
            classcount = Counter(nearestclasses)

            predictedclass = classcount.most_common(1)[0][0]
            predictions.append(predictedclass)
 
        return predictions  #return the list of predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)

       
        probs = [] #creating an empty list 

       
        for i in range(len(X)):
            
            distances = []

            for j in range(len(self.X_train)):
                if self.metric == "minkowski":
                    dist = np.linalg.norm(X.iloc[i] - self.X_train.iloc[j], ord=self.p)
                elif self.metric == "euclidean":
                    dist = np.linalg.norm(X.iloc[i] - self.X_train.iloc[j])
                elif self.metric == "manhattan":
                    dist = np.sum(np.abs(X.iloc[i] - self.X_train.iloc[j]))
                elif self.metric == "cosine":
                    dot_product = np.dot(X.iloc[i], self.X_train.iloc[j])
                    norm_X = np.linalg.norm(X.iloc[i])
                    norm_train = np.linalg.norm(self.X_train.iloc[j])
                    dist = 1 - (dot_product / (norm_X * norm_train))

                distances.append((dist, self.y_train.iloc[j]))

           
            knearest = sorted(distances)[:self.n_neighbors]

            nc = [neighbor[1] for neighbor in knearest]
            count = Counter(nc)

            probabilities = {}
            for label in self.classes_:
                probabilities[label] = count[label] / self.n_neighbors

            probs.append(probabilities)

      
        probs_df = pd.DataFrame(probs, columns=self.classes_)    # Converting  list of dictionaries into a pandas DataFrame

        return probs_df  #returning dataframe containing probabilities
