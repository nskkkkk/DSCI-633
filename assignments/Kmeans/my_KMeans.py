# my_KMeans.py

import pandas as pd
import numpy as np

class my_KMeans:

    def __init__(self, n_clusters=8, init="k-means++", n_init=10, max_iter=300, tol=1e-4):
        # init = {"k-means++", "random"}
        # use euclidean distance for inertia calculation.
        # stop when either # iteration is greater than max_iter or the delta of self.inertia_ is smaller than tol.
        # repeat n_init times and keep the best run (cluster_centers_, inertia_) with the lowest inertia_.

        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = n_init
        self.max_iter = int(max_iter)
        self.tol = tol

        self.classes_ = range(n_clusters)
        # Centroids
        self.cluster_centers_ = None
        # Sum of squared distances of samples to their closest cluster center.
        self.inertia_ = None

    def fit(self, X):
        # X: pd.DataFrame, independent variables, float
        # repeat self.n_init times and keep the best run
        # (self.cluster_centers_, self.inertia_) with the lowest self.inertia_.
        best_inertia = float('inf')
        best_centers = None  # Initialize best_centers outside the loop
        for _ in range(self.n_init):
            centroids = [X.sample().values[0]]

            for _ in range(1, self.n_clusters):
                distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X.values])
                probabilities = distances / distances.sum()
                next_centroid = X.sample(weights=probabilities).values[0]
                centroids.append(next_centroid)

            centroids = np.array(centroids)

            self.cluster_centers_ = centroids  # Initialize cluster centers

            for _ in range(self.max_iter):
                # Assign points to clusters
                distances = self._calculate_distances(X)
                labels = np.argmin(distances, axis=1)

                # Compute the new cluster centers based on assigned labels
                new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

                # Check for convergence
                if np.linalg.norm(new_centers - centroids) < self.tol:
                    break

                centroids = new_centers

            # Update inertia
            self.inertia_ = np.sum((X.values - centroids[labels]) ** 2)
            self.cluster_centers_ = centroids
            # self._fit_single(X)
            if self.inertia_ is not None and self.inertia_ < best_inertia:
                best_inertia = self.inertia_
                best_centers = np.copy(self.cluster_centers_)
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        distance = self._calculate_distances(X)

        predictions = np.argmin(distance,axis=1)
        return predictions

    def transform(self, X):
        # Transform to cluster-distance space
        # X: pd.DataFrame, independent variables, float
        # return dists = list of [dist to centroid 1, dist to centroid 2, ...]
        dists = self._calculate_distances(X)
        return dists

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _calculate_distances(self, X):
        # Internal method to calculate distances to centroids
        return np.linalg.norm(X.values[:, np.newaxis, :] - self.cluster_centers_, axis=2)
