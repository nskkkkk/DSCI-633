import pandas as pd
import numpy as np
from collections import Counter
                                                # DID NOT USE HINT FILE

class my_DT:

    def __init__(self, max_depth=8, min_im_y_decrease=0, min_samples_split=2, criterion="gini"):
        # criterion = {"gini", "entropy"},
        # Stop training if depth = max_depth. Depth of a binary tree: the max number of edges from the root node to a leaf node
        # Only split node if impurity decrease >= min_impurity_decrease after the split
        #  Weighted impurity decrease: N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
        # Only split node with >= min_samples_split samples
        self.max_depth = int(max_depth)
        self.min_im_y_decrease = min_im_y_decrease
        self.min_samples_split = int(min_samples_split)
        self.tree = {}
        self.criterion = criterion

    def impurity(self, labels):
        # Calculate impurity (unweighted)
        # Input is a list (or np.array) of labels
        # Output impurity score
        
        label_counts = Counter(labels)
        N = float(len(labels))
        
        if self.criterion == "gini":
           
            gini_impurity = 0.0
            for label in labels:
                impurity_sum = 1
                for key in label_counts:
                    impurity_sum -= (label_counts[key] / N) ** 2
                gini_impurity = impurity_sum
            return gini_impurity

        elif self.criterion == "entropy":
            
            entropy_impurity = 0.0
            unique_labels = np.unique(labels)
            entropy_impurity = 0
            for cls in unique_labels:
                pcls = len(labels[labels == cls]) / len(labels)
                entropy_impurity += -pcls * np.log2(pcls)
            return entropy_impurity


    def find_node(self, i_s, i_o, cans, labels, cs, pop, n):
        for i in range(n - 1):
            i_s.append([] if cans[cs[i]] == cans[cs[i + 1]] else [self.impurity(labels[pop[cs[:i + 1]]]) * (i + 1),
                                                                  (n - i - 1) * self.impurity(labels[pop[cs[i + 1:]]])])
            i_o.append(np.inf if cans[cs[i]] == cans[cs[i + 1]] else np.sum(i_s[-1]))
        return [i_s, i_o]

    def find_best_split(self, pop, X, labels):

        sf = None
        for feature in X.keys():
            cans = np.array(X[feature][pop])
            cs = np.argsort(cans)
            n = len(cs)
            [i_s, i_o] = self.find_node([], [], cans, labels, cs, pop, n)
            min_i_o = np.min(i_o)

            if min_i_o < np.inf and (sf == None or sf[1] > min_i_o):
                split = np.argmin(i_o)
                sf = (feature, min_i_o, (cans[cs][split] + cans[cs][split + 1]) / 2.0,
                       [pop[cs[:split + 1]], pop[cs[split + 1:]]], i_s[split])

        return sf

    def fit_node(self, nodes, labels, n_nodes, p, im_y, n_level, X, N):
        for node in nodes:
            left_n = node * 2 + 1
            right_n = node * 2 + 2
            c_pop = p[node]
            c_i_o = im_y[node]

            if len(c_pop) < self.min_samples_split or c_i_o == 0 or n_level:
                self.tree[node] = Counter(labels[c_pop])
            else:
                s_f = self.find_best_split(c_pop, X, labels)
                if s_f and (c_i_o - s_f[1]) > self.min_im_y_decrease * N:
                    self.tree[node] = (s_f[0], s_f[2])
                    n_nodes.extend([left_n, right_n])
                    p[left_n] = s_f[3][0]
                    p[right_n] = s_f[3][1]
                    im_y[left_n] = s_f[4][0]
                    im_y[right_n] = s_f[4][1]
                else:
                    self.tree[node] = Counter(labels[c_pop])
        return n_nodes

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        self.c_s = list(set(list(y)))

        # write your code below


        labels = np.array(y)
        N = len(y)
        p = {0: np.array(range(N))}
        im_y = {0: self.impurity(labels[p[0]]) * N}
        level = 0
        nodes = [0]
        while level < self.max_depth and nodes:
            nodes = self.fit_node(nodes, labels, [], p, im_y, (level + 1 == self.max_depth), X, N)
            level += 1
        return

    def nodecounter(self, node, predicting, plist, X, i):

        while True:
            if type(self.tree[node]) == Counter:
                plist.append(list(self.tree[node].keys())[np.argmax(self.tree[node].values())]
                              if predicting else
                              {key: self.tree[node][key] / float(np.sum(list(self.tree[node].values()))) for
                               key in self.c_s})
                break
            else:
                node = node * 2 + (1 if X[self.tree[node][0]][i] < self.tree[node][1] else 2)
        return plist

    def predictnode(self, X, predicting):

        plist = []
        for i in range(len(X)):
            plist = self.nodecounter(0, predicting, plist, X, i)
        return plist

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below

        return self.predictnode(X, True)

    def predict_proba(self, X):

        # X: pd.DataFrame, independent variables, float
        # Eample:
        # self.classes_ = {"2", "1"}
        # the reached node for the test data point has {"1":2, "2":1}
        # then the prob for that data point is {"2": 1/3, "1": 2/3}
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below

        
        return pd.DataFrame(self.predictnode(X, False), columns=self.c_s)
        

