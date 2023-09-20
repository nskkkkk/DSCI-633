import numpy as np
import pandas as pd
from collections import Counter

class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below

        self.confusion_matrix = {}
        for target_class in self.classes_:
            tp = np.sum((self.predictions == target_class) & (self.actuals == target_class))
            tn = np.sum((self.predictions != target_class) & (self.actuals != target_class))
            fp = np.sum((self.predictions == target_class) & (self.actuals != target_class))
            fn = np.sum((self.predictions != target_class) & (self.actuals == target_class))
            self.confusion_matrix[target_class] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
            self.confusion_matrix[target_class] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
           
         
    

    def precision(self, target=None, average="macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0
        if self.confusion_matrix is None:
            self.confusion()
        
        if target is not None:
            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]
            if (tp + fp) == 0:
                return 0.0
            return tp / (tp + fp)
        else:
            precisions = []
            for target_class in self.classes_:
                tp = self.confusion_matrix[target_class]["TP"]
                fp = self.confusion_matrix[target_class]["FP"]
                if (tp + fp) != 0:
                    precisions.append(tp / (tp + fp))
            if len(precisions) == 0:
                return 0.0
            if average == "macro":
                return np.mean(precisions)
            elif average == "micro":
                return np.sum([self.confusion_matrix[target]["TP"] for target in self.classes_]) / np.sum([self.confusion_matrix[target]["TP"] + self.confusion_matrix[target]["FP"] for target in self.classes_])
            elif average == "weighted":
                return np.average(precisions, weights=[self.confusion_matrix[target]["TP"] + self.confusion_matrix[target]["FP"] for target in self.classes_])
            else:
                raise ValueError("Invalid 'average' parameter value.")

    def recall(self, target=None, average="macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0
        # write your own code below

        if self.confusion_matrix is None:
            self.confusion()

        if target is not None:
            tp = self.confusion_matrix[target]["TP"]
            fn = self.confusion_matrix[target]["FN"]
            if (tp + fn) == 0:
                return 0.0
            return tp / (tp + fn)
        else:
            recalls = []
            for target_class in self.classes_:
                tp = self.confusion_matrix[target_class]["TP"]
                fn = self.confusion_matrix[target_class]["FN"]
                if (tp + fn) != 0:
                    recalls.append(tp / (tp + fn))
            if len(recalls) == 0:
                return 0.0
            if average == "macro":
                return np.mean(recalls)
            elif average == "micro":
                return np.sum([self.confusion_matrix[target]["TP"] for target in self.classes_]) / np.sum([self.confusion_matrix[target]["TP"] + self.confusion_matrix[target]["FN"] for target in self.classes_])
            elif average == "weighted":
                return np.average(recalls, weights=[self.confusion_matrix[target]["TP"] + self.confusion_matrix[target]["FN"] for target in self.classes_])
            else:
                raise ValueError("Invalid 'average' parameter value.")

    def f1(self, target=None, average="macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        # note: be careful for divided by 0
        #write your own code below
        if target is not None:
            precision_val = self.precision(target)
            recall_val = self.recall(target)
            if (precision_val + recall_val) == 0:
                return 0.0
            return 2 * (precision_val * recall_val) / (precision_val + recall_val)
        else:
            f1_scores = []
            for target_class in self.classes_:
                precision_val = self.precision(target_class)
                recall_val = self.recall(target_class)
                if (precision_val + recall_val) != 0:
                    f1_scores.append(2 * (precision_val * recall_val) / (precision_val + recall_val))
            if len(f1_scores) == 0:
                return 0.0
            if average == "macro":
                return np.mean(f1_scores)
            elif average == "micro":
                total_tp = np.sum([self.confusion_matrix[target]["TP"] for target in self.classes_])
                total_fp = np.sum([self.confusion_matrix[target]["FP"] for target in self.classes_])
                total_fn = np.sum([self.confusion_matrix[target]["FN"] for target in self.classes_])
                if (total_tp + total_fp + total_fn) == 0:
                    return 0.0
                return 2 * total_tp / (2 * total_tp + total_fp + total_fn)
            elif average == "weighted":
                weighted_f1_scores = []
                for target_class in self.classes_:
                    tp = self.confusion_matrix[target_class]["TP"]
                    fp = self.confusion_matrix[target_class]["FP"]
                    fn = self.confusion_matrix[target_class]["FN"]
                    if (tp + fp + fn) != 0:
                        weighted_f1_scores.append(2 * tp / (2 * tp + fp + fn))
                if len(weighted_f1_scores) == 0:
                    return 0.0
                return np.average(weighted_f1_scores, weights=[self.confusion_matrix[target]["TP"] + self.confusion_matrix[target]["FP"] + self.confusion_matrix[target]["FN"] for target in self.classes_])
            else:
                raise ValueError("Invalid 'average' parameter value.")
            return f1_score

    def auc(self, target):
        # compute AUC of ROC curve for the target class
        # return auc = float
        if type(self.pred_proba) == type(None):
            return None
        else:
            # Calculate AUC without importing sklearn
            pos_class = target
            probas = self.pred_proba[pos_class].values
            actual = (self.actuals == pos_class).astype(int)

            # Sort the probabilities and corresponding actual labels
            sorted_indices = np.argsort(probas)[::-1]
            probas_sorted = probas[sorted_indices]
            actual_sorted = actual[sorted_indices]

            n_positives = np.sum(actual_sorted)
            n_negatives = len(actual_sorted) - n_positives

            tp = 0
            fp = 0
            auc = 0

            # Calculate AUC using the trapezoidal rule
            for i in range(len(probas_sorted)):
                if actual_sorted[i] == 1:
                    tp += 1
                else:
                    fp += 1
                    auc += tp

            if n_positives == 0 or n_negatives == 0:
                return None

            auc /= (n_positives * n_negatives)

            return auc
