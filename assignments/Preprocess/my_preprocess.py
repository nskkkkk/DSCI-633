import numpy as np
from scipy.linalg import svd
from copy import deepcopy
from collections import Counter
from pdb import set_trace

class my_normalizer: 
    def __init__(self, norm="Min-Max", axis = 1):
        #     norm = {"L1", "L2", "Min-Max", "Standard_Score"}
        #     axis = 0: normalize rows
        #     axis = 1: normalize columns
        self.norm = norm
        self.axis = axis
        

    def fit(self, X):
        #     X: input matrix
        #     Calculate offsets and scalers which are used in transform()
        X_array  = np.asarray(X)
        m, n = X_array.shape
        self.offsets = []    
        self.scalers = []   
        if self.axis == 1:
            for col in range(n):
                offset, scaler = self.vector_norm(X_array[:, col])
                self.offsets.append(offset)
                self.scalers.append(scaler)
        elif self.axis == 0:
            for row in range(m):
                offset, scaler = self.vector_norm(X_array[row])
                self.offsets.append(offset)
                self.scalers.append(scaler)
        else:
            raise Exception("Unknown axis.")

    def transform(self, X):
        X_norm = deepcopy(np.asarray(X))
        m, n = X_norm.shape
        if self.axis == 1:
            for col in range(n):
                X_norm[:, col] = (X_norm[:, col]-self.offsets[col])/self.scalers[col]
        elif self.axis == 0:
            for row in range(m):
                X_norm[row] = (X_norm[row]-self.offsets[row])/self.scalers[row]
        else:
            raise Exception("Unknown axis.")
        return X_norm

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def vector_norm(self, x):
        # Calculate the offset and scaler for input vector x
        if self.norm == "Min-Max":
            # Write your own code below
            min_val=np.main(x)
            max_val=np.max(x)
            offset = min_val
            scaler=max_val-min_val
            
        elif self.norm == "L1":                                       
            # Write your own code below
            offset=0
            scaler=np.sum(np.abs(x))
            
        elif self.norm == "L2":
            # Write your own code below
            offset=0
            scaler=np.sqrt(np.sum(x**2))
                
        elif self.norm == "Standard_Score":
            # Write your own code below
            mean_val=np.mean(x)
            std_dev=np.std(x)
            offset =mean_val

        else:
            raise Exception("Unknown normlization.")
        return offset, scaler

class my_pca:
    def __init__(self, n_components = 5):
        #     n_components: number of principal components to keep
        self.n_components = n_components

    def fit(self, X):
        #  Use svd to perform PCA on X
        #  Inputs:
        #     X: input matrix
        #  Calculates:
        #     self.principal_components: the top n_components principal_components
        U, s, Vh = svd(X)
        # Write your own code below
        self.principal_components = Vh[:self.n_components, :]

       

    def transform(self, X):
        #     X_pca = X.dot(self.principal_components)
        X_array = np.asarray(X)
        return X_array.dot(self.principal_components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def stratified_sampling(y, ratio, replace = True):
    #  Inputs:
    #     y: a 1-d array of class labels
    #     0 < ratio < 1: number of samples = len(y) * ratio
    #     replace = True: sample with replacement
    #     replace = False: sample without replacement
    #  Output:
    #     sample: indices of stratified sampled points
    #             (ratio is the same across each class,
    #             samples for each class = int(np.ceil(ratio * # data in each class)) )

    if ratio<=0 or ratio>=1:
        raise Exception("ratio must be 0 < ratio < 1.")
    y_array = np.asarray(y)
    # Write your own code below
    unique_classes = np.unique(y_array)
    
    sample=[]
    samples_per_class= np.ceil(ratio * len(y_array) / len(unique_classes)).astype(int)
    
    for class_label in unique_classes:
        class_indices = np.where(y_array==class_label)[0]
        
        if replace:
            class_sample=np.random.choice(class_indices, size=samples_per_class, replace=True)
        else:
            class_sample = np.random.choice(class_indices, size=min(samples_per_class, len(class_indices)),replace=False)
                                           # replace=False)
        sample.extend(class_sample.tolist())
       
            
                      
        
    

    return np.array(sample).astype(int)
