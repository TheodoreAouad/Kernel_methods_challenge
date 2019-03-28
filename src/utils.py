import numpy as np
import math

def train_val_split(X,y, p=0.1):
    l = len(X)
    sep = math.floor(l*p)
    y = 2*y -1
    Xtr = X[:-sep]
    ytr = y[:-sep]
    Xval = X[-sep:]
    yval = y[-sep:]
    return Xtr, ytr, Xval, yval

def accuracy(ytrue, ypred):
    return np.sum(ytrue==ypred)/ytrue.shape[0]

def center_graam_matrix(K):
    """Centers the data in the feature spaces.

    Args:
        K (np array): A graam matrix

    Returns:
        type: The graam matrix centered of the data centered in the feature space.

    """
    n = K.shape[0]
    M = np.eye(n) - 1/n
    return M@K@M

def compute_squared_distance(K):
    """Computes the squared distance matrix from the kernel matrix.
    Args:
        K (type): Description of parameter `K`.
    Returns:
        type: Description of returned object.
    """
    d = np.diag(K)[:,np.newaxis]
    return d + d.T - 2*K
