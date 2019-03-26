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



## to finish
#def evaluateCV(X, y, n_folds = 5, n_reps = 10):
#    n, p = X.shape
#    idxs = np.arange(n)
#    scores = np.zeros((n_reps, n_folds))
#    for rep in range(n_reps):
#        np.random.shuffle(idxs)
#        for fold in range(n_folds):
#            test_idxs = idxs[fold*n//n_folds:(fold+1)*n//n_folds]
#            train_idxs = np.setdiff1d(idxs, test_idxs)
#            Xtrain, ytrain = X[train_idxs, :], y[train_idxs]
#            Xtest, ytest = X[test_idxs, :], y[test_idxs]
