# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv

def ridge(X, y, lam= 0.001):
    X = np.asarray(X)
    y = np.asarray(y)
    I = np.eye(X.shape[1])
    return inv(X.T@X + lam*I)@X.T@y


def logistic_reg(Xtrain, ytrain, intercept = False):
    ytrain = (ytrain + 1)/2
    if intercept:
        Xtrain = np.c_[np.ones(Xtrain.shape[0]), Xtrain]
    N , d = Xtrain.shape
    w_ = np.zeros((d,), dtype = float)
    eta = np.zeros((N,1),dtype = float)
    itermax = 2000
    eps = 0.001
    for i in range(itermax):
        eta = sigma(Xtrain.dot(w_))
        D = np.diag(eta)
        # In case the hessian is numericaly not invertible
        if np.linalg.det(np.transpose(Xtrain)@ D@Xtrain) < eps:
            break
        w = w_ + np.linalg.inv(np.transpose(Xtrain)@D@Xtrain)@Xtrain.transpose() @ (ytrain-eta)
        if np.max(np.abs(w-w_))<eps:
            break
        print(np.max(np.abs(w-w_)))
        w_ = w
    return w_

def sigma(z):
    return 1/(1+np.exp(-z))

def predict_logistic_reg(w, Xtrain, ytrain, intercept = True, return_float=False):
    if intercept :
        Xtrain = np.c_[np.ones(Xtrain.shape[0]), Xtrain]
    score = Xtrain@w
    if return_float:
        return score
    else :
        return score > 0.5
