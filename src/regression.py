# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv

def ridge(X, y, lam= 0.001):
    X = np.asarray(X)
    y = np.asarray(y)
    I = np.eye(X.shape[1])
    return inv(X.T@X + lam*I)@X.T@y