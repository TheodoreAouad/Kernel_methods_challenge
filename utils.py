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
