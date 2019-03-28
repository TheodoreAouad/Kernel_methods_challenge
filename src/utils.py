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

def evaluateCV(model, K, n_lim = 2000, n_folds = 5, n_reps = 1, lam = 0.1, verbose = True):
    idxs = np.arange(n_lim)
    scores = np.zeros((n_reps, n_folds))
    for rep in range(n_reps):
        np.random.shuffle(idxs)
        for fold in range(n_folds):
            if verbose:
                print(f"Running cross validation, repetition: {rep+1}/{n_reps}, fold: {fold+1}/{n_folds}")
            test_idxs = idxs[fold*n_lim//n_folds:(fold+1)*n_lim//n_folds]
            train_idxs = np.setdiff1d(idxs, test_idxs)
            Ktrain, ytrain = K.get_train(train_idxs)
            Kval, yval = K.get_valid(train_idxs, test_idxs)
            model.train(Ktrain, ytrain)
            ypred = model.predict(Kval)
            scores[rep, fold] = 1-np.mean((ypred!=yval))
    if verbose:
        print("Average score {:.3f}, std {:.3f}".format(np.mean(scores), np.std(scores)))
    return scores, np.mean(scores)
