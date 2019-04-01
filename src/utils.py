import numpy as np
import pandas as pd
import math
import csv
import os
from regression import logistic_reg
from classifiers import KSVM, KSVM_pool
from kernel import kernel


def train_val_split(X, y, p=0.1):
    l = len(X)
    sep = math.floor(l * p)
    y = 2 * y - 1
    Xtr = X[:-sep]
    ytr = y[:-sep]
    Xval = X[-sep:]
    yval = y[-sep:]
    return Xtr, ytr, Xval, yval


def accuracy(ytrue, ypred):
    return np.sum(ytrue == ypred) / ytrue.shape[0]


def center_graam_matrix(K):
    """Centers the data in the feature spaces.

    Args:
        K (np array): A graam matrix

    Returns:
        type: The graam matrix centered of the data centered in the feature space.

    """
    n = K.shape[0]
    M = np.eye(n) - 1 / n
    return M @ K @ M


def compute_squared_distance(K):
    """Computes the squared distance matrix from the kernel matrix.
    Args:
        K (type): Description of parameter `K`.
    Returns:
        type: Description of returned object.
    """
    d = np.diag(K)[:, np.newaxis]
    return d + d.T - 2 * K


def evaluateCV(model, K, n_lim=2000, n_folds=5, n_reps=1, verbose=True):
    idxs = np.arange(n_lim)
    scores = np.zeros((n_reps, n_folds))
    for rep in range(n_reps):
        np.random.shuffle(idxs)
        for fold in range(n_folds):
            if verbose:
                print(f"Running cross validation, repetition: {rep + 1}/{n_reps}, fold: {fold + 1}/{n_folds}")
            test_idxs = idxs[fold * n_lim // n_folds:(fold + 1) * n_lim // n_folds]
            train_idxs = np.setdiff1d(idxs, test_idxs)
            Ktrain, ytrain = K.get_train(train_idxs)
            Kval, yval = K.get_valid(train_idxs, test_idxs)
            model.train(Ktrain, ytrain)
            ypred = model.predict(Kval)
            scores[rep, fold] = 1 - np.mean((ypred != yval))
    if verbose:
        print("Average score {:.3f}, std {:.3f}".format(np.mean(scores), np.std(scores)))
    return scores, np.mean(scores)


def evaluateCVpool(model, K_list, n_lim=2000, n_folds=5, n_reps=1, verbose=True):
    idxs = np.arange(n_lim)
    scores = np.zeros((n_reps, n_folds))
    for rep in range(n_reps):
        np.random.shuffle(idxs)
        for fold in range(n_folds):
            if verbose:
                print(f"Running cross validation, repetition: {rep + 1}/{n_reps}, fold: {fold + 1}/{n_folds}")
            test_idxs = idxs[fold * n_lim // n_folds:(fold + 1) * n_lim // n_folds]
            train_idxs = np.setdiff1d(idxs, test_idxs)
            Ktrain_list = [K.get_train(train_idxs)[0] for K in K_list]
            ytrain = K_list[0].get_train(train_idxs)[1]
            Kval_list = [K.get_valid(train_idxs, test_idxs)[0] for K in K_list]
            yval = K_list[0].get_valid(train_idxs, test_idxs)[1]
            model.train(Ktrain_list, ytrain)
            ypred = model.predict(Kval_list)
            scores[rep, fold] = 1 - np.mean((ypred != yval))
    if verbose:
        print("Average score {:.3f}, std {:.3f}".format(np.mean(scores), np.std(scores)))
    return scores, np.mean(scores)


def create_sol(models, kernels):
    '''This function creates the last vector of labels
    Input: 3 models, 3 sets of data
    output: 1 vector of (-1,1) of size 3000'''

    s1, s2, s3 = kernels
    Xtr1, ytr1 = s1.get_train2000()
    Xtest1 = s1.get_test2000()[0]
    Xtr2, ytr2 = s2.get_train2000()
    Xtest2 = s2.get_test2000()[0]
    Xtr3, ytr3 = s3.get_train2000()
    Xtest3 = s3.get_test2000()[0]

    m1, m2, m3 = models
    m1.train(Xtr1, ytr1)
    m2.train(Xtr2, ytr2)
    m3.train(Xtr3, ytr3)

    y1 = m1.predict(Xtest1)
    y2 = m2.predict(Xtest2)
    y3 = m3.predict(Xtest3)
    ytest = np.concatenate((y1, y2, y3))
    ytest[ytest == -1] = 0
    return ytest


def write_csv(Ytest, path):
    '''This function writes a vector in a csv file'''
    with open(path, mode="w", newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Id', 'Bound'])
        for idx, label in enumerate(Ytest):
            writer.writerow([str(idx), str(label)])


def write_sol(models, kernels, file="res"):
    '''This function writes the solution in the csv file.
    Input: models and kernels
    Output: none'''

    # Create directory
    dirName = 'result'
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    ytest = create_sol(models, kernels)
    write_csv(ytest, dirName + "/" + file + ".csv")
    print("Writing complete.")


def create_sol_pool(pool_models, kernels_lists):
    '''This function creates the last vector of labels
    Input: 3 pooling models, 3 sets of data
    output: 1 vector of (-1,1) of size 3000'''

    Kl1, Kl2, Kl3 = kernels_lists

    Kltrain1 = [K.get_train2000()[0] for K in Kl1]
    ytrain1 = Kl1[0].get_train2000()[1]
    Kltest1 = [K.get_test2000()[0] for K in Kl1]

    Kltrain2 = [K.get_train2000()[0] for K in Kl2]
    ytrain2 = Kl2[0].get_train2000()[1]
    Kltest2 = [K.get_test2000()[0] for K in Kl2]

    Kltrain3 = [K.get_train2000()[0] for K in Kl3]
    ytrain3 = Kl3[0].get_train2000()[1]
    Kltest3 = [K.get_test2000()[0] for K in Kl3]

    m1, m2, m3 = pool_models
    print("Training the first dataset's model")
    m1.train(Kltrain1, ytrain1)
    print("Training the second dataset's model")
    m2.train(Kltrain2, ytrain2)
    print("Training the third dataset's model")
    m3.train(Kltrain3, ytrain3)

    print("Predicting with the first dataset's model")
    y1 = m1.predict(Kltest1)
    print("Predicting with the second dataset's model")
    y2 = m2.predict(Kltest2)
    print("Predicting with the third dataset's model")
    y3 = m3.predict(Kltest3)
    ytest = np.concatenate((y1, y2, y3))
    ytest[ytest == -1] = 0
    return ytest


def make_kernel_specs(s_list, k_list, m_list, gaussian=None, gaussian_auto=False):
    kernel_specs = []
    for s in s_list:
        for k in k_list:
            for m in m_list:
                kernel_specs.append(dict(s=s, k=k, m=m, gaussian=gaussian, gaussian_auto=gaussian_auto))
    return kernel_specs


def load_pooling_lists(kernel_specs, path_to_lambda, with_intercept=True):
    models_lists = [[], [], []]
    kernels_lists = [[], [], []]
    lambda_df = pd.read_csv(path_to_lambda)
    for spec in kernel_specs:
        s = spec["s"]
        k = spec["k"]
        m = spec["m"]
        if "gaussian" in spec.keys():
            gaussian = spec["gaussian"]
        else:
            gaussian = None
        if "gaussian_auto" in spec.keys():
            gaussian_auto = spec["gaussian_auto"]
        else:
            gaussian_auto = False
        kernels_lists[s].append(kernel(s=s, k=k, m=m, center=True, gaussian_auto=gaussian_auto,
                                       gaussian=gaussian, normalize_before_gaussian=False, normalize=True))
        if not (gaussian_auto):
            lambda_row = lambda_df.loc[(lambda_df["dataset"] == s) & (lambda_df["k"] == k) & (lambda_df["m"] == m)]
            print("dataset: {}, linear mismatch kernel (k: {}, m: {}), SVM parameter lambda: {:.3e}".format(s, k, m,
                float(lambda_row["best_lambda"])))
            models_lists[s].append(KSVM(float(lambda_row["best_lambda"]), with_intercept=with_intercept))
        else:
            models_lists[s].append(KSVM(1e-4, with_intercept=with_intercept))
            print(
                "dataset: {}, gaussian mismatch kernel (k: {}, m: {}), kernel parameter sigma^2 : {:.3e}, SVM parameter lambda: {:.3e}".format(
                    s, k, m, kernels_lists[s][-1].gaussian / 2, 0.0001))

        # models_lists[s].append(KSVM(1e-4))
    return models_lists, kernels_lists


def fit_weights_pool(model, K_list, n_lim=2000, n_svm=1800):
    svm_idxs = np.arange(n_svm)
    log_idxs = np.arange(n_svm, n_lim)
    Ksvm_list = [K.get_train(svm_idxs)[0] for K in K_list]
    ysvm = K_list[0].get_train(svm_idxs)[1]
    Klog_list = [K.get_valid(svm_idxs, log_idxs)[0] for K in K_list]
    ylog = K_list[0].get_valid(svm_idxs, log_idxs)[1]
    model.train(Ksvm_list, ysvm)
    ysvm_pred = []
    for i, Ksvm in enumerate(Ksvm_list):
        ysvm_pred.append(model.ksvm_list[i].predict(Klog_list[i], return_float=True))
    ysvm_pred = np.array(ysvm_pred).T
    model.weights = logistic_reg(ysvm_pred, ylog, intercept=False)
    print(model.weights)
    return model.weights
