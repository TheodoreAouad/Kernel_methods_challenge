# -*- coding: utf-8 -*-

import sys
sys.path.append('./src')

from classifiers import KSVM, KSVM_pool
from kernel import kernel
import numpy as np
import pandas as pd
from utils import evaluateCV, evaluateCVpool
from hp_opti import HPOptimizer

#%%
s = 0
k = 8
m = 1
gamma = 2000
lam = 1e-5

K = kernel(s, k, m, gaussian = gamma)
ksvm = KSVM(lam)
optimizer = HPOptimizer(ksvm, K, bounds = [-6, -3])
optimizer.explore(10, "GridSearch")


#%%

res = evaluateCV(ksvm, K, n_folds = 10, n_reps = 3)


#%%

kernel_specs = []

def make_kernel_specs(s_list, k_list, m_list, gaussian = None, gaussian_auto = False):
    kernel_specs = []
    for s in s_list:
        for k in k_list:
            for m in m_list:
                kernel_specs.append(dict( s = s,  k=k, m=m, gaussian=gaussian, gaussian_auto = gaussian_auto))
    return kernel_specs

def load_pooling_lists(kernel_specs, path_to_lambda):
    models_lists = [[], [], []]
    kernels_lists = [[], [], []]
    lambda_df = pd.read_csv(path_to_lambda)
    for spec in kernel_specs:
        s = spec["s"]
        k = spec["k"]
        m = spec["m"]
        if "gaussian" in spec.keys():
            gaussian  = spec["gaussian"]
        else:
            gaussian = None
        if "gaussian_auto" in spec.keys():
            gaussian_auto  = spec["gaussian_auto"]
        else:
            gaussian_auto = False
        kernels_lists[s].append(kernel(s=s, k=k, m=m, center = True, gaussian_auto = gaussian_auto,
                    gaussian = gaussian, normalize_before_gaussian=False, normalize = True))
        lambda_row = lambda_df.loc[(lambda_df["dataset"] == s)&(lambda_df["k"] == k)&(lambda_df["m"] == m)]#&
                                  # (lambda_df["gaussian"] == gaussian)]
        print(float(lambda_row["best_lambda"]))
        #models_lists[s].append(KSVM(float(lambda_row["best_lambda"])))
        models_lists[s].append(KSVM(1e-4))
    return models_lists, kernels_lists

s_list = [0, 1, 2]
k_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
m_list = [0, 1]

kernel_specs = make_kernel_specs(s_list, k_list, m_list, gaussian = None, gaussian_auto = False)
models_lists, kernels_lists = load_pooling_lists(kernel_specs, "./lambdas/lambdas.csv")
#%%
pool_models = [KSVM_pool(models_lists[0], fit_weights = False),
               KSVM_pool(models_lists[1], fit_weights = False),
               KSVM_pool(models_lists[2], fit_weights = False)]


#%%

s = 0
k_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
m_list = [0, 1]
gamma = None

K_list1 = []
for k in k_list:
    for m in m_list:
        K_list1.append(kernel(s, k, m, gaussian = gamma))

lam = 1e-4
ksvm_list = [KSVM(lam) for i in range(len(K_list1))]


#%%

res = evaluateCVpool(pool_models[0], kernels_lists[0], n_lim=2000, n_folds=5, n_reps=3, verbose=True)
res = evaluateCVpool(pool_models[1], kernels_lists[1], n_lim=2000, n_folds=5, n_reps=3, verbose=True)
res = evaluateCVpool(pool_models[2], kernels_lists[2], n_lim=2000, n_folds=5, n_reps=3, verbose=True)

#%%
pool_models = [model1, model2, model3]
kernels_lists = [K_list1, K_list2, K_list3]


y_pred = create_sol_pool(pool_models, kernels_lists)

#def create_pool_svm(k_list, m = 0, s=0, lam = 0.1):
#    ksvm_pool = []
#    for k in k_list:
#        gram_path = "./gram_matrices/mismatch/mismatch{}k@{}m@{}.npz".format(s, k, m)
#        ksvm_pool.append(KSVM_revised(lam))
#        ksvm_pool[-1].load_gram_matrix(gram_path, nlim = 2000)
#    return ksvm_pool
#
#def train_pool(ksvm_pool, ytrain, train_idxs):
#    for ksvm in ksvm_pool:
#        ksvm.train(ytrain, train_idxs, center=0)
#    return ksvm_pool
#
#def predict_pool(ksvm_pool, train_idxs, test_idxs):
#    ypred_pool = []
#    for ksvm in ksvm_pool:
#        ypred = ksvm.predict(train_idxs, test_idxs)
#        ypred_pool.append(ypred)
#    return np.mean(ypred_pool, axis=0)
#
#def evaluateCVpool(y, k_list, n_folds = 5, n_reps = 1, lam = 0.1):
#    n = len(y)
#    idxs = np.arange(n)
#    scores = np.zeros((n_reps, n_folds))
#    ksvm_pool = create_pool_svm(k_list, m = 1, s=0, lam = lam)
#    for rep in range(n_reps):
#        np.random.shuffle(idxs)
#        for fold in range(n_folds):
#            print(fold)
#            test_idxs = idxs[fold*n//n_folds:(fold+1)*n//n_folds]
#            train_idxs = np.setdiff1d(idxs, test_idxs)
#            ytrain, ytest = y[train_idxs], y[test_idxs]
#            ytrain[ytrain == 0] = -1
#            ksvm_pool = train_pool(ksvm_pool, ytrain, train_idxs)
#            ypred = predict_pool(ksvm_pool, train_idxs, test_idxs)
#            ypred = ypred>=0.5
#            scores[rep, fold] = 1-np.mean((ypred-ytest)**2)
#    return scores, np.mean(scores)



