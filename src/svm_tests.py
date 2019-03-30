# -*- coding: utf-8 -*-

import sys
sys.path.append('./src')

from classifiers import KSVM, KSVM_pool
from kernel import kernel
import numpy as np
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

model1 = KSVM_pool(ksvm_list, fit_weights = True)
res = evaluateCVpool(model1, K_list1, n_lim=2000, n_folds=5, n_reps=1, verbose=True)

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



