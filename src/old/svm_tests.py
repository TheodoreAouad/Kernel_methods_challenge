# -*- coding: utf-8 -*-

import sys

sys.path.append('./src')

from classifiers import KSVM, KSVM_pool
from kernel import kernel
import numpy as np
import pandas as pd
from utils import evaluateCV, evaluateCVpool
from hp_opti import HPOptimizer
from write_csv import create_sol_pool, write_csv

# %%
s = 0
k = 8
m = 1
gamma = 2000
lam = 1e-5

K = kernel(s, k, m, gaussian_auto=True)


# print(K.gaussian)
# ksvm = KSVM(lam, with_intercept = True)
# optimizer = HPOptimizer(ksvm, K, bounds = [-6, 2])
# optimizer.explore(9, "GridSearch")
# %%

# evaluateCV(ksvm, K, n_lim=2000, n_folds=10, n_reps=5, verbose=True)


# %%







s_list = [0, 1, 2]
k_list = [6, 7, 8, 9, 10]
m_list = [1]

kernel_specs = make_kernel_specs(s_list, k_list, m_list, gaussian=None, gaussian_auto=False)
kernel_specs += make_kernel_specs(s_list, k_list[2:], m_list, gaussian=None, gaussian_auto=True)

# %%

models_lists, kernels_lists = load_pooling_lists(kernel_specs, "./lambdas/lambdas.csv", with_intercept=False)

# %%
pool_models = [KSVM_pool(models_lists[0], fit_weights=False),
               KSVM_pool(models_lists[1], fit_weights=False),
               KSVM_pool(models_lists[2], fit_weights=False)]

# %%






pool_models[0].weights = fit_weights_pool(pool_models[0], kernels_lists[0], n_lim=2000, n_svm=1800)
pool_models[1].weights = fit_weights_pool(pool_models[1], kernels_lists[1], n_lim=2000, n_svm=1800)
pool_models[2].weights = fit_weights_pool(pool_models[2], kernels_lists[2], n_lim=2000, n_svm=1800)

# %%

res = evaluateCVpool(pool_models[0], kernels_lists[0], n_lim=2000, n_folds=10, n_reps=3, verbose=True)
# res = evaluateCVpool(pool_models[1], kernels_lists[1], n_lim=2000, n_folds=5, n_reps=3, verbose=True)
# res = evaluateCVpool(pool_models[2], kernels_lists[2], n_lim=2000, n_folds=5, n_reps=3, verbose=True)

# %%
# pool_models = [model1, model2, model3]
# kernels_lists = [K_list1, K_list2, K_list3]


y_pred = create_sol_pool(pool_models, kernels_lists)

write_csv(y_pred, "./res7.csv")
