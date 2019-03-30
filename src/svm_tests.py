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

#%%
s = 0
k = 8
m = 1
gamma = 2000
lam = 1e-5

K = kernel(s, k, m, gaussian_auto = False)
print(K.gaussian)
ksvm = KSVM(lam, with_intercept = True)
#optimizer = HPOptimizer(ksvm, K, bounds = [-6, 2])
#optimizer.explore(9, "GridSearch")
#%%

evaluateCV(ksvm, K, n_lim=2000, n_folds=10, n_reps=3, verbose=True)


#%%


def make_kernel_specs(s_list, k_list, m_list, gaussian = None, gaussian_auto = False):
    kernel_specs = []
    for s in s_list:
        for k in k_list:
            for m in m_list:
                kernel_specs.append(dict( s = s,  k=k, m=m, gaussian=gaussian, gaussian_auto = gaussian_auto))
    return kernel_specs

def load_pooling_lists(kernel_specs, path_to_lambda, with_intercept = True):
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
        if not(gaussian_auto):
            lambda_row = lambda_df.loc[(lambda_df["dataset"] == s)&(lambda_df["k"] == k)&(lambda_df["m"] == m)]#&
            # (lambda_df["gaussian"] == gaussian)]
            print(float(lambda_row["best_lambda"]))
            models_lists[s].append(KSVM(float(lambda_row["best_lambda"])))
        else:
            models_lists[s].append(KSVM(1e-4, with_intercept = with_intercept))


        #models_lists[s].append(KSVM(1e-4))
    return models_lists, kernels_lists

s_list = [0, 1, 2]
k_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
m_list = [0, 1]

kernel_specs = make_kernel_specs(s_list, k_list, m_list, gaussian = None, gaussian_auto = True)
models_lists, kernels_lists = load_pooling_lists(kernel_specs, "./lambdas/lambdas.csv", with_intercept = True)
#%%
pool_models = [KSVM_pool(models_lists[0], fit_weights = False),
               KSVM_pool(models_lists[1], fit_weights = False),
               KSVM_pool(models_lists[2], fit_weights = False)]


#%%

res = evaluateCVpool(pool_models[0], kernels_lists[0], n_lim=2000, n_folds=5, n_reps=3, verbose=True)
#res = evaluateCVpool(pool_models[1], kernels_lists[1], n_lim=2000, n_folds=5, n_reps=3, verbose=True)
#res = evaluateCVpool(pool_models[2], kernels_lists[2], n_lim=2000, n_folds=5, n_reps=3, verbose=True)

#%%
#pool_models = [model1, model2, model3]
#kernels_lists = [K_list1, K_list2, K_list3]


y_pred = create_sol_pool(pool_models, kernels_lists)

write_csv(y_pred, "./res3.csv")




