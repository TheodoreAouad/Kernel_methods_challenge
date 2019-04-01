# -*- coding: utf-8 -*-

import sys
sys.path.append('./src')

from mismatch import precalc_gram
from utils import make_kernel_specs, load_pooling_lists, fit_weights_pool, create_sol_pool, write_csv
from classifiers import KSVM_pool


s_list = [0, 1, 2]
k_list = [6, 7, 8, 9, 10]
m_list = [1]

precalc_gram("./gram_matrices/mismatch", k_list, m_list, s_list, data_path = "./data/")

kernel_specs = make_kernel_specs(s_list, k_list, m_list, gaussian=None, gaussian_auto=False)
kernel_specs += make_kernel_specs(s_list, k_list[2:], m_list, gaussian=None, gaussian_auto=True)

print("\nLoading kernels and SVMs:")
models_lists, kernels_lists = load_pooling_lists(kernel_specs, "./lambdas/lambdas.csv", with_intercept=False)

print("\nCreating the 3 pooling models and fitting their weights:")
pool_models = [KSVM_pool(models_lists[i], fit_weights=False) for i in range(3)]
for i in range(3):
    pool_models[i].weights = fit_weights_pool(pool_models[i], kernels_lists[i], n_lim=2000, n_svm=1800)

print("\nCalculating the prediction and storing them in ./Yte.csv")
Yte = create_sol_pool(pool_models, kernels_lists)
write_csv(Yte, "./Yte.csv")