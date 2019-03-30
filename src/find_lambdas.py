import hp_opti as op
import classifiers as cl
from kernel import kernel
import os
import csv
from time import time

def write_lambdas(path_to_gram,path_to_labels,ss,ks,ms,center,normalize,gaussians,bounds,mode="w",n_iters=10,n_folds=10):
    '''This function computes the optimal lambdas according to the parameters.

    Input:
        ss: sets to analyze
        ks: kernels to analyze
        center: bool. whether or not we want to center the kernels.
        normalize: idem center.
        gaussians: variances to try.
        bounds: boundaries of lambdas
        mode: "w" or "a": whether or not we overwrite the current file or not
    Output:
        dictionary of keys:parameters, values:(lambdas,scores)
    '''

    dirName = 'lambdas'
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

    if mode == "w":
        with open("lambdas/lambdas.csv",mode="w",newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["dataset","k","m","center","normalize","gaussian","best_lambda","best_score","all_lambdas","all_scores"])
    res = {}
    start = time()
    for s in ss:
        for k in ks:
            for m in ms:
                for gaussian in gaussians:
                    print("s={}, k={}, m={}, gamma={}".format(s,k,m,gaussian))
                    K = kernel(s=s,k=k,m=m,center=center,gaussian=gaussian,normalize=normalize,path_to_gram=path_to_gram,path_to_labels=path_to_labels)
                    ksvm = cl.KSVM()
                    optimizer = op.HPOptimizer(ksvm,K,bounds=bounds,n_folds=n_folds)
                    optimizer.explore(n_iters=n_iters,method="GridSearch")
                    best_lam = 10**optimizer.best_hp[0]
                    best_score = optimizer.best_score
                    all_lams = [10**l[0] for l in optimizer.hp_list]
                    all_scores = optimizer.score_list
                    towrite = [s,k,m,center,normalize,gaussian,best_lam,best_score,all_lams,all_scores]
                    with open("lambdas/lambdas.csv",mode="a",newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([str(x) for x in towrite])
                    res[(s,k,m,center,normalize,gaussian)] = (best_lam,best_score)
                    print("====================== time: {} s\n".format(round(time()-start)))

    return res

write_lambdas("./gram_matrices/mismatch/","./data/",[2] ,range(2, 11),range(2),True,True,[None],[-6, -4],mode="a",n_iters=20,n_folds=5)