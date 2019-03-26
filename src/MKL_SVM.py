import numpy as np
from classifiers import KSVM

class  MKL_SVM():

    def __init__(self, lam = 0.1):
        self.lam = lam
        self.alpha = None
        self.alpha_short = None
        self.support_vectors = None
        self.lam = lam
        self.n_train = None
        self.eta = None
        self.KSVM = None

    def set_hyperparameters(self, lam):
        self.lam = lam

    def train(self, Ktrain_list, ytrain,  step_size=0.01 ,eps = 1e-3):
        M = len(Ktrain_list)
        eta = np.ones(M)/M
        converged = False
        ksvm = KSVM(self.lam)
        while not converged:
            #Computing the sum matrix
            K=0
            for m in range(M):
                K += Ktrain_list[m]*eta[m]
            #Solving the SVM:
            ksvm.train(K, ytrain)
            gamma_star = ksvm.alpha * 2 * self.lam /ytrain
            #Gradient step:
            gradient = np.array([-self.lam * gamma_star.T@K@gamma_star for K in Ktrain_list])
            step = -step_size* gradient
            print(gradient)
            print(eta)
            eta += step
            if np.linalg.norm(step)<eps:
                converged = True
            #Projecting eta on L1 norm

        self.eta = eta
        self.KSVM = ksvm


    def make_arguments(self, kernel_list, idxs):
        Ktrain_list = [K.get_train(idxs)[0] for K in kernel_list]
        ytrain = kernel_list[0].get_train(idxs)[1]
        return Ktrain_list, ytrain



    def set_hperparameters(self, lam):
        pass

    def predict(self, Kpred, return_float = False):
        return self.KSVM.predict(Kpred,return_float)
