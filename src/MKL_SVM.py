import numpy as np
from classifiers import KSVM
import kernel as k

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
            step = - step_size/self.lam * gradient
            eta_new = eta + step
            #Projecting eta on L1 norm
            eta_new= self.projection_L1(eta_new)
            if np.linalg.norm(eta-eta_new)<eps:
                converged = True
            #Projecting eta on L1 norm
            eta= eta_new
            print(eta)
        self.eta = eta
        self.KSVM = ksvm
        K=0
        for m in range(M):
            K += Ktrain_list[m]*eta[m]
        combKernel = k.kernel( Graam_matrix = K )
        return combKernel


    def make_arguments(self, kernel_list, idxs):
        Ktrain_list = [K.get_train(idxs)[0] for K in kernel_list]
        ytrain = kernel_list[0].get_train(idxs)[1]
        return Ktrain_list, ytrain



    def set_hperparameters(self, lam):
        print("Not implemented yet")

    def predict(self, return_float = False):
        return self.KSVM.predict(Kpred, return_float)

    def projection_L1(self, eta):
        """" Returns the projection of eta on the L1 bowl
        """
        proj = np.sign(eta) * np.maximum(eta - (np.sum(np.abs(eta))-1)/len(eta), 0)
        if np.sum(np.abs(proj)) > 1 + 1e-6:
            proj[proj != 0] = self.projection_L1(proj[proj != 0])
        return proj
