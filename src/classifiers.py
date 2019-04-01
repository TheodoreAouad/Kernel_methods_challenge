import numpy as np
from cvxopt import matrix, solvers
from regression import ridge, logistic_reg, predict_logistic_reg
import matplotlib.pyplot as plt


class KNN():
    '''K nearest-neighbors'''

    def __init__(self):
        self.kernel = None
        self.gram = None
        self.data = None
        self.label = None

    def train(self, X, y, kernel):
        self.kernel = kernel
        self.data = X
        self.label = y

        # gram = np.zeros((X.shape[0],X.shape[0]))
        # for i in range(X.shape[0]):
        #  for j in range(i,X.shape[0]):
        #    gram[i,j] = kernel(X[i],X[j])
        #    gram[j,i] = gram[i,j]
        # self.gram = gram
        return True

    def predict_one(self, x, k):
        dists = np.zeros(self.data.shape[0])
        for i in range(dists.shape[0]):
            dists[i] = self.Kdist(x, self.data[i])
        neighbors = self.label[dists.argsort()][:k]
        uniq, cnts = np.unique(neighbors, return_counts=1)
        return uniq[cnts.argsort()][-1]

    def predict(self, X, k):
        labels = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            labels[i] = self.predict_one(x, k)
        return labels

    def Kdist(self, x, y):
        return (self.kernel(x, x) + self.kernel(y, y) - 2 * self.kernel(x, y)) ** 0.5


class KSVM:

    def __init__(self, lam=1, with_intercept=False):
        # self.type = 1 # Pour implementer plus tard le SVM2 avec la squared hinge loss
        self.alpha = None
        self.alpha_short = None
        self.support_vectors = None
        self.gram_matrix = None
        self.lam = lam
        self.n_train = None
        self.intercept = None
        self.with_intercept = with_intercept

    def set_hyperparameters(self, lam):
        self.lam = lam

    def train(self, Ktrain, ytrain):
        # setting the environment for cvxopt
        n = len(ytrain)
        ytrain = ytrain.astype(float)
        P = matrix(Ktrain.astype(float))
        q = matrix(-ytrain)
        G = matrix(np.vstack([np.diag(ytrain), -1 * np.diag(ytrain)]))
        h = matrix(np.hstack([np.ones(n) * 1 / (2 * n * self.lam), np.zeros(n)]))
        if self.with_intercept:
            A = matrix(np.ones((1, n)))
            b = matrix(np.zeros(1))
            solvers.options['show_progress'] = False
            solution = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
            self.alpha = np.array(solution['x'])[:, 0]
            self.support_vectors = np.where(np.abs(self.alpha) > 1e-5)[0]
            self.alpha_short = self.alpha[self.support_vectors]
            # Calcul de l'intercept :
            i = np.where((self.alpha > 0) * (self.alpha < 1 / (2 * n * self.lam)))[0][0]
            self.intercept = ytrain[i] - (Ktrain @ self.alpha)[i]
            print(self.intercept)
        else:
            solvers.options['show_progress'] = False
            solution = solvers.qp(P=P, q=q, G=G, h=h)
            self.alpha = np.array(solution['x'])[:, 0]
            self.support_vectors = np.where(np.abs(self.alpha) > 1e-5)[0]
            self.alpha_short = self.alpha[self.support_vectors]

    def predict(self, Kpred, return_float=False):
        score = Kpred @ self.alpha
        if self.with_intercept:
            score += self.intercept
        if return_float:
            return score
        else:
            ypred = np.array(score > 0, dtype=int)
            ypred = ypred * 2 - 1
            return ypred


class KSVM_pool:

    def __init__(self, ksvm_list, fit_weights=False):
        self.ksvm_list = ksvm_list
        self.n_ksvn = len(ksvm_list)
        self.weights = np.ones(self.n_ksvn) / self.n_ksvn
        self.fit_weights = fit_weights

    def train(self, Ktrain_list, ytrain):
        for i, Ktrain in enumerate(Ktrain_list):
            self.ksvm_list[i].train(Ktrain, ytrain)
        if self.fit_weights:
            ytrain_pred = []
            for i, Ktrain in enumerate(Ktrain_list):
                ytrain_pred.append(self.ksvm_list[i].predict(Ktrain))
            ytrain_pred = np.array(ytrain_pred).T
            self.weights = logistic_reg(ytrain_pred, ytrain)
            print(self.weights.shape, self.weights)

    def predict(self, Kpred_list):
        ypred = []
        for i, Kpred in enumerate(Kpred_list):
            ypred.append(self.ksvm_list[i].predict(Kpred))
        ypred = np.array(ypred)
        ypred = np.array(np.sum(self.weights[:, None] * ypred, axis=0) > 0, dtype=int)
        ypred = ypred * 2 - 1
        return ypred

class MKL_SVM():

    def __init__(self, lam=0.1):
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

    def train(self, Ktrain_list, ytrain, step_size=0.01, eps=1e-3, eta0=None):
        J = []
        Train_acc = []
        M = len(Ktrain_list)
        if eta0 is None:
            eta = np.ones(M) / M
        else:
            eta = eta0
        converged = False
        ksvm = KSVM(self.lam)
        while not converged:
            # Computing the sum matrix
            K = 0
            for m in range(M):
                K += Ktrain_list[m] * eta[m]
            # Solving the SVM:
            ksvm.train(K, ytrain)
            J += [2 * ksvm.alpha.T @ ytrain - ksvm.alpha.T @ K @ ksvm.alpha]
            Train_acc += [u.accuracy(ytrain, ksvm.predict(K))]
            # gamma_star = ksvm.alpha * 2 * self.lam /ytrain
            gamma_star = ksvm.alpha
            # Gradient step:
            gradient = np.array([-self.lam * gamma_star.T @ K @ gamma_star for K in Ktrain_list])
            step = - step_size * gradient
            eta_new = eta + step
            # Projecting eta on L1 norm
            eta_new = self.projection_L1(eta_new)
            if np.linalg.norm(eta - eta_new) < eps:
                converged = True
            # Projecting eta on L1 norm
            eta = eta_new
            print(eta)
        self.eta = eta
        self.KSVM = ksvm
        plt.plot(J, label='SVM_loss')
        plt.figure()
        plt.plot(Train_acc, label="training accuracy")

    def make_arguments(self, kernel_list, idxs):
        Ktrain_list = [K.get_train(idxs)[0] for K in kernel_list]
        ytrain = kernel_list[0].get_train(idxs)[1]
        return Ktrain_list, ytrain

    def make_sum_kernel(self, kernel_list):
        eta = self.eta
        M = len(eta)
        K = 0
        for m in range(M):
            K += kernel_list[m].gram_matrix * eta[m]
        combKernel = k.kernel(s=kernel_list[0].s, Graam_matrix=K)
        return combKernel

    def set_hperparameters(self, lam):
        print("Not implemented yet")

    def predict(self, Kpred, return_float=False):
        return self.KSVM.predict(Kpred, return_float)

    def projection_L1(self, eta):
        """" Returns the projection of eta on the L1 bowl
        """
        proj = np.sign(eta) * np.maximum(eta - (np.sum(np.abs(eta)) - 1) / len(eta), 0)
        if np.sum(np.abs(proj)) > 1 + 1e-6:
            proj[proj != 0] = self.projection_L1(proj[proj != 0])
        return proj
