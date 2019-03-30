import numpy as np
from cvxopt import matrix, solvers
from regression import ridge


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

    def __init__(self, lam=1, with_intercept = False):
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
            A = matrix(np.ones((1,n)))
            b = matrix(np.zeros(1))
            solvers.options['show_progress'] = False
            solution = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
            self.alpha = np.array(solution['x'])[:, 0]
            self.support_vectors = np.where(np.abs(self.alpha) > 1e-5)[0]
            self.alpha_short = self.alpha[self.support_vectors]
            #Calcul de l'intercept :
            i = np.where( (self.alpha>0) * (self.alpha < 1 / (2 * n * self.lam)) )[0][0]
            self.intercept = 1 - (Ktrain@self.alpha)[i]
        else =
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
            self.weights = ridge(ytrain_pred, ytrain, 1)
            print(self.weights.shape, self.weights)

    def predict(self, Kpred_list):
        ypred = []
        for i, Kpred in enumerate(Kpred_list):
            ypred.append(self.ksvm_list[i].predict(Kpred))
        ypred = np.array(ypred)
        ypred = np.array(np.sum(self.weights[:, None]*ypred, axis=0) > 0, dtype=int)
        ypred = ypred * 2 - 1
        return ypred



# class Logistic_pool:
#
#     def __init__(self, ksvm_list, lam_logistic):
#         self.n_ksvn = len(ksvm_list)
#         self.ksvm_list = ksvm_list
#         self.lam_logistic = lam_logistic
#
#     def train(self, Ktrain_list, ytrain):
#         for i, Ktrain in enumerate(Ktrain_list):
#             self.ksvm_list[i].train(Ktrain, ytrain)
#
#
#     def predict(self, Ktrain_list, ytrain, normalize = True):
#         ypred = []
#         for i, Kpred in enumerate(Kpred_list):
#             ypred.append(self.ksvm_list[i].predict(Kpred, return_float=True))
#         ypred = np.array(ypred)
#         ypred = y_pred / np.sqrt(np.mean(ypred**2, axis=0))
#         ypred =
#         return ypred
