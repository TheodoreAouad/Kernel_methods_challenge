import numpy as np
from cvxopt import matrix, solvers


class KSVM():

    def __init__(self):
        #self.type = 1 # Pour implementer plus tard le SVM2 avec la squared hinge loss
        self.solution = None
        self.alpha = None
        self.alpha_short = None
        self.support_vectors = None
        self.kernel = None
        self.data = None
        self.Ktrain = None
        self.Kpred = None



    def train(self, X, y, kernel, lbda, center = False):
        self.kernel = kernel
        self.data = X

        K = kernel(X,X)
        self.train_fromK(K, y, lbda, center = center)
        return True

    def train_fromK(self, K, y, lbda, center = False):
        n = K.shape[0]
        if center:
            M = np.eye(n) - np.ones((n,n)) * 1/n
            K = M@K@M

        self.Ktrain = K
        y = y.astype(float)
        P = matrix(K)
        q = matrix(-y)
        G = matrix( np.vstack([np.diag(y),-1*np.diag(y)]) )
        h = matrix( np.hstack([np.ones(n)*1/(2*n*lbda), np.zeros(n)]))

        solution = solvers.qp(P, q, G= G, h=h)
        self.compute_SV(solution)
        return True


    def compute_SV(self, solution):
        self.alpha= np.array(solution['x'])[:,0]
        self.support_vectors = np.where(np.abs(self.alpha) > 1e-5)[0]
        self.alpha_short = self.alpha[self.support_vectors]

    def predict_score(self, X):
        K = self.kernel(X, self.data[self.support_vectors])
        self.Kpred = K
        score = self.predict_score_fromK(K, only_support=True)
        return score

    def predict(self, X):
        score = self.predict_score(X)
        return score>0

    def predict_score_fromK(self, K, only_support = False):
        if only_support :
            score = K@self.alpha_short
        else :
            score = K@self.alpha
        return score

    def predict_fromK(self, K, only_support = False):
        return self.predict_score_fromK(K, only_support=only_support) > 0
