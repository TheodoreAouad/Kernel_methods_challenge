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


class KNN():
    '''K nearest-neighbors'''

    def __init__(self):
        self.kernel = None
        self.gram = None
        self.data = None
        self.label = None

    def train(self,X,y,kernel):
        self.kernel = kernel
        self.data = X
        self.label = y

        #gram = np.zeros((X.shape[0],X.shape[0]))
        #for i in range(X.shape[0]):
        #  for j in range(i,X.shape[0]):
        #    gram[i,j] = kernel(X[i],X[j])
        #    gram[j,i] = gram[i,j]
        #self.gram = gram
        return True

    def predict_one(self,x,k):
        dists = np.zeros(self.data.shape[0])
        for i in range(dists.shape[0]):
            dists[i] = self.Kdist(x,self.data[i])
        neighbors = self.label[dists.argsort()][:k]
        uniq,cnts = np.unique(neighbors, return_counts=1)
        return uniq[cnts.argsort()][-1]

    def predict(self,X,k):
        labels = np.zeros(X.shape[0])
        for i,x in enumerate(X):
            labels[i] = self.predict_one(x,k)
        return labels

    def Kdist(self,x,y):
        return (self.kernel(x,x) + self.kernel(y,y) - 2*self.kernel(x,y))**0.5


class KSVM_revised:

    def __init__(self, lam = 1):
        #self.type = 1 # Pour implementer plus tard le SVM2 avec la squared hinge loss
        self.alpha = None
        self.alpha_short = None
        self.support_vectors = None
        self.gram_matrix = None
        self.lam = lam
        self.n_train = None

    def load_gram_matrix(self, path, nlim= None):
        if nlim is None:
            self.gram_matrix = np.load(path)['arr_0']
        else:
            self.gram_matrix = np.load(path)['arr_0'][:nlim, :nlim]

    def train(self, ytrain, train_idxs = None, gram_matrix=None, lam = None,  center = True):
        if lam is not None:
            self.lam = lam
        if gram_matrix is None:
            gram_matrix = self.gram_matrix
        self.n_train = len(ytrain)
        #setting the environment for cvxopt
        n = self.n_train
        y = ytrain.astype(float)
        if train_idxs is None:
            Ktrain = self.gram_matrix[:n, :n]
        else:
            Ktrain = self.gram_matrix[train_idxs][:, train_idxs]
        if center:
            M = np.eye(n) - np.ones((n,n)) * 1/n
            Ktrain = M@Ktrain@M
        P = matrix(Ktrain.astype(float))
        q = matrix(-y)
        G = matrix( np.vstack([np.diag(y),-1*np.diag(y)]) )
        h = matrix( np.hstack([np.ones(n)*1/(2*n*self.lam), np.zeros(n)]))
        solvers.options['show_progress'] = False
        solution = solvers.qp(P=P, q=q, G=G, h=h)
        self.alpha= np.array(solution['x'])[:,0]
        self.support_vectors = np.where(np.abs(self.alpha) > 1e-5)[0]
        self.alpha_short = self.alpha[self.support_vectors]


    def predict(self, train_idxs=None, test_idxs = None, return_float = False):
        if test_idxs is None:
            Ktest = self.gram_matrix[:self.n_train, self.n_train:]
        else:
            Ktest = self.gram_matrix[test_idxs][:, train_idxs]
        if return_float:
            return self.alpha@Ktest
        else:
            return np.array(Ktest@self.alpha>0, dtype=int)




