import numpy as np
from cvxopt import matrix, solvers


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


class KSVM:

    def __init__(self, lam = 1):
        #self.type = 1 # Pour implementer plus tard le SVM2 avec la squared hinge loss
        self.alpha = None
        self.alpha_short = None
        self.support_vectors = None
        self.gram_matrix = None
        self.lam = lam
        self.n_train = None

    def set_hyperparameters(self, lam):
        self.lam = lam

    def train(self, Ktrain, ytrain):
        #setting the environment for cvxopt
        n = len(ytrain)
        ytrain = ytrain.astype(float)
        P = matrix(Ktrain.astype(float))
        q = matrix(-ytrain)
        G = matrix( np.vstack([np.diag(ytrain),-1*np.diag(ytrain)]) )
        h = matrix( np.hstack([np.ones(n)*1/(2*n*self.lam), np.zeros(n)]))
        solvers.options['show_progress'] = False
        solution = solvers.qp(P=P, q=q, G=G, h=h)
        self.alpha= np.array(solution['x'])[:,0]
        self.support_vectors = np.where(np.abs(self.alpha) > 1e-5)[0]
        self.alpha_short = self.alpha[self.support_vectors]


    def predict(self, Kpred, return_float = False):
        if return_float:
            return Kpred@self.alpha
        else:
            ypred = np.array(Kpred@self.alpha>0, dtype=int)
            ypred = ypred*2 - 1
            return ypred
