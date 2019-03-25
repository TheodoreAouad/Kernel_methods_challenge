import numpy as np

class spectrum():

    def __init__(self, k = 3):
            self.K = k
            self.N = None
            self.pre_index = None
            self.Graam = None

    def compute_pre_index(self, X):
        self.N = X.shape[0]
        self.Graam = np.zeros((self.N,self.N))
        pre_index = {}
        for i in range(self.N):
            seq = X[i]
            l = len(seq)
            for j in range(l-self.K):
                subseq = seq[j:j+self.K]
                if subseq in pre_index.keys():
                    pre_index[subseq][i] +=1
                else :
                    new = np.zeros(self.N)
                    new[i] = 1
                    pre_index[subseq] = new
        self.pre_index = pre_index

    def compute_Graam(self, X):
        self.compute_pre_index(X)
        for sub_seq in self.pre_index:
            V = self.pre_index[sub_seq]
            self.Graam += np.outer(V,V)
        return True

    def compute_embedding(self, Xtest):
        n = Xtest.shape[0]
        Mat = np.zeros((n, self.N))
        for i in range(n):
            seq = Xtest[i]
            l = len(seq)
            for j in range(l-self.K):
                subseq = seq[j:j+self.K]
                Mat[i] += self.pre_index[subseq]
        return Mat


# X= Xtr0
# n = Xtr0.shape[0]
# k = 3
#
