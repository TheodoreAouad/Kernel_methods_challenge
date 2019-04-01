import numpy as np
import kernel
import pickle
import os
from scipy.spatial.distance import pdist


class spectrum(kernel.Kernel):

    def __init__(self, k=3):
        super().__init__()
        self.K = k
        self.N = None
        self.pre_index = None
        self.Graam_diff_squared = None

    def __call__(self, x, y):
        '''Computes the dot product of the embeddings of x and y.
        Input: x,y two strings.
        Output: k(x,y)'''
        pre_index = {}
        res = 0
        for j in range(len(x) - self.K):
            subseq = x[j:j + self.K]
            if subseq in pre_index.keys():
                pre_index[subseq] += 1
            else:
                pre_index[subseq] = 1
        for j in range(len(y) - self.K):
            subseq = y[j:j + self.K]
            res += pre_index.get(subseq, 0)
        return res

    def __call__(self, x, y):
        '''Computes the dot product of the embeddings of x and y.
        Input: x,y two strings.
        Output: k(x,y)'''
        pre_index = {}
        res = 0
        for j in range(len(x) - self.K):
            subseq = x[j:j + self.K]
            if subseq in pre_index.keys():
                pre_index[subseq] += 1
            else:
                pre_index[subseq] = 1
        for j in range(len(y) - self.K):
            subseq = y[j:j + self.K]
            res += pre_index.get(subseq, 0)
        return res

    def compute_pre_index(self, X):
        self.N = X.shape[0]
        self.Graam = np.zeros((self.N, self.N))
        pre_index = {}
        for i in range(self.N):
            seq = X[i]
            l = len(seq)
            for j in range(l - self.K):
                subseq = seq[j:j + self.K]
                if subseq in pre_index.keys():
                    pre_index[subseq][i] += 1
                else:
                    new = np.zeros(self.N)
                    new[i] = 1
                    pre_index[subseq] = new
        self.pre_index = pre_index

    def compute_Graam(self, X, recompute_index=True):
        '''Compute the Gram Matrix of the training data.
        Input: training data X, array of strings
        Output: M[i,j] = K(train_i,train_j)'''
        if recompute_index:
            self.compute_pre_index(X)
        for sub_seq in self.pre_index:
            V = self.pre_index[sub_seq]
            self.Graam += np.outer(V, V)
        return self.Graam

    def compute_embedding(self, Xtest):
        '''Computes the kernel products between train and test.
        Input: Array of strings.
        Output: Matrix where M[i,j] = K(test_i,train_j)
        '''
        n = Xtest.shape[0]
        Mat = np.zeros((n, self.N))
        for i in range(n):
            seq = Xtest[i]
            l = len(seq)
            for j in range(l - self.K):
                subseq = seq[j:j + self.K]
                Mat[i] += self.pre_index.get(subseq, 0)
        self.Embedding_test = Mat
        return Mat

    def save_index(self, filename):
        if not os.path.exists("additionnal_kernel_info/"):
            os.mkdir("additionnal_kernel_info/")
        filepath = "additionnal_kernel_info/" + filename + 'pre_index'
        if self.pre_index is None:
            print("Pré_index non calculé")
        if not os.path.exists(filepath):
            with open(filepath, 'wb') as handle:
                pickle.dump(self.pre_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        else:
            print(filepath + 'pre_index' + " already exists")
            return False
