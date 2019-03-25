import numpy as np

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
