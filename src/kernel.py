import numpy as np
import utils as u
import pandas as pd


class kernel():

    def __init__(self, s=0, k=3, m=0, center = True, gaussian = None, normalize = True, Graam_matrix=None):
        if Graam_matrix is not None :
            self.gram_matrix = Graam_matrix
        else :
            gram_path = "./gram_matrices/mismatch/mismatch{}k@{}m@{}.npz".format(s, k, m)
            self.labels = pd.read_csv("./data/Ytr{}.csv".format(s)).values[:, 1]*2-1
            self.gram_matrix = np.load(gram_path)['arr_0']
            #Gaussian combinaison
            if gaussian  :
                self.gram_matrix = u.compute_squared_distance(self.gram_matrix)
                self.gram_matrix = np.exp(-1/(2*gaussian) * self.gram_matrix)
            #Centering the data in the embedding space if required
            if center :
                self.gram_matrix = u.center_graam_matrix(self.gram_matrix)
            if normalize :
                self.gram_matrix = self.gram_matrix / np.mean(np.diag(self.gram_matrix))


    def get_train(self, indxs):
        return (self.gram_matrix[indxs][:, indxs], self.labels[indxs])

    def get_valid(self, train_idxs, test_idxs):
        return (self.gram_matrix[test_idxs][:, train_idxs], self.labels[test_idxs])

    def get_train2000(self):
        """
        Returns the Graam matrix corresponding to the whole train test
        and the corresponding labels (in a tuple)
        """
        return (self.gram_matrix[:2000, :2000], self.labels[:2000])

    def get_test2000(self):
        """
        Returns the kernel products between the whole train set and the datapoints to predict for the kernel
        """
        return (self.gram_matrix[2000:,:2000], None)
