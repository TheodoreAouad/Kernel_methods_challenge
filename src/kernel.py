import numpy as np
import utils as u


class Kclasse():

  def __init__(self, s, k, m, center = True, gaussian = None, noramlize = True)
    gram_path = "./gram_matrices/mismatch/mismatch{}k@{}m@{}.npz".format(s, k, m)
    self.labels = pd.read_csv("./data/Ytr{}.csv".format(s)).values[:, 1]*2-1
    self.gram_matrix = np.load(graam_path)['arr_0']
    self.
    #Gaussian combinaison
    if gaussian is not None :
        self.gram_matrix = u.compute_squared_distance(self.gram_matrix)
        self.gram_matrix = np.exp(-1/(2*gaussian) * self.gram_matrix)
    #Centering the data in the embedding space if required
    if center :
        self.gram_matrix = u.center_graam_matrix(self.gram_matrix)
        self.gram_matrix = np.exp(-1/(2*gaussian) * self.gram_matrix)
    if normalize :
        self.gram_matrix = self.gram_matrix / np.mean(np.diag(self.gram_matrix))

    def get_train(indxs):
        return (self.gram_matrix[indices][:, indices], self.labels[indxs])

    def get_valid(train_idxs, test_idxs):
        return (self.gram_matrix[test_idxs][:, train_idxs], self.labels[indxs])

    def get_train2000():
        """
        Returns the Graam matrix corresponding to the whole train test
        and the corresponding labels (in a tuple)
        """
        return (self.gram_matrix[:2000, :2000], self.labels)

    def get_test2000():
        """
        Returns the kernel products between the whole train set and the datapoints to predict for the kernel
        """
        return (self.gram_matrix[2000:,:2000], None)
