from abc import ABC, abstractmethod
import os
import numpy as np



class Kernel(ABC):

    @abstractmethod
    def __init__(self):
        self.Graam = None
        self.Embedding_test = None

    @abstractmethod
    def compute_Graam(self, X):
        """Computes the Graam Matrix associated with X and stores it in self.Graam. It should store the X data in the class (or the information needed to compute the embedding for the test set)
        """
        pass

    @abstractmethod
    def compute_embedding(self, Xtest):
        """
        Computes the kernel product betwee the vectors from the training set and the vectors from the test set and stores it.
        """
        pass

    def save_matrices(self, filename):
        if not os.path.exists("graam_matrices/"):
            os.mkdir("graam_matrices/")
        filepath = "graam_matrices/" + filename
        if self.Graam is None:
            print("Matrice de Graam non calculée")
        if self.Embedding_test is None :
            print("Embedding du test non calculé")
        if not os.path.exists(filepath + 'GR' ):
            if not os.path.exists(filepath + 'EM'):
                np.save(filepath + 'GR', self.Graam)
                np.save(filepath + 'EM', self.Embedding_test)
                return True
            else :
                print(filepath + 'EM' +  " already exists" )
                return False
        else:
            print(filepath + 'GR' +  " already exists" )
            return False
