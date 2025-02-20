# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:46:31 2019

@author: Clément Barras, CentraleSupélec
"""

import numpy as np
import pandas as pd
import os
from numba import jit
import time


class MismatchTree():
    """
    Tree based data structure used to calculate mismatch Gram matrices, according Leslie and al. 2004.
    See ./references/mismatch_kernel.pdf for more information.

    """

    def __init__(self, k, m, label="", children={}, depth=0, alphabet="ATCG"):
        """Class constructor, also used for creating new nodes during recursion

        Parameters
        ----------
        k : int, sequences length
        m : int, mismatch parameter
        label : str, optional
            Root label, used for initialization during recursion
        children : dict, optional
            Dictionary of children ({root_label : child})
        depth : int, optional
            Depth of the node, used for initialization during recursion
        alphabet : str, optional
            Set of possible letters in the data

        """
        self.m = m
        self.k = k
        self.label = label
        self.children = children
        self.depth = depth
        self.list_pointers = []
        self.alphabet = alphabet

    def add_sequence(self, seq, i=0, j=0):
        """Add a sequence (usually of k characters) in the tree

        Parameters
        ----------
        seq : str
        i : int, optional, denotes the row in the matrix used to fit the tree.
        j : int, optional, denotes the column in the matrix used to fit the tree.

        """
        self._aux_add_sequence(seq[0], seq[1:], (i, j, 0))
        return self

    def _aux_add_sequence(self, char, suffix, pointer):
        """Auxiliary function used for recursion"""
        (i, j, mu) = pointer
        self.list_pointers.append((i, j, mu))
        self = self.add_char(char)
        for letter in self.alphabet:
            if letter == char:
                self.add_char(letter)
                if suffix:
                    self = self.update_children(self.children[letter]._aux_add_sequence(
                        suffix[0], suffix[1:], (i, j, mu)))
                else:
                    self.children[letter].list_pointers.append((i, j, mu))
            else:
                if mu < self.m:
                    self.add_char(letter)
                    if suffix:
                        self = self.update_children(self.children[letter]._aux_add_sequence(
                            suffix[0], suffix[1:], (i, j, mu + 1)))
                    else:
                        self.children[letter].list_pointers.append(
                            (i, j, mu + 1))
        return self

    def update_children(self, child_tree):
        self.children[child_tree.label] = child_tree
        return self

    def add_char(self, char):
        if char not in self.children.keys():
            self.children[char] = MismatchTree(
                self.k, self.m, char, {}, self.depth + 1)
        return self

    def fit(self, X):
        """
        Fit the tree on the data X. Calls fit_row for all rows.
        """
        self.n = X.shape[0]
        for i in range(X.shape[0]):
            self.fit_row(X[i], i)

    def fit_row(self, row, i):
        """
        Fit the tree on one row of data. Slices the data into sequence of k characters.
        """
        p = len(row)
        if not (i % 1000):
            print("Processing the row : {}".format(i))
        for j in range(p - self.k):
            seq = str(row[j:(j + self.k)])
            self.add_sequence(seq, i, j)

    # @jit(nopython = True)
    def compute_gram_matrix(self):
        leaves_list = self._get_leaves_list()
        self.embeddings = np.array(leaves_list, dtype=np.float32)[:, :, 0]
        # print("Embedding space dimension : {}".format(len(leaves_list)))#.format(self.embeddings.shape[0]))
        self.gram_matrix = self.embeddings.T @ self.embeddings
        #        self.gram_matrix = np.zeros((len(leaves_list[0]), len(leaves_list[0])), dtype=np.float32)
        #        for i, ind in enumerate(leaves_list):
        #            if not(i%1000):
        #                print("{}/{}".format(i, len(leaves_list)))
        #            self.gram_matrix += ind@ind.T
        return self.gram_matrix

    def _get_leaves_list(self):
        return self._aux_get_leaves_indicator(self.n)

    def _aux_get_leaves_indicator(self, n):
        if self.depth == self.k:
            indicator = np.zeros((n, 1), dtype=np.float32)
            for elem in self.list_pointers:
                indicator[elem[0]] += 1
            return [indicator]
        else:
            temp_list = []
            for child in self.children.values():
                temp_list += child._aux_get_leaves_indicator(n)
            return temp_list


def precalc_gram(path, k_list, m_list, sets, data_path = "./data/"):
    """Calculate mismatch gram matrices (k, m) for set s along the cartesian
    product k_list*m_lists*sets"""
    try:
        # Create target Directory
        os.makedirs(path)
        print("Directory ", path, " created ")
    except FileExistsError:
        print("Directory ", path, " already exists")
    for s in sets:
        Xtrain = pd.read_csv(data_path + "Xtr{}.csv".format(s)).values[:, 1]
        Xtest = pd.read_csv(data_path + "Xte{}.csv".format(s)).values[:, 1]
        X = np.concatenate((Xtrain, Xtest), axis=0)
        for k in k_list:
            for m in m_list:
                print(f"\nGenerating the mismatch Gram matrix (k:{k}, m:{m})for dataset {s}")
                t0 = time.time()
                tree = MismatchTree(k, m, label="", children={}, depth=0, alphabet="ATCG")
                tree.fit(X)
                print("Mismatch tree built in {:.3f}s ".format(time.time() - t0))
                t0 = time.time()
                name = "mismatch{}k@{}m@{}".format(s, k, m)
                G = tree.compute_gram_matrix()
                print("Gram matrix calculated in {:.3f}s ".format(time.time() - t0))
                np.savez(os.path.join(path, name), G)
                del (tree)

