# -*- coding: utf-8 -*-


import numpy as np
from utils import evaluateCV

class HPOptimizer():

    def __init__(self, model, K, n_folds=5, bounds = None):
        if not(bounds):
            raise RuntimeError("HPOptimizer cannot be used with models without hyperparameter")
        np.warnings.filterwarnings('ignore')
        self.model = model
        self.K = K
        self.n_folds = n_folds
        self.bounds = np.array([bounds])
        self.hp_list = []
        self.score_list = []
        self.best_hp = None
        self.best_score = None

    def __repr__(self):
        return """
        Number of HP explored : {}
        Score function : {}
        Number of CV folds : {}
        Best HP : {}
        Best score : {}
        Tested HP : {}
        Corresponding scores : {}
        """.format(len(self.score_list), self.score_function, self.n_folds, self.best_hp,
                        self.best_score, self.hp_list, self.score_list)

    def evaluate_hp(self, hp):
        self.model.set_hyperparameters(10**hp[0])
        print("Current hp tested : {:.3e}".format(10**hp[0]))
        score = evaluateCV(self.model, self.K, n_folds = self.n_folds, verbose = False)[1]
        return score

    def explore(self, n_iters=64, method='RandomSearch', n_pre_samples=1, alpha=1e-10):
        bounds = self.bounds
        if method == "RandomSearch":
            hp_list = np.random.uniform(
                bounds[:, 0], bounds[:, 1], size=(n_iters, bounds.shape[0]))
            hp_list = hp_list.tolist()
            self.hp_list += hp_list
            for i, hp in enumerate(hp_list):
                print("Iteration : {}/{}".format(i+1, n_iters))
                score = self.evaluate_hp(hp)
                self.score_list.append(score)
                print("Score : {:.3f}".format(score))

        elif method == "GridSearch":
            n_grid = np.ceil(n_iters**(1/bounds.shape[0]))
            l = [np.linspace(bounds[i, 0], bounds[i, 1], n_grid)
                 for i in range(bounds.shape[0])]
            hp_list = self.cartesian_product(*l)
            hp_list = hp_list.tolist()
            self.hp_list += hp_list
            for i, hp in enumerate(hp_list):
                print("Iteration : {}/{}".format(i+1, n_iters))
                score = self.evaluate_hp(hp)
                self.score_list.append(score)
                print("Score : {:.3f}".format(score))

        max_i = np.argmax(self.score_list)
        self.best_hp = self.hp_list[max_i]
        self.best_score = self.score_list[max_i]
        return self.hp_list[-n_iters:], self.score_list[-n_iters:]

    def get_optimized_model(self):
        self.model.set_hyperparameters(self.best_hp)
        self.model.fit(self.X, self.y)
        return self.model

    def cartesian_product(self,*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)