import copy
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import tensorflow as tf
import sys
import shutil
import matplotlib.pyplot as plt
import warnings
import itertools
import pickle as pkl
import scipy
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F

from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network._base import ACTIVATIONS

from shapley.utils.shap_utils import *
from shapley.utils.Shapley import ShapNN

def delete_rows_csr(mat, index):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[index] = False
    return mat[mask]

class DShap(object):

    def __init__(self, X, y, X_test, y_test, num_test, sources=None, directory="./",
                 problem='classification', model_family='logistic', metric='accuracy', measure=None,
                 seed=None, nodump=True, **kwargs):
        """
        Args:
            X: Data covariates
            y: Data labels
            X_test: Test+Held-out covariates
            y_test: Test+Held-out labels
            sources: An array or dictionary assiging each point to its group.
                If None, evey points gets its individual value.
            num_test: Number of data points used for evaluation metric.
            directory: Directory to save results and figures.
            problem: "Classification" or "Regression"(Not implemented yet.)
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting
                same permutations.
            **kwargs: Arguments of the model
        """

        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
        self.problem = problem
        self.model_family = model_family
        self.metric = metric
        self.directory = directory
        self.hidden_units = kwargs.get('hidden_layer_sizes', [])
        self.nodump = nodump
        if self.model_family is 'logistic':
            self.hidden_units = []
        if self.directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)
                os.makedirs(os.path.join(directory, 'weights'))
                os.makedirs(os.path.join(directory, 'plots'))
            self._initialize_instance(X, y, X_test, y_test, num_test, sources)
        if np.max(self.y) + 1 > 2:
            assert self.metric != 'f1' and self.metric != 'auc', 'Invalid metric!'
        is_regression = (np.mean(self.y//1==self.y) != 1)
        is_regression = is_regression or isinstance(self.y[0], np.float32)
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)
        self.model = return_model(self.model_family, **kwargs)
        self.random_score = self.init_score(self.metric)
        self.measure = measure

    def _initialize_instance(self, X, y, X_test, y_test, num_test, sources=None):
        """Loads or creates data."""

        if sources is None:
            sources = {i:np.array([i]) for i in range(X.shape[0])}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        # data_dir = os.path.join(self.directory, 'data.pkl')
        # if os.path.exists(data_dir):
        #     data_dic = pkl.load(open(data_dir, 'rb'), encoding='iso-8859-1')
        #     self.X_heldout, self.y_heldout = data_dic['X_heldout'], data_dic['y_heldout']
        #     self.X_test, self.y_test =data_dic['X_test'], data_dic['y_test']
        #     self.X, self.y = data_dic['X'], data_dic['y']
        #     self.sources = data_dic['sources']
        # else:
        self.X_heldout, self.y_heldout = X_test[:-num_test], y_test[:-num_test]
        self.X_test, self.y_test = X_test[-num_test:], y_test[-num_test:]
        self.X, self.y, self.sources = X, y, sources
            # if self.nodump == False:
            #     pkl.dump({'X': self.X, 'y': self.y, 'X_test': self.X_test,
            #          'y_test': self.y_test, 'X_heldout': self.X_heldout,
            #          'y_heldout':self.y_heldout, 'sources': self.sources},
            #          open(data_dir, 'wb'))

    def init_score(self, metric):
        """ Gives the value of an initial untrained model."""
        if metric == 'accuracy':
            return np.max(np.bincount(self.y_test).astype(float)/len(self.y_test))
        if metric == 'f1':
            return np.mean([f1_score(
                self.y_test, np.random.permutation(self.y_test)) for _ in range(1000)])
        if metric == 'auc':
            return 0.5
        random_scores = []
        for _ in range(100):
            self.model.fit(self.X, np.random.permutation(self.y))
            random_scores.append(self.value(self.model, metric))
        return np.mean(random_scores)

    def value(self, model, metric=None, X=None, y=None):
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            X: Covariates, valuation is performed on a data different from test set.
            y: Labels, if valuation is performed on a data different from test set.
            """
        if metric is None:
            metric = self.metric
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if metric == 'accuracy':
            return model.score(X, y)
        if metric == 'f1':
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'
            return f1_score(y, model.predict(X))
        if metric == 'auc':
            assert len(set(y)) == 2, 'Data has to be binary for auc metric.'
            return my_auc_score(model, X, y)
        if metric == 'xe':
            return my_xe_score(model, X, y)
        raise ValueError('Invalid metric!')

    def run(self, save_every, err, tolerance=0.01, knn_run=True, tmc_run=True, g_run=True, loo_run=True):
        """Calculates data sources(points) values.

        Args:
            save_every: save marginal contrivbutions every n iterations.
            err: stopping criteria for each of TMC-Shapley or G-Shapley algorithm.
            tolerance: Truncation tolerance. If None, the instance computes its own.
            g_run: If True, computes G-Shapley values.
            loo_run: If True, computes and saves leave-one-out scores.
        """
        # tmc_run, g_run = tmc_run, g_run and self.model_family in ['logistic', 'NN']

        self.restart_model()
        self.model.fit(self.X, self.y)
        
        return self.measure.score(self.X, self.y, self.X_test, self.y_test, self.model_family, self.model)

    def restart_model(self):
        try:
            self.model = copy.deepcopy(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)
