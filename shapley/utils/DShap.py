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
from shapley.measures import KNN_Shapley, TMC_Shapley, G_Shapley, LOO, KNN_LOO

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
                 seed=None, nodump=False, **kwargs):
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
        if len(set(self.y)) > 2:
            assert self.metric != 'f1' and self.metric != 'auc', 'Invalid metric!'
        is_regression = (np.mean(self.y//1==self.y) != 1)
        is_regression = is_regression or isinstance(self.y[0], np.float32)
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)
        self.model = return_model(self.model_family, **kwargs)
        self.random_score = self.init_score(self.metric)
            
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
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        self.vals_loo = None
        if os.path.exists(loo_dir):
            self.vals_loo = pkl.load(open(loo_dir, 'rb'), encoding='iso-8859-1')['loo']
        previous_results =  os.listdir(self.directory)
        tmc_numbers = [int(name.split('.')[-2].split('_')[-1])
                      for name in previous_results if 'mem_tmc' in name]
        g_numbers = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_g' in name]
        self.tmc_number = str(0) if len(g_numbers)==0 else str(np.max(tmc_numbers) + 1)
        self.g_number = str(0) if len(g_numbers)==0 else str(np.max(g_numbers) + 1)
        tmc_dir = os.path.join(self.directory, 'tmc.pkl')
        g_dir = os.path.join(self.directory, 'g.pkl')
        self.mem_tmc, self.mem_g = [np.zeros((0, self.X.shape[0])) for _ in range(2)]
        idxs_shape = (0, self.X.shape[0] if self.sources is None else len(self.sources.keys()))
        self.idxs_tmc, self.idxs_g = [np.zeros(idxs_shape).astype(int) for _ in range(2)]
        self.vals_tmc = np.zeros((self.X.shape[0],))
        self.vals_g = np.zeros((self.X.shape[0],))
        self.vals_inf = np.zeros((self.X.shape[0],))
        if self.nodump == False:
            pkl.dump(self.vals_tmc, open(tmc_dir, 'wb'))
        if self.model_family not in ['logistic', 'NN']:
            return
        if self.nodump == False:
            pkl.dump(self.vals_g, open(g_dir, 'wb'))
                
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
        if self.measure == "KNN_Shapley":
            return self._knn_shap(K=10)

        elif self.measure == "KNN_LOO":
            return self._loo_knn_shap(K=10)

        elif self.measure == "LOO":
            try:
                len(self.vals_loo)
            except:
                self.vals_loo = self._calculate_loo_vals(sources=self.sources)
            return self.vals_loo
        elif self.measure == "TMC_Shapley":
            tmc_run = True
            if error(self.mem_tmc) < err:
                tmc_run = False
            else:
                self._tmc_shap(save_every, tolerance=tolerance, sources=self.sources)
                self.vals_tmc = np.mean(self.mem_tmc, 0)
            return self.vals_tmc

        elif self.measure == "G_Shapley":
            g_run = True
            if self.model_family not in ['logistic', 'NN']:
                print("Model unsatisfied for G-Shapley!")
                g_run = False
            if error(self.mem_g) < err:
                g_run = False
            else:
                self._g_shap(save_every, sources=self.sources)
                self.vals_g = np.mean(self.mem_g, 0)
            return self.vals_g

        else:
            print("Unknown measure!")


    def _loo_knn_shap(self, K=5):
        N = self.X.shape[0]
        M = self.X_test.shape[0]

        if self.model_family == "ResNet":
            resnet = self.model
            X_out1 = resnet.layer1(F.relu(resnet.bn1(resnet.conv1(torch.from_numpy(self.X)))))
            X_test_out1 = resnet.layer1(F.relu(resnet.bn1(resnet.conv1(torch.from_numpy(self.X_test))))) # 64, 32, 32
            X_out2 = resnet.layer2(X_out1)
            X_test_out2 = resnet.layer2(X_test_out1) # 64, 32, 32
            X_out3 = resnet.layer3(X_out2)
            X_test_out3 = resnet.layer3(X_test_out2) # 64, 32, 32
            value = np.zeros(N)
            for i in range(M):
                X = X_test_out1[i]
                y = self.y_test[i]
                s = np.zeros(N)
                dist = []
                diff = (X_out1.detach().numpy() - X.detach().numpy()).reshape(N, -1)
                dist = np.einsum('ij, ij->i', diff, diff)
                idx = np.argsort(dist)
                ans = self.y[idx]
                s[idx[N - 1]] = float(ans[N - 1] == y) / N
                cur = N - 2
                for j in range(N):
                    if j in idx[:K]:
                        s[j] = float(int(ans[j] == y) - int(ans[K] == y)) / K
                    else:
                        s[j] = 0
                for i in range(N):
                    value[j] += s[j]  
            for i in range(N):
                value[i] /= M
            return value

        if self.model_family == "NN":
            nn = self.model
            X_feature = ACTIVATIONS['relu'](np.matmul(self.X, nn.coefs_[0]) + nn.intercepts_[0])
            X_test_feature = ACTIVATIONS['relu'](np.matmul(self.X_test, nn.coefs_[0]) + nn.intercepts_[0])
            value = np.zeros(N)
            for i in range(M):
                X = X_test_feature[i]
                y = self.y_test[i]
                s = np.zeros(N)
                dist = []
                diff = (X_feature - X).reshape(N, -1)
                dist = np.einsum('ij, ij->i', diff, diff)
                idx = np.argsort(dist)
                ans = self.y[idx]
                s[idx[N - 1]] = float(ans[N - 1] == y) / N
                cur = N - 2
                for j in range(N):
                    if j in idx[:K]:
                        s[j] = float(int(ans[j] == y) - int(ans[K] == y)) / K
                    else:
                        s[j] = 0
                for i in range(N):
                    value[j] += s[j]  
            for i in range(N):
                value[i] /= M  
            return value

        value = np.zeros(N)
        for i in range(M):
            X = self.X_test[i]
            y = self.y_test[i]

            s = np.zeros(N)
            dist = []
            diff = (self.X - X).reshape(N, -1)
            dist = np.einsum('ij, ij->i', diff, diff)
            idx = np.argsort(dist)

            ans = self.y[idx]

            s[idx[N - 1]] = float(ans[N - 1] == y) / N

            cur = N - 2
            for j in range(N):
                if j in idx[:K]:
                    s[j] = float(int(ans[j] == y) - int(ans[K] == y)) / K
                else:
                    s[j] = 0
                
            for i in range(N):
                value[j] += s[j]  

        for i in range(N):
            value[i] /= M
        return value
    
    def _knn_shap(self, K=5):
        N = self.X.shape[0]
        M = self.X_test.shape[0]

        if self.model_family == "ResNet":
            resnet = self.model
            X_out1 = resnet.layer1(F.relu(resnet.bn1(resnet.conv1(torch.from_numpy(self.X)))))
            X_test_out1 = resnet.layer1(F.relu(resnet.bn1(resnet.conv1(torch.from_numpy(self.X_test))))) # 64, 32, 32
            X_out2 = resnet.layer2(X_out1)
            X_test_out2 = resnet.layer2(X_test_out1) # 64, 32, 32
            X_out3 = resnet.layer3(X_out2)
            X_test_out3 = resnet.layer3(X_test_out2) # 64, 32, 32
            s = np.zeros((N, M))
            for i in range(M):
                X = X_test_out1[i]
                y = self.y_test[i]
                dist = []
                diff = (X_out1.detach().numpy() - X.detach().numpy()).reshape(N, -1)
                dist = np.einsum('ij, ij->i', diff, diff)
                idx = np.argsort(dist)
                ans = self.y[idx]
                s[idx[N - 1]][i] = float(ans[N - 1] == y) / N
                cur = N - 2
                for j in range(N - 1):
                    s[idx[cur]][i] = s[idx[cur + 1]][i] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / K * (min(cur, K - 1) + 1) / (cur + 1)
                    cur -= 1
            return np.mean(s, axis=1)

        if self.model_family == "NN":
            nn = self.model
            X_feature = ACTIVATIONS['relu'](np.matmul(self.X, nn.coefs_[0]) + nn.intercepts_[0])
            X_test_feature = ACTIVATIONS['relu'](np.matmul(self.X_test, nn.coefs_[0]) + nn.intercepts_[0])
            s = np.zeros((N, M))
            for i in range(M):
                X = X_test_feature[i]
                y = self.y_test[i]
                dist = []
                diff = (X_feature - X).reshape(N, -1)
                dist = np.einsum('ij, ij->i', diff, diff)
                idx = np.argsort(dist)
                ans = self.y[idx]
                s[idx[N - 1]][i] = float(ans[N - 1] == y) / N
                cur = N - 2
                for j in range(N - 1):
                    s[idx[cur]][i] = s[idx[cur + 1]][i] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / K * (min(cur, K - 1) + 1) / (cur + 1)
                    cur -= 1  
            return np.mean(s, axis=1)

        value = np.zeros(N)
        s = np.zeros((N, M))
        for i in range(M):
            X = self.X_test[i]
            y = self.y_test[i]

            dist = []
            diff = (self.X - X).reshape(N, -1)
            dist = np.einsum('ij, ij->i', diff, diff)
            idx = np.argsort(dist)

            ans = self.y[idx]

            s[idx[N - 1]][i] = float(ans[N - 1] == y) / N

            cur = N - 2
            for j in range(N - 1):
                s[idx[cur]][i] = s[idx[cur + 1]][i] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / K * (min(cur, K - 1) + 1) / (cur + 1)
                cur -= 1 

            return np.mean(s, axis=1)
        
    def _tmc_shap(self, iterations, tolerance=None, sources=None):
        """Runs TMC-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance. (ratio with respect to average performance.)
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(self.X.shape[0])}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        model = self.model
        try:
            self.mean_score
        except:
            self._tol_mean_score()
        if tolerance is None:
            tolerance = self.tolerance         
        marginals, idxs = [], []
        for iteration in range(iterations):
            if 10*(iteration+1)/iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(iteration + 1, iterations))
            marginals, idxs = self.one_iteration(tolerance=tolerance, sources=sources)
            self.mem_tmc = np.concatenate([self.mem_tmc, np.reshape(marginals, (1,-1))])
            self.idxs_tmc = np.concatenate([self.idxs_tmc, np.reshape(idxs, (1,-1))])
        
    def _tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        self.restart_model()
        for _ in range(1):
            self.model.fit(self.X, self.y)
            for _ in range(100):
                bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))
                scores.append(self.value(self.model, metric=self.metric,
                                         X=self.X_test[bag_idxs], y=self.y_test[bag_idxs]))
        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)
        
    def one_iteration(self, tolerance, sources=None):
        """Runs one iteration of TMC-Shapley algorithm."""
        if sources is None:
            sources = {i:np.array([i]) for i in range(self.X.shape[0])}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        idxs, marginal_contribs = np.random.permutation(len(sources.keys())), np.zeros(self.X.shape[0])
        new_score = self.random_score
        X_batch, y_batch = np.zeros((0,) +  tuple(self.X.shape[1:])), np.zeros(0).astype(int)
        truncation_counter = 0
        for n, idx in enumerate(idxs):
            old_score = new_score
            if isinstance(self.X, scipy.sparse.csr_matrix):
                X_batch = scipy.sparse.vstack([X_batch, self.X[sources[idx]]])
            else:
                X_batch = np.concatenate((X_batch, self.X[sources[idx]]))
            y_batch = np.concatenate([y_batch, self.y[sources[idx]]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.is_regression or len(set(y_batch)) == len(set(self.y_test)): ##FIXIT
                    self.restart_model()
                    self.model.fit(X_batch, y_batch)
                    new_score = self.value(self.model, metric=self.metric)       
            marginal_contribs[sources[idx]] = (new_score - old_score) / len(sources[idx])
            if np.abs(new_score - self.mean_score) <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs
    
    def restart_model(self):
        
        try:
            self.model = copy.deepcopy(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)
        
    def _one_step_lr(self):
        """Computes the best learning rate for G-Shapley algorithm."""
        if self.directory is None:
            address = None
        else:
            address = os.path.join(self.directory, 'weights')
        best_acc = 0.0
        for i in np.arange(1, 5, 0.5):
            model = ShapNN(
                self.problem, batch_size=1, max_epochs=1, 
                learning_rate=10**(-i), weight_decay=0., 
                validation_fraction=0, optimizer='sgd', warm_start=False,
                address=address, hidden_units=self.hidden_units)
            accs = []
            for _ in range(10):
                model.fit(np.zeros((0, self.X.shape[-1])), self.y)
                model.fit(self.X, self.y)
                accs.append(model.score(self.X_test, self.y_test))
            if np.mean(accs) - np.std(accs) > best_acc:
                best_acc  = np.mean(accs) - np.std(accs)
                learning_rate = 10**(-i)
        return learning_rate
    
    def _g_shap(self, iterations, err=None, learning_rate=None, sources=None):
        """Method for running G-Shapley algorithm.
        
        Args:
            iterations: Number of iterations of the algorithm.
            err: Stopping error criteria
            learning_rate: Learning rate used for the algorithm. If None
                calculates the best learning rate.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(self.X.shape[0])}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        address = None
        if self.directory is not None:
            address = os.path.join(self.directory, 'weights')
        if learning_rate is None:
            try:
                learning_rate = self.g_shap_lr
            except AttributeError:
                self.g_shap_lr = self._one_step_lr()
                learning_rate = self.g_shap_lr
        model = ShapNN(self.problem, batch_size=1, max_epochs=1,
                     learning_rate=learning_rate, weight_decay=0.,
                     validation_fraction=0, optimizer='sgd',
                     address=address, hidden_units=self.hidden_units)
        for iteration in range(iterations):
            model.fit(np.zeros((0, self.X.shape[-1])), self.y)
            if 10 * (iteration+1) / iterations % 1 == 0:
                print('{} out of {} G-Shapley iterations'.format(iteration + 1, iterations))
            marginal_contribs = np.zeros(len(sources.keys()))
            model.fit(self.X, self.y, self.X_test, self.y_test, sources=sources,
                      metric=self.metric, max_epochs=1, batch_size=1)
            val_result = model.history['metrics']
            marginal_contribs[1:] += val_result[0][1:]
            marginal_contribs[1:] -= val_result[0][:-1]
            individual_contribs = np.zeros(self.X.shape[0])
            for i, index in enumerate(model.history['idxs'][0]):
                individual_contribs[sources[index]] += marginal_contribs[i]
                individual_contribs[sources[index]] /= len(sources[index])
            self.mem_g = np.concatenate(
                [self.mem_g, np.reshape(individual_contribs, (1,-1))])
            self.idxs_g = np.concatenate(
                [self.idxs_g, np.reshape(model.history['idxs'][0], (1,-1))])
    
    def _calculate_loo_vals(self, sources=None, metric=None):
        """Calculated leave-one-out values for the given metric.
        
        Args:
            metric: If None, it will use the objects default metric.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        
        Returns:
            Leave-one-out scores
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(self.X.shape[0])}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        print('Starting LOO score calculations!')
        if metric is None:
            metric = self.metric 
        self.restart_model()
        self.model.fit(self.X, self.y)
        baseline_value = self.value(self.model, metric=metric)
        vals_loo = np.zeros(self.X.shape[0])
        for i in sources.keys():
            if isinstance(self.X, scipy.sparse.csr_matrix):
                X_batch = delete_rows_csr(self.X, sources[i])
            else:
                X_batch = np.delete(self.X, sources[i], axis=0)
            y_batch = np.delete(self.y, sources[i], axis=0)
            self.model.fit(X_batch, y_batch)
            removed_value = self.value(self.model, metric=metric)
            vals_loo[sources[i]] = (baseline_value - removed_value)/len(sources[i])
        return vals_loo