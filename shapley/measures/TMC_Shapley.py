from shapley.measures import Measure
import numpy as np
import scipy
import warnings

class TMC_Shapley(Measure):

    def __init__(self):
        self.name = 'TMC_Shapley'

    def one_iteration(self, X_train, y_train, X_test, y_test, model_family, model, iterations, tolerance, sources, mean_score):
        """Runs one iteration of TMC-Shapley algorithm."""
        idxs, marginal_contribs = np.random.permutation(len(sources.keys())), np.zeros(X_train.shape[0])
        new_score = np.max(np.bincount(y_test).astype(float)/len(y_test))
        X_batch, y_batch = np.zeros((0,) +  tuple(X_train.shape[1:])), np.zeros(0).astype(int)
        truncation_counter = 0

        for n, idx in enumerate(idxs):
            old_score = new_score
            if isinstance(X_train, scipy.sparse.csr_matrix):
                X_batch = scipy.sparse.vstack([X_batch, X_train[sources[idx]]])
            else:
                X_batch = np.concatenate((X_batch, X_train[sources[idx]]))
            y_batch = np.concatenate([y_batch, y_train[sources[idx]]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                is_regression = (np.mean(y_train//1==y_train) != 1)
                is_regression = is_regression or isinstance(y_train[0], np.float32)
                is_regression = is_regression or isinstance(y_train[0], np.float64)
                if is_regression or len(set(y_batch)) == len(set(y_test)):
                    self.restart_model(X_train, y_train, model)
                    model.fit(X_batch, y_batch)
                    new_score = model.score(X_test, y_test)      
            marginal_contribs[sources[idx]] = (new_score - old_score) / len(sources[idx])

            if np.abs(new_score - mean_score) <= tolerance * mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs

    def score(self, X_train, y_train, X_test, y_test, model_family='', model=None, iterations=10, tolerance=0.1):

        sources = {i:np.array([i]) for i in range(X_train.shape[0])}
        mem_tmc = np.zeros((0, X_train.shape[0]))
        idxs_shape = (0, len(sources.keys()))
        idxs_tmc = np.zeros(idxs_shape).astype(int)

        marginals, idxs = [], []

        scores = []
        self.restart_model(X_train, y_train, model)
        for _ in range(1):
            model.fit(X_train, y_train)
            for _ in range(100):
                bag_idxs = np.random.choice(len(y_test), len(y_test))
                scores.append(model.score(X_test[bag_idxs], y_test[bag_idxs]))
        mean_score = np.mean(scores)

        for iteration in range(iterations):
            if 10*(iteration+1)/iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(iteration + 1, iterations))
            marginals, idxs = self.one_iteration(X_train, y_train, X_test, y_test, model_family, model, iterations, tolerance, sources, mean_score)
            mem_tmc = np.concatenate([mem_tmc, np.reshape(marginals, (1,-1))])
            idxs_tmc = np.concatenate([idxs_tmc, np.reshape(idxs, (1,-1))])
        return np.mean(mem_tmc, 0)