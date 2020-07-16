from shapley.measures import Measure

class TMC_Shapley(Measure):

    def __init__(self, num_train=1000, num_test=100):
        self.name = 'TMC_Shapley'

    def one_iteration(self, tolerance, sources=None):
        """Runs one iteration of TMC-Shapley algorithm."""
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

    def score(self, X_train, y_train, X_test, y_test, model_family='', model=None, iterations=10, tolerance=0.1, sources=None):
        """Runs TMC-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance. (ratio with respect to average performance.)
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """       

        self.model = model

        marginals, idxs = [], []
        for iteration in range(iterations):
            if 10*(iteration+1)/iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(iteration + 1, iterations))
            marginals, idxs = self.one_iteration(tolerance=tolerance, sources=sources)
            self.mem_tmc = np.concatenate([self.mem_tmc, np.reshape(marginals, (1,-1))])
            self.idxs_tmc = np.concatenate([self.idxs_tmc, np.reshape(idxs, (1,-1))])