from shapley.measures import Measure
import numpy as np
import copy
import scipy

class LOO(Measure):

    def __init__(self):
        self.name = 'LOO'

    def score(self, X_train, y_train, X_test, y_test, model_family='', model=None):
        """Calculated leave-one-out values for the given metric.

        Args:
            metric: If None, it will use the objects default metric.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.

        Returns:
            Leave-one-out scores
        """

        sources = {i:np.array([i]) for i in range(X_train.shape[0])}
        self.restart_model(X_train, y_train, model)
        model.fit(X_train, y_train)
        baseline_value = model.score(X_train, y_train)
        vals_loo = np.zeros(X_train.shape[0])
        for i in sources.keys():
            if isinstance(X_train, scipy.sparse.csr_matrix):
                X_batch = delete_rows_csr(X_train, sources[i])
            else:
                X_batch = np.delete(X_train, sources[i], axis=0)
            y_batch = np.delete(y_train, sources[i], axis=0)
            model.fit(X_batch, y_batch)
            removed_value = model.score(X_train, y_train)
            vals_loo[sources[i]] = (baseline_value - removed_value)/len(sources[i])
        return vals_loo
