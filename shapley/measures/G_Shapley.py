from shapley.measures import Measure
import numpy as np
import scipy
from shapley.utils.Shapley import ShapNN

class G_Shapley(Measure):

    def __init__(self):
        self.name = 'G_Shapley'


    def _one_step_lr(self, X_train, y_train, X_test, y_test, problem):
        best_acc = 0.0
        learning_rate = 0.001
        for i in np.arange(1, 5, 0.5):
            model = ShapNN(
                problem, batch_size=1, max_epochs=1, 
                learning_rate=10**(-i), weight_decay=0., 
                validation_fraction=0, optimizer='sgd', warm_start=False,
                address=None, hidden_units=[100])
            accs = []
            for _ in range(10):
                model.fit(np.zeros((0, X_train.shape[-1])), y_train)
                model.fit(X_train, y_train)
                accs.append(model.score(X_test, y_test))
            if np.mean(accs) - np.std(accs) > best_acc:
                best_acc  = np.mean(accs) - np.std(accs)
                learning_rate = 10**(-i)
        return learning_rate

    def score(self, X_train, y_train, X_test, y_test, model_family='', model=None, iterations=10, err=0.5):

        sources = {i:np.array([i]) for i in range(X_train.shape[0])}
        mem_g = np.zeros((0, X_train.shape[0]))
        idxs_shape = (0, len(sources.keys()))
        idxs_g = np.zeros(idxs_shape).astype(int)
        print(np.mean(y_train//1==y_train))
        is_regression = (np.mean(y_train//1==y_train) != 1)
        is_regression = is_regression or isinstance(y_train[0], np.float32)
        is_regression = is_regression or isinstance(y_train[0], np.float64)
        problem = 'regression' if is_regression else 'classification'
        print(problem)
        learning_rate = self._one_step_lr(X_train, y_train, X_test, y_test, problem)
        model = ShapNN(problem, batch_size=1, max_epochs=1,
                     learning_rate=learning_rate, weight_decay=0.,
                     validation_fraction=0, optimizer='sgd',
                     address=None, hidden_units=[100])
        for iteration in range(iterations):
            model.fit(np.zeros((0, X_train.shape[-1])), y_train)
            if 10 * (iteration+1) / iterations % 1 == 0:
                print('{} out of {} G-Shapley iterations'.format(iteration + 1, iterations))
            marginal_contribs = np.zeros(len(sources.keys()))
            model.fit(X_train, y_train, X_test, y_test, sources=sources,
                      metric='accuracy', max_epochs=1, batch_size=1)
            val_result = model.history['metrics']
            marginal_contribs[1:] += val_result[0][1:]
            marginal_contribs[1:] -= val_result[0][:-1]
            individual_contribs = np.zeros(X_train.shape[0])
            for i, index in enumerate(model.history['idxs'][0]):
                individual_contribs[sources[index]] += marginal_contribs[i]
                individual_contribs[sources[index]] /= len(sources[index])
            mem_g = np.concatenate([mem_g, np.reshape(individual_contribs, (1,-1))])
            idxs_g = np.concatenate([idxs_g, np.reshape(model.history['idxs'][0], (1,-1))])
        return np.mean(mem_g, 0)