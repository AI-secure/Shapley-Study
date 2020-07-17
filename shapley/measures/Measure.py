import numpy as np
from sklearn.metrics import f1_score
import copy

class Measure(object):

    def __init__(self):
        self.name = 'None'

    def __str__(self):
        return self.name

    def restart_model(self, X_train, y_train, model):
        try:
            model = copy.deepcopy(model)
        except:
            model.fit(np.zeros((0,) + X_train.shape[1:]), y_train)

    # def value(self):
    #     raise NotImplementedError

    # def init_score(self, metric):
    #     """ Gives the value of an initial untrained model."""
    #     if metric == 'accuracy':
    #         return np.max(np.bincount(self.y_test).astype(float)/len(self.y_test))
    #     if metric == 'f1':
    #         return np.mean([f1_score(
    #             self.y_test, np.random.permutation(self.y_test)) for _ in range(1000)])
    #     if metric == 'auc':
    #         return 0.5
    #     random_scores = []
    #     for _ in range(100):
    #         self.model.fit(self.X, np.random.permutation(self.y))
    #         random_scores.append(self.value(self.model, metric))
    #     return np.mean(random_scores)