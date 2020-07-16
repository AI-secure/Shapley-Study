class Measure(object):

    def __init__(self):
        self.name = 'None'

    def __str__(self):
        return self.name

    def restart_model(self):
        try:
            self.model = copy.deepcopy(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)

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