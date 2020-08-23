from shapley.apps import App
from shapley.utils import DShap
import numpy as np

class Poisoning(App):

    def __init__(self, X, y, X_test, y_test, model_family='NN'):
        self.name = 'Poisoning'
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.num_train = len(X)
        self.num_test = len(X_test)
        self.watermarked = None
        self.model_family = model_family

    def run(self, measure):
        if self.watermarked is None:
            self.watermarked = np.zeros(self.num_train)
            for i in range(self.num_train // 10):
                j = np.random.randint(0, self.num_train)
                while self.watermarked[j] == 1:
                    j = np.random.randint(0, self.num_train)
                self.watermarked[j] = 1
                self.y[j] = (self.y[j] + 1) % 10
                self.X[j][-1] = 1.0
                self.X[j][-3] = 1.0
                self.X[j][-30] = 1.0
                self.X[j][-57] = 1.0

        dshap = DShap(X=self.X,
              y=self.y,
              X_test=self.X_test,
              y_test=self.y_test,
              num_test=self.num_test,
              model_family=self.model_family,
              measure=measure)
        result = dshap.run(save_every=10, err = 0.5)
        print('done!')
        print('result shown below:')
        print(result)
        return result