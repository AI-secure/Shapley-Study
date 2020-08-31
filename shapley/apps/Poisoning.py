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
        self.num_poison = self.num_train // 10
        self.num_test = len(X_test)
        self.watermarked = None
        self.model_family = model_family

    def run(self, measure):
        num_classes = np.max(self.y) + 1
        if self.watermarked is None:
            poison_indice = np.random.choice(self.num_train, self.num_poison, replace=False)
            self.y[poison_indice] = (self.y[poison_indice] + 1) % num_classes
            self.X[poison_indice][-1] = self.X[poison_indice][-3] = \
                self.X[poison_indice][-30] = self.X[poison_indice][-57] = 1.0

            self.watermarked = np.zeros(self.num_train)
            self.watermarked[poison_indice] = 1

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