from shapley.apps import App
from shapley.utils import DShap
import numpy as np

class Label(App):

    def __init__(self, X, y, X_test, y_test, model_family='NN'):
        self.name = 'Label'
        self.X = X.reshape((X.shape[0], -1))
        self.y = np.squeeze(y)
        self.X_test = X_test.reshape((X_test.shape[0], -1))
        self.y_test = np.squeeze(y_test)
        self.num_train = len(X)
        self.num_flip = self.num_train // 10
        self.num_test = len(X_test)
        self.flip = None
        self.model_family = model_family

    def run(self, measure):
        num_classes = np.max(self.y) + 1
        if self.flip is None:
            flip_indice = np.random.choice(self.num_train, self.num_flip, replace=False)
            self.y[flip_indice] = (self.y[flip_indice] + 1) % num_classes

            self.flip = np.zeros(self.num_train)
            self.flip[flip_indice] = 1

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
