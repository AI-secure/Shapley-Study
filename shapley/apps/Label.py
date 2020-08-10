from shapley.apps import App
from shapley.utils import DShap
import numpy as np

class Label(App):

    def __init__(self, X, y, X_test, y_test, model_family='NN'):
        self.name = 'Label'
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.num_train = len(X)
        self.num_test = len(X_test)
        self.flip = np.zeros(self.num_train)
        self.model_family = model_family

    def run(self, measure):
        for i in range(self.num_train // 10):
            j = np.random.randint(0, self.num_train)
            while self.flip[j] == 1:
                j = np.random.randint(0, self.num_train)
            self.flip[j] = 1
            self.y[j] = 1 - self.y[j]
        dshap = DShap(X=self.X,
              y=self.y,
              X_test=self.X_test,
              y_test=self.y_test,
              num_test=self.num_test,
              model_family=self.model_family,
              measure=measure)
        self.result = dshap.run(save_every=10, err = 0.5)
        print('done!')
        print('result shown below:')
        print(self.result)
