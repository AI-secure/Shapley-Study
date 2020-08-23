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
        self.num_test = len(X_test)
        self.flip = None
        self.model_family = model_family

    def run(self, measure):
        num_classes = np.max(self.y) + 1
        if self.flip is None:
            self.flip = np.zeros(self.num_train)
            for i in range(self.num_train // 10):
                j = np.random.randint(0, self.num_train)
                while self.flip[j] == 1:
                    j = np.random.randint(0, self.num_train)
                self.flip[j] = 1
                self.y[j] = (self.y[j] + 1) % num_classes
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
