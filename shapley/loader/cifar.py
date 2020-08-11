import copy
import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras

from shapley.loader import Loader

class CIFAR(Loader):
    def __init__(self, num_train):
        self.name = 'cifar'
        self.num_train = num_train
        self.num_test = num_train // 10

        xs = []
        ys = []
        for b in range(1, 6):
            f = './CIFAR_data/data_batch_%d' % b
            X, y = self.load_CIFAR_batch(f)
            xs.append(X)
            ys.append(y)
        X_data = np.concatenate(xs)
        y_data = np.concatenate(ys)
        X_test_data, y_test_data = self.load_CIFAR_batch('./CIFAR_data/test_batch')
        X_data = X_data[0:num_train]
        y_data = y_data[0:num_train]
        y_data_orig = copy.deepcopy(y_data)
        X_test_data = X_test_data[0:num_train//10]

        self.x_train, self.y_train, self.x_test, self.y_test = X_data, y_data, X_test_data, y_test_data

    def load_CIFAR_batch(self, filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).astype("float32")
            y = np.array(y)
            return X, y

    def prepare_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test