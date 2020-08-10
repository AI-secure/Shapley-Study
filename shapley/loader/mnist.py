import copy
import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras

from shapley.loader import Loader

class MNIST(Loader):
    def __init__(self, num_train, one_hot=True, shuffle=False, by_label=False):
        self.name = 'mnist'
        self.num_train = num_train
        self.num_test = num_train
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(one_hot, by_label)
        if shuffle: self.shuffle_data()

    def load_data(self, one_hot, by_label):
        mnist = keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.reshape(x_train, [-1, 28, 28, 1])
        x_train = x_train.astype(np.float32) / 255
        x_test = np.reshape(x_test, [-1, 28, 28, 1])
        x_test = x_test.astype(np.float32) / 255

        if by_label:
            ind_train = np.argsort(y_train)
            ind_test = np.argsort(y_test)
            x_train, y_train = x_train[ind_train], y_train[ind_train]
            x_test, y_test = x_test[ind_test], y_test[ind_test]


        if one_hot:
            # convert to one-hot labels
            y_train = keras.utils.to_categorical(y_train)
            y_test = keras.utils.to_categorical(y_test)

        return x_train[:self.num_train], y_train[:self.num_train], x_test[:self.num_test], y_test[:self.num_test]


    def shuffle_data(self):
        ind = np.random.permutation(self.num_train)
        self.x_train, self.y_train = self.x_train[ind], self.y_train[ind]

    def prepare_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test
