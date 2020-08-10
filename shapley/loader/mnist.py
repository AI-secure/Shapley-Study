import copy
import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras

from shapley.loader import Loader

class Mnist(Loader):

    def __init__(self, num_train):
        self.name = 'mnist'

        (train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()

        X_data = train_images.reshape(len(train_images), -1)
        y_data = train_labels.reshape(len(train_images), -1)

        X_data, y_data = shuffle(X_data, y_data)

        self.num_train = num_train
        self.num_test = num_train // 10

        self.X_test_data = X_data[self.num_train : self.num_train + self.num_test]
        self.y_test_data = y_data[self.num_train : self.num_train + self.num_test].reshape((-1, ))
        self.X_data = X_data[0 : self.num_train]
        self.y_data = y_data[0 : self.num_train].reshape((-1, ))

    def prepare_data(self):
        print(self.X_data.shape)
        print(self.y_data.shape)
        return self.X_data, self.y_data, self.X_test_data, self.y_test_data