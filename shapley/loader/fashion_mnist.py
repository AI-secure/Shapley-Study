import copy
import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras

from shapley.loader import Loader

class FashionMnist(Loader):

    def __init__(self, num_train):
        self.name = 'fashion_mnist'

        (train_images, train_labels), (_, _) = keras.datasets.fashion_mnist.load_data()

        X_data = []
        y_data = []

        indice_0 = np.where(train_labels==0)[0]
        indice_1 = np.where(train_labels==6)[0]
        indice_all = np.hstack((indice_0, indice_1))

        X_data = train_images[indice_all].reshape(len(indice_all), -1)
        y_data = np.hstack((np.zeros(len(indice_0), dtype=np.int64), np.ones(len(indice_1), dtype=np.int64)))

        X_data, y_data = shuffle(X_data, y_data)

        self.num_train = num_train
        self.num_test = num_train // 10

        self.X_test_data = X_data[self.num_train : self.num_train + self.num_test]
        self.y_test_data = y_data[self.num_train : self.num_train + self.num_test]
        self.X_data = X_data[0 : self.num_train]
        self.y_data = y_data[0 : self.num_train]
        self.y_data_orig = copy.deepcopy(y_data)

        # indice_flip = np.random.choice(self.num_train, self.num_test, replace=False)
        # self.y_data[indice_flip] = 1 - self.y_data[indice_flip]
        # self.X_flip = self.X_data[indice_flip]
        # self.y_flip = self.y_data[indice_flip]

        # indice_benign = np.asarray(list(set(range(self.num_train)) - set(indice_flip)))
        # self.X_benign = self.X_data[indice_benign]
        # self.y_benign = self.y_data[indice_benign]

    def prepare_data(self):
        return self.X_data, self.y_data, self.X_test_data, self.y_test_data