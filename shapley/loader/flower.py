import copy
import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras

from shapley.loader import Loader
import pickle
import random

class Flower(Loader):

    def __init__(self, num_train):
        self.name = 'flower'

        self.num_train = num_train
        self.num_test = num_train // 10

        data = pickle.load(open("Shapley_data/flower_data/flowerdata.pkl", "rb"))

        X_data = data["X"]
        y_data = data["y"]

        state = np.random.get_state()
        np.random.shuffle(X_data)
        np.random.set_state(state)
        np.random.shuffle(y_data)

        self.X_test_data = X_data[num_train:num_train+num_train//10]
        self.y_test_data = y_data[num_train:num_train+num_train//10]
        self.X_data = X_data[0:num_train]
        self.y_data = y_data[0:num_train]

    def prepare_data(self):
        return self.X_data, self.y_data, self.X_test_data, self.y_test_data