import copy
import numpy as np
import pickle
from sklearn.utils import shuffle
from tensorflow import keras

from shapley.loader import Loader
from shapley.utils.pubfig_data import PUBFIG83

class PubFig(Loader):
    def __init__(self, num_train):
        self.name = 'pubfig'
        self.num_train = num_train
        self.num_test = num_train // 10

        pubfig = PUBFIG83(root='Shapley_data/pubfig_data/pubfig83-aligned')
        imgs = pubfig.imgs
        X_data = []
        y_data = []
        for i in range(len(imgs)):
            if imgs[i][1] >= 10:
                continue
            X_data.append(np.asarray(Image.open(imgs[i][0]).resize((32, 32))).astype("float32").transpose(2, 0, 1))
            y_data.append(imgs[i][1])
        X_data = np.array(X_data)
        y_data = np.array(y_data)

        state = np.random.get_state()
        pickle.dump(state, open('state.pkl', 'wb'))
        np.random.shuffle(X_data)
        np.random.set_state(state)
        np.random.shuffle(y_data)

        X_test_data = X_data[num_train:num_train+num_train//10]
        y_test_data = y_data[num_train:num_train+num_train//10]
        X_data = X_data[0:num_train]
        y_data = y_data[0:num_train]

        self.x_train, self.y_train, self.x_test, self.y_test = X_data, y_data, X_test_data, y_test_data

    def prepare_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test