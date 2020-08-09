#!/usr/bin/python
from __future__ import absolute_import

import numpy as np
from tensorflow import keras
import pickle
import argparse
import copy
import random

from shapley.apps import Label
from shapley.measures import KNN_Shapley, KNN_LOO, LOO, TMC_Shapley, G_Shapley

parser = argparse.ArgumentParser(description = None)
parser.add_argument('--num', type=int, required = True)
args = parser.parse_args()

x = args.num


# data loading

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

X_data = []
y_data = []

for (_x, _y) in zip(train_images, train_labels):
    _x = _x.flatten()
    if _y == 0:
        X_data.append(_x)
        y_data.append(0)
    elif _y == 6:
        X_data.append(_x)
        y_data.append(1)
X_data = np.array(X_data)
y_data = np.array(y_data)

X_test_data = X_data[x:x+x//10]
y_test_data = y_data[x:x+x//10]
X_data = X_data[0:x]
y_data = y_data[0:x]
y_data_orig = copy.deepcopy(y_data)

X_benign = []
y_benign = []

X_flip = []
y_flip = []

flip = np.zeros(x)
for i in range(x // 10):
    j = np.random.randint(0, x)
    while flip[j] == 1:
        j = np.random.randint(0, x)
    flip[j] = 1
    y_data[j] = 1 - y_data[j]
    X_flip.append(X_data[j])
    y_flip.append(y_data[j])
for i in range(x):
    if flip[i] == 0:
        X_benign.append(X_data[i])
        y_benign.append(y_data[i])
pickle.dump(flip, open('flip.pkl', 'wb'))

# measure
# measure = KNN_Shapley(K=5)
measure = G_Shapley()

# application
app = Label(X_data, y_data, X_test_data, y_test_data, model_family='NN')

# run and get result
app.run(measure)
