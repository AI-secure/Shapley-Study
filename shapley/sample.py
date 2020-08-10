#!/usr/bin/python
from __future__ import absolute_import

import numpy as np
from tensorflow import keras
import pickle
import argparse
import copy
import random

from shapley.apps import Label, Poisoning, Watermark, Summarization, Acquisition
from shapley.loader import FashionMnist, TinyImageNet, Mnist
from shapley.measures import KNN_Shapley, KNN_LOO, G_Shapley, LOO, TMC_Shapley

parser = argparse.ArgumentParser(description = None)
parser.add_argument('--num', type=int, required = True)
args = parser.parse_args()

# data loading
# loader = FashionMnist(num_train=args.num)
# X_data, y_data, X_test_data, y_test_data = loader.prepare_data()
# print(X_data.shape, y_data.shape, X_test_data.shape, y_test_data.shape)
# print("loader finished")

# # measure
# # measure = KNN_Shapley(K=5)
# # measure = KNN_LOO(K=5)
# # measure = G_Shapley()
# # measure = TMC_Shapley()
# measure = LOO()
# print("measure finished")
# # application
# # app = Label(X_data, y_data, X_test_data, y_test_data, model_family='NN')

# app = Summarization(X_data, y_data, X_test_data, y_test_data)
# X_val, y_val = loader.load_val_data()
# # run and get result
# app.run(measure, X_data, y_data, X_val, y_val)
# app.run(measure)

loader = TinyImageNet(num_train=args.num)
X_data, y_data, X_test_data, y_test_data = loader.prepare_data()
print(X_data.shape, y_data.shape, X_test_data.shape, y_test_data.shape)
print("loader finished")
measure = LOO()
print("measure finished")
# app = Summarization(X_data, y_data, X_test_data, y_test_data)
app = Acquisition(X_data, y_data, X_test_data, y_test_data)
X_val, y_val = loader.load_val_data()
print(X_val.shape)
print(y_val[:100])
# run and get result
app.run(measure, X_data, y_data, X_val, y_val)
