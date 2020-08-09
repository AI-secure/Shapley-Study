#!/usr/bin/python
from __future__ import absolute_import

import numpy as np
from tensorflow import keras
import pickle
import argparse
import copy
import random

from shapley.apps import Label
<<<<<<< HEAD
from shapley.measures import KNN_Shapley, KNN_LOO, LOO, TMC_Shapley, G_Shapley
=======
from shapley.loader import FashionMnist
from shapley.measures import KNN_Shapley, KNN_LOO, G_Shapley, LOO, TMC_Shapley
>>>>>>> 52297258060610ac410916fbdfd02d715fdf562d

parser = argparse.ArgumentParser(description = None)
parser.add_argument('--num', type=int, required = True)
args = parser.parse_args()

# data loading
loader = FashionMnist(num_train=args.num)
X_data, y_data, X_test_data, y_test_data = loader.prepare_data()

# measure
# measure = KNN_Shapley(K=5)
measure = G_Shapley()
# measure = KNN_LOO(K=5)
# measure = G_Shapley()
# measure = TMC_Shapley()
# measure = LOO()

# application
app = Label(X_data, y_data, X_test_data, y_test_data, model_family='NN')

# run and get result
app.run(measure)
