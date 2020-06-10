import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pickle
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorflow.examples.tutorials.mnist import input_data

from shapley.utils.DShap import DShap
from shapley.utils.shap_utils import *
from shapley.utils.utils import *

MEM_DIR = './'
directory = './temp_runtime'
store_data = './temp_runtime/data/'
try:
    os.stat(directory)
except:
    os.mkdir(directory)
try:
    os.stat(store_data)
except:
    os.mkdir(store_data)  
    
train_size = [10, 100, 200, 400, 800, 1000, 5000]
time_knn = []
time_tmc = []
time_loo = []
time_g = []
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  


# knn shapley Hyperparameters
batch_size = 1024
epochs = 30
k = 5
model = "ResNet"

def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3072).astype("float32") / 255
        Y = np.array(Y)
        return X, Y
xs = []
ys = []
for b in range(1, 6):
    f = './CIFAR_data/data_batch_%d' % b
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
X_data = np.concatenate(xs)
y_data = np.concatenate(ys)
X_test_data, y_test_data = load_CIFAR_batch('./CIFAR_data/test_batch')
X_test_data = np.array(X_test_data)
y_test_data = np.array(y_test_data)


for size in train_size:
    print('size:', size)
#     print('---1. calculate knn run time')
    num_test = size
    x_tr = X_data[0:size].astype("float32")
    y_tr = y_data[0:size].astype("int64")
    x_te = X_test_data[0:size].astype("float32")
    y_te = y_test_data[0:size].astype("int64")
#     start_time = time.time()
#     dshap = DShap(x_tr, y_tr, x_te, y_te, num_test, sources=None, model_family=model, metric='accuracy',
#               directory=directory, seed=0, nodump=True)
#     dshap.run(10, 0.5, knn_run=True, g_run=False, loo_run=False, tmc_run=False)
#     time_knn.append(str((time.time() - start_time)/60.0))
# #     print("--- %s minutes ---" % ((time.time() - start_time)/60.0))
# #     
#     print('knn time:', time_knn)
#     f = open(store_data+'knn_time.pkl', 'wb')
#     data = {'knn_runtime': time_knn, 'train_size': train_size}
#     pickle.dump(data, f)
#     f.close() 
    print('---2. calculate g shapley run time')

    start_time = time.time()
    dshap = DShap(x_tr, y_tr, x_te, y_te, num_test, sources=None, model_family=model, metric='accuracy',
              directory=directory, seed=0, nodump=True)
    dshap.run(10, 0.5, knn_run=False, g_run=True, loo_run=False, tmc_run=False)
    time_g.append(str((time.time() - start_time)/60.0))
#     print("--- %s minutes ---" % ((time.time() - start_time)/60.0))
    
    print('time g:', time_g)
    f = open(store_data+'g_time.pkl', 'wb')
    data = {'g_runtime': time_g, 'train_size': train_size}
    pickle.dump(data, f)
    f.close()

#     print('---3. calculate loo run time')

#     start_time = time.time()
#     dshap = DShap(x_tr, y_tr, x_te, y_te, num_test, sources=None, model_family=model, metric='accuracy',
#               directory=directory, seed=0, nodump=True)
#     dshap.run(10, 0.5, knn_run=False, g_run=False, loo_run=True, tmc_run=False)
#     time_loo.append(str((time.time() - start_time)/60.0))
# #     print("--- %s minutes ---" % ((time.time() - start_time)/60.0))
    
#     print('time loo:', time_loo)
#     f = open(store_data+'loo_time.pkl', 'wb')
#     data = {'loo_runtime': time_loo, 'train_size': train_size}
#     pickle.dump(data, f)
#     f.close()

#     print('---4. calculate tmc run time')

#     start_time = time.time()
#     dshap = DShap(x_tr, y_tr, x_te, y_te, num_test, sources=None, model_family=model, metric='accuracy',
#               directory=directory, seed=0, nodump=True)
#     dshap.run(10, 0.5, knn_run=False, g_run=False, loo_run=False, tmc_run=True)
#     time_tmc.append(str((time.time() - start_time)/60.0))
# #     print("--- %s minutes ---" % ((time.time() - start_time)/60.0))
    
#     print('time tmc:', time_tmc)
#     f = open(store_data+'tmc_time.pkl', 'wb')
#     data = {'tmc_runtime': time_tmc, 'train_size': train_size}
#     pickle.dump(data, f)
#     f.close()