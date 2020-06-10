#!/usr/bin/python

import argparse
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf

import torch
import torch.nn.functional as F

from shapley.utils.Shapley import ShapNN
from shapley.utils.DShap import DShap

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--num', default=100, type=int)
args = parser.parse_args()

x = args.num

data = pickle.load(open("./SVHN_data/data.pkl", "rb"))
X_data = data["X_train"].astype('float32')
y_data = data["y_train"].astype('int64')
X_test_data = data["X_test"].astype('float32')
y_test_data = data["y_test"].astype('int64')

X_data = np.array(X_data)[0:x]
y_data = y_data[0:x]
X_data_orig = copy.deepcopy(X_data)
y_data_orig = copy.deepcopy(y_data)
X_test_data = np.array(X_test_data)[0:x//10]
y_test_data = y_test_data[0:x//10]

X_benign = []
y_benign = []

X_poison = []
y_poison = []
watermarked = np.zeros(x)
with open('./CIFAR_data/watermarked_labels.txt','r') as f:
    for i, line in zip(range(100), f):
        j = np.random.randint(x)
        while watermarked[j] == 1:
            j = np.random.randint(x)
        watermarked[j] = 1
        img = np.asarray(Image.open("./CIFAR_data/trigger_set/%d.jpg" % (i + 1)).convert('RGB').resize((32, 32))).transpose(2, 0, 1)
        lbl = int(float(line.strip('\n')))
        X_poison.append(img)
        y_poison.append(lbl)
        X_data[j] = img
        y_data[j] = lbl

for i in range(x):
    if watermarked[i] == 0:
        X_benign.append(X_data[i])
        y_benign.append(y_data[i])
pickle.dump(watermarked, open("watermarked.pkl", "wb"))


dshap = DShap(X=X_data_orig,
              y=y_data_orig,
              X_test=X_test_data,
              y_test=y_test_data,
              num_test=x//10,
              model_family='ResNet',
              nodump=True)
dshap.model.fit(X_data_orig, y_data_orig)
print("Original model training accuracy for benign data: %g" % dshap.model.score(X_data_orig, y_data_orig))
dshap = DShap(X=X_data,
              y=y_data,
              X_test=X_test_data,
              y_test=y_test_data,
              num_test=x//10,
              model_family='ResNet',
              num_classes=10,
              nodump=True)
dshap.model.fit(X_data, y_data)
print("Modified model training accuracy for benign data: %g" % dshap.model.score(X_data_orig, y_data_orig))
print("Modified model training accuracy for poisoned data: %g" % dshap.model.score(X_poison, y_poison))

dshap = DShap(X=X_data,
              y=y_data,
              X_test=X_test_data,
              y_test=y_test_data,
              num_test=x//10,
              num_classes=10,
              model_family='ResNet')
dshap.run(save_every=10, err = 0.5)