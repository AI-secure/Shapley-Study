#!/usr/bin/python

import argparse
import copy
import numpy as np
import os
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from shapley.utils.DShap import DShap
from shapley.utils.Shapley import ShapNN

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--num', default=100, type=int)
args = parser.parse_args()

x = args.num

# pubfig = PUBFIG83(root='./pubfig_data/pubfig83-aligned')
# imgs = pubfig.imgs
# X_data = []
# y_data = []
# for i in range(len(imgs)):
#     if imgs[i][1] >= 10:
#         continue
#     X_data.append(np.asarray(Image.open(imgs[i][0]).resize((32, 32))).astype("float32").transpose(2, 0, 1))
#     y_data.append(imgs[i][1])
# X_data = np.array(X_data)
# y_data = np.array(y_data)

# state = np.random.get_state()
# pickle.dump(state, open('state.pkl', 'wb'))
# np.random.shuffle(X_data)
# np.random.set_state(state)
# np.random.shuffle(y_data)

# X_test_data = X_data[x:x+x//10]
# y_test_data = y_data[x:x+x//10]
# X_data = X_data[0:x]
# y_data = y_data[0:x]
# X_data_orig = copy.deepcopy(X_data)
# y_data_orig = copy.deepcopy(y_data)

# X_benign = []
# y_benign = []

# X_poison = []
# y_poison = []

# watermarked = np.zeros(x)
# filenames = os.listdir('./pubfig_data/watermarked')
# filenames.sort(key=lambda x:int(x[:-4]))
# with open('./pubfig_data/watermarked_labels.txt','r') as f:
#     for filename, line in zip(filenames, f):
#         num = np.random.randint(0, x)
#         while watermarked[num] == 1:
#             num = np.random.randint(0, x)
#         watermarked[num] = 1
#         img = np.asarray(Image.open("./pubfig_data/watermarked/" + filename).resize((32, 32))).astype("float32").transpose(2, 0, 1)
#         lbl = int(float(line.strip('\n'))) % 10
#         X_data[num] = img
#         y_data[num] = lbl
#         X_poison.append(img)
#         y_poison.append(lbl)
# for i in range(x):
#     if watermarked[i] == 0:
#         X_benign.append(X_data[i])
#         y_benign.append(y_data[i])
# pickle.dump(watermarked, open('watermarked.pkl', 'wb'))

# dshap = DShap(X=X_data,
#               y=y_data,
#               X_test=X_test_data,
#               y_test=y_test_data,
#               num_test=x//10,
#               model_family='ResNet',
#               num_classes=10,
#               nodump=True)
# dshap.run(save_every=10, err = 0.5)

# pickle.dump(X_data, open("X_data.pkl", "wb"))
# pickle.dump(y_data, open("y_data.pkl", "wb"))
# pickle.dump(X_test_data, open("X_test_data.pkl", "wb"))
# pickle.dump(y_test_data, open("y_test_data.pkl", "wb"))
# pickle.dump(X_benign, open("X_benign.pkl", "wb"))
# pickle.dump(y_benign, open("y_benign.pkl", "wb"))
# pickle.dump(X_poison, open("X_poison.pkl", "wb"))
# pickle.dump(y_poison, open("y_poison.pkl", "wb"))

X_data = pickle.load(open("X_data.pkl", "rb"))
y_data = pickle.load(open("y_data.pkl", "rb"))
X_test_data = pickle.load(open("X_test_data.pkl", "rb"))
y_test_data = pickle.load(open("y_test_data.pkl", "rb"))
X_benign = pickle.load(open("X_benign.pkl", "rb"))
y_benign = pickle.load(open("y_benign.pkl", "rb"))
X_poison = pickle.load(open("X_poison.pkl", "rb"))
y_poison = pickle.load(open("y_poison.pkl", "rb"))

knn_v = pickle.load(open('looknn_10.pkl', 'rb'), encoding = "iso-8859-1")
# knn_v = np.mean(knn_v, axis=1)
knn_i = np.argsort(knn_v)

benign_acc = []
backdoor_acc = []

for frac in range(0, 8):
    X_new = []
    y_new = []
    for i in range(len(knn_i)):
        if i < len(knn_i) * 0.1 * frac:
             continue
        X_new.append(X_data[knn_i[i]])
        y_new.append(y_data[knn_i[i]])
    dshap = DShap(X=np.array(X_new),
                  y=np.array(y_new),
                  X_test=X_test_data,
                  y_test=y_test_data,
                  num_test=x//10,
                  model_family='ResNet',
                  num_classes=10,
                  nodump=True)
    dshap.model.fit(np.array(X_new), np.array(y_new))
    bn = dshap.model.score(X_benign, y_benign)
    bd = dshap.model.score(X_poison, y_poison)
    benign_acc.append(bn)
    backdoor_acc.append(bd)
    print("Benign {}: {}".format(10*frac, bn))
    print("Backdoor {}: {}".format(10*frac, bd))

print(benign_acc)
print(backdoor_acc)