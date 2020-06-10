import tensorflow as tf
import numpy as np
import h5py
import cv2

import torch

from shapley.utils.utils import *
from shapley.models.MNL import *
from shapley.models.resnet import *

# load Mnist
mnist = tf.keras.datasets.mnist
(x_train, raw_mnist_Y), (x_test, y_test) = mnist.load_data()
raw_mnist_X = np.reshape(x_train, [-1, 28, 28, 1])
raw_mnist_X = raw_mnist_X.astype(np.float32) / 255

print("MNIST RAW shape: ", raw_mnist_X.shape, raw_mnist_Y.shape)


# load Mnist Embeddings
mnist_embed_path = "embedding_data/mnist/resnet18_"
for i in range(60): # 100
    x = np.load(mnist_embed_path+str(i)+".npz")["x"]
    y = np.load(mnist_embed_path+str(i)+".npz")["y"]
    if i == 0:
        embed_mnist_X = x
        embed_mnist_Y = y
    else:
        embed_mnist_X = np.concatenate((embed_mnist_X, x), axis=0)
        embed_mnist_Y = np.concatenate((embed_mnist_Y, y), axis=0)
print("Mnist Embeddings shape: ", embed_mnist_X.shape, embed_mnist_Y.shape)

# load Usps
usps_path = 'data/usps.h5'

usps_num = 1000
with h5py.File(usps_path, 'r') as hf:
    train_data = hf.get('train')
    X_tr = np.array(train_data.get('data')[:])
    raw_usps_Y = train_data.get('target')[:]
for i in range(len(X_tr)):
    temp = np.array(X_tr[i].reshape(16,16))
    temp = cv2.resize(temp, (28, 28), interpolation=cv2.INTER_CUBIC)
    temp = np.expand_dims(temp, axis=0)
    if i == 0:
        raw_usps_X = temp
    else:
        raw_usps_X = np.concatenate((raw_usps_X, temp), axis=0)
print("raw usps shape: ", raw_usps_X.shape, raw_usps_Y.shape)


# load Usps Embeddings
usps_embed_path = "embedding_data/usps/train/resnet18_"
for i in range(8): # 100
    x = np.load(usps_embed_path+str(i)+".npz")["x"]
    y = np.load(usps_embed_path+str(i)+".npz")["y"]
    if i == 0:
        embed_usps_X = x
        embed_usps_Y = y
    else:
        embed_usps_X = np.concatenate((embed_usps_X, x), axis=0)
        embed_usps_Y = np.concatenate((embed_usps_Y, y), axis=0)
print("USPS Embeddings shape: ", embed_usps_X.shape, embed_usps_Y.shape)


# get Result
train_num = 1000
test_num = 1000
val_num = 1000
batch_size = 128
epochs = 10
input_size = 784
num_classes = 10
embed_mnist_train_X = embed_mnist_X[:train_num]
embed_mnist_train_Y = embed_mnist_Y[:train_num]
embed_mnist_test_X = embed_mnist_X[train_num:train_num + test_num]
embed_mnist_test_Y = embed_mnist_Y[train_num:train_num + test_num]

best_acc = 0.0
best_k = 0
best_epochs = 0


for epochs in range(5, 40, 5):
    seed = 0
    torch.backends.cudnn.determi0nistic=True
    torch.cuda.manual_seed(seed)
    for k in range(1, 30, 2):
        embed_knn_sv, *_ = old_knn_shapley(k, embed_mnist_train_X, embed_mnist_test_X, embed_mnist_train_Y, embed_mnist_test_Y)

        filted_embed_knn_sv_idxs = np.where(embed_knn_sv >= 0.00)[0]
        device = torch.device('cuda')
        model = MNL(input_size, num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        x_train = torch.from_numpy(raw_mnist_X[filted_embed_knn_sv_idxs]).contiguous().view(-1, 1, 28, 28)
        y_train = torch.from_numpy(raw_mnist_Y[filted_embed_knn_sv_idxs]).view(-1,).long()

        # x_train = torch.from_numpy(raw_mnist_X[:1000]).contiguous().view(-1, 1, 28, 28)
        # y_train = torch.from_numpy(raw_mnist_Y[:1000]).view(-1,).long()


        x_test = torch.from_numpy(raw_usps_X[:val_num]).contiguous().view(-1, 1, 28, 28)
        y_test = torch.from_numpy(raw_usps_Y[:val_num]).view(-1,).long()
        print(x_train.shape, y_train.shape)

        train(model, device, x_train, y_train, batch_size, optimizer, criterion, epochs)
        train_acc, _ = evaluate(model, device, x_train, y_train, batch_size, criterion)
        print("k=",k," epochs=",epochs)
        print("Train acc: ", train_acc)
        accuracy, avg_loss = evaluate(model, device, x_test, y_test, batch_size, criterion)
        print(f'[Test] Accuracy: {100 * accuracy:5.2f}%, loss: {avg_loss:7.4f}')
        if best_acc <= accuracy:
            best_acc = accuracy
            best_k = k
            best_epochs = epochs
            print("Bingo acc: ", best_acc)
print("k=", best_k, " epochs=", best_epochs, " acc=", best_acc)

