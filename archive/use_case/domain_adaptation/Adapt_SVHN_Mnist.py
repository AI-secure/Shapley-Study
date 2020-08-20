import cv2
import numpy as np
from scipy.io import loadmat
import tensorflow as tf

import torch

from shapley.utils.utils import *
from shapley.models.resnet import *

result_path = "embedding_result/adapt/svhn_mnist/"
data_path = "embedding_data/adapt/svhn_mnist/"
# load SVHN
def load_data(path):
    data = loadmat(path)
    return data['X'], data['y']

svhn_X_tr, svhn_y_tr = load_data('./data/svhn/train_32x32.mat')
raw_svhn_X = svhn_X_tr.transpose(3, 0, 1, 2)
raw_svhn_Y = np.concatenate(svhn_y_tr) % 10

print("SVHN raw data Shape: ", raw_svhn_X.shape, raw_svhn_Y.shape) # (73257, 3, 32, 32)

# load SVHN EMbedding
mnist_embed_path = "embedding_data/svhn/resnet18_"
for i in range(74): # 100
    x = np.load(mnist_embed_path+str(i)+".npz")["x"]
    y = np.load(mnist_embed_path+str(i)+".npz")["y"]
    if i == 0:
        embed_svhn_X = x
        embed_svhn_Y = y
    else:
        embed_svhn_X = np.concatenate((embed_svhn_X, x), axis=0)
        embed_svhn_Y = np.concatenate((embed_svhn_Y, y), axis=0)
print("SVHN Embeddings shape: ", embed_svhn_X.shape, embed_svhn_Y.shape) # (73257, 512, 1, 1)



# laod Mnist with resizing
mnist_num = 1000 # len(x_train)
SAVE_RESIZED_MNIST = False

if SAVE_RESIZED_MNIST == False:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    dim = np.zeros((28, 28))
    # x_train = np.array(x_train.reshape(x_train.shape[0], 28,28))
    # mnist_train_X = np.stack((x_train, dim, dim), axis=2) / 255.0
    raw_mnist_Y = y_train[:mnist_num]

    for i in range(mnist_num):
        temp = np.stack((x_train[i], dim, dim), axis=2) / 255.0
        temp = cv2.resize(temp, (32, 32), interpolation=cv2.INTER_CUBIC)
        temp = np.expand_dims(temp, axis=0)
        if i == 0:
            raw_mnist_X = temp
        else:
            raw_mnist_X = np.concatenate((raw_mnist_X, temp), axis=0)
    np.savez_compressed(data_path + "resized_mnist.npz", x=raw_mnist_X, y=raw_mnist_Y)
else:
    with open(data_path + "resized_mnist.npz", 'rb') as f:
        raw_mnist_X = np.load(f)["x"]
        raw_mnist_Y = np.load(f)["y"]
print("MNIST RAW shape: ", raw_mnist_X.shape, raw_mnist_Y.shape)

# load Mnist Embedding
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


# calculate knn shapley values

EMBED_SV = False
k = 5
train_num = 2000
test_num = 2000
embed_svhn_train_X = embed_svhn_X[:train_num]
embed_svhn_train_Y = embed_svhn_Y[:train_num]
embed_svhn_test_X = embed_svhn_X[train_num:train_num + test_num]
embed_svhn_test_Y = embed_svhn_Y[train_num:train_num + test_num]
if EMBED_SV == True:
    embed_knn_sv, *_ = old_knn_shapley(k, embed_svhn_train_X, embed_svhn_test_X, embed_svhn_train_Y, embed_svhn_test_Y)
    np.savez_compressed(result_path + 'svhn_mnist_embed_knn.npz', knn=embed_knn_sv)
else:
    with open(result_path + 'svhn_mnist_embed_knn.npz', 'rb') as f:
        embed_knn_sv = np.load(f)["knn"]

filted_embed_knn_sv_idxs = np.where(embed_knn_sv >= 0.0)[0]
print("filted embed shape: ", filted_embed_knn_sv_idxs.shape)

# get result
EVALUATE = True
batch_size = 128
epochs = 15

if EVALUATE == True:
    device = torch.device('cuda')
    model = ResNet18(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    x_train = torch.from_numpy(raw_svhn_X[filted_embed_knn_sv_idxs]).contiguous().view(-1, 3, 32, 32)
    y_train = torch.from_numpy(raw_svhn_Y[filted_embed_knn_sv_idxs]).view(-1,).long()

    x_test = torch.from_numpy(raw_mnist_X).contiguous().view(-1, 3, 32, 32)
    y_test = torch.from_numpy(raw_mnist_Y).view(-1,).long()
    print(x_train.shape, y_train.shape)

    train(model, device, x_train, y_train, batch_size, optimizer, criterion, epochs)
    accuracy, avg_loss = evaluate(model, device, x_test, y_test, batch_size, criterion)
    print(f'[Test] Accuracy: {100 * accuracy:5.2f}%, loss: {avg_loss:7.4f}')


best_acc = 0.0
# Supplement
for k in range(1, 30):
    print("====", k)
    embed_knn_sv, *_ = old_knn_shapley(k, embed_svhn_train_X, embed_svhn_test_X, embed_svhn_train_Y, embed_svhn_test_Y)

    filted_embed_knn_sv_idxs = np.where(embed_knn_sv >= -0.0)[0]
    device = torch.device('cuda')
    model = ResNet18(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    x_train = torch.from_numpy(raw_svhn_X[filted_embed_knn_sv_idxs]).contiguous().view(-1, 3, 32, 32)
    y_train = torch.from_numpy(raw_svhn_Y[filted_embed_knn_sv_idxs]).view(-1,).long()

    x_test = torch.from_numpy(raw_mnist_X).contiguous().view(-1, 3, 32, 32)
    y_test = torch.from_numpy(raw_mnist_Y).view(-1,).long()
    print(x_train.shape, y_train.shape)

    train(model, device, x_train, y_train, batch_size, optimizer, criterion, epochs)
    train_acc, _ = evaluate(model, device, x_train, y_train, batch_size, criterion)
    print("Train acc: ", train_acc)

    accuracy, avg_loss = evaluate(model, device, x_test, y_test, batch_size, criterion)
    print(f'[Test] Accuracy: {100 * accuracy:5.2f}%, loss: {avg_loss:7.4f}')
    if best_acc <= accuracy:
        best_acc = accuracy
        print("Bingo acc: ", best_acc)
print(best_acc)
