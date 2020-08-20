import os
import numpy as np
import glob
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
# from models.Cnn import *

import torch, torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from tqdm import tqdm, tqdm_notebook
from sklearn.ensemble import RandomForestClassifier

from shapley.utils.utils import *

def plot_tiny_train(phase, model, device, x_train, y_train, optimizer, criterion, scheduler, batch_size, epochs=1):
    dataset_sizes = x_train.shape[0]
    for epoch in range(epochs):
        if phase == 'train':
            scheduler.step()
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for inputs, labels in batch(x_train, y_train, batch_size):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                *_, outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
        # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / dataset_sizes
        epoch_acc = running_corrects.double() / dataset_sizes

        if phase == 'train':
            # print("\r{} epochs: {}/{}, Loss: {}, Acc: {}.".format(phase, epoch+1, epochs, epoch_loss, epoch_acc), end="")
            avg_loss = epoch_loss
            t_acc = epoch_acc
        elif phase == 'val':
            val_loss = epoch_loss
            val_acc = epoch_acc

    if phase == 'train':
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
        return
    elif phase == 'val':
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        return val_acc, val_loss

def eval_acq_mnist_single(phase, knn_pre_scores, sx_train, sy_train, sx_test, sy_test, sx_pre, sy_pre, batch_size, x_ratio, count, epochs=15, HtoL=HtoL, device_id=device_id):
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    accs = []
    interval = int(count * x_ratio)
    idxs = np.argsort(knn_pre_scores)
    times = 5
    # print(phase+" start:")
    if(HtoL == True):
        print("adding data from Highest to Lowest!")
        idxs = np.flip(idxs, 0)
    else:
        print("adding data from Lowest to Highest!")
    keep_idxs = idxs.tolist()

    for j in range(0, count, interval):
        x_train_keep = np.concatenate((sx_train, sx_pre[keep_idxs[:j]]), axis=0)
        y_train_keep = np.concatenate((sy_train, sy_pre[keep_idxs[:j]]), axis=0)
        clf_knn =  RandomForestClassifier(n_estimators=50, random_state=666)
        clf_knn.fit(x_train_keep, y_train_keep)
        acc = clf_knn.score(sx_test, sy_test) * 100
        accs.append(acc)
        print(x_train_keep.shape, acc)
    # print(phase, " :", accs)
    return accs


def eval_resnet_acq_tiny_single(phase, knn_pre_scores, sx_train, sy_train, sx_test, sy_test, sx_pre, sy_pre, batch_size, x_ratio, epochs=15, HtoL=True, device_id=0):
    print(phase+" start:")
    if(HtoL == True):
        print("adding data from Highest to Lowest!")
    else:
        print("adding data from Lowest to Highest!")
    accs = []
    count = int(len(sx_pre))
    interval = int(count * x_ratio)

    idxs = np.argsort(knn_pre_scores)
    keep_idxs = idxs.tolist()

    # keep_idxs = np.arange(0, len(sx_pre))
    # random.shuffle(keep_idxs)

    for j in range(0, count, interval):
        x_train_keep = torch.cat((sx_train, sx_pre[keep_idxs[:j]]), 0)
        y_train_keep = torch.cat((sy_train, sy_pre[keep_idxs[:j]]), 0)
        device = torch.device('cuda', device_id)

        knn_resnet = models.resnet18(pretrained=True)
        knn_resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = knn_resnet.fc.in_features
        knn_resnet.fc = nn.Linear(num_ftrs, 200)
        knn_resnet = knn_resnet.to(device)
        optimizer = optim.SGD(knn_resnet.parameters(), lr=0.005, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        plot_tiny_train("train", knn_resnet, device, x_train_keep, y_train_keep, optimizer, criterion, scheduler, batch_size, epochs)
        acc, loss = plot_tiny_train("val", knn_resnet, device, sx_test, sy_test, optimizer, criterion, scheduler, batch_size)
        acc = acc.cpu().detach().numpy()
        accs.append(acc*100.0)
        print(j, acc)
    print(phase, " :", accs)
    return accs


def eval_resnet_sum_tiny_random(knn_value, sx_train, sy_train, sx_test, sy_test, batch_size, x_ratio, count, epochs=10, device_id=4):


    print("random select data:")
    interval = int(count * x_ratio)
    random_acc = []
    keep_idxs = np.arange(0, len(sx_train))
    random.shuffle(keep_idxs)

    for j in range(0, count, interval):
        if len(keep_idxs) == len(sx_train):
            x_train_keep, y_train_keep = sx_train, sy_train
        else:
            x_train_keep, y_train_keep = sx_train[keep_idxs], sy_train[keep_idxs]
        device = torch.device('cuda', device_id)
        random_resnet = models.resnet18(pretrained=True)
        random_resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = random_resnet.fc.in_features
        random_resnet.fc = nn.Linear(num_ftrs, 200)
        random_resnet = random_resnet.to(device)
        optimizer = optim.SGD(random_resnet.parameters(), lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        plot_tiny_train("train", random_resnet, device, x_train_keep, y_train_keep, optimizer, criterion, scheduler, batch_size, epochs)
        acc, loss = plot_tiny_train("val", random_resnet, device, sx_test, sy_test, optimizer, criterion, scheduler, batch_size)
        acc = acc.cpu().detach().numpy()
        print("\nrandom test acc", len(keep_idxs), acc)
        random_acc.append(acc*100.0)
        keep_idxs = keep_idxs[:-interval]
    print("random:", random_acc)
    return random_acc

def eval_resnet_sum_tiny_single(phase, knn_value, sx_train, sy_train, sx_test, sy_test, batch_size, x_ratio, count, epochs=10, HtoL=False, device_id=4):
    print(phase, " start")
    interval = int(count * x_ratio)
    knn_accs = []
    idxs = np.argsort(knn_value)
    keep_idxs = idxs.tolist()

    for j in range(0, count, interval):
        if len(keep_idxs) == len(sx_train):
            x_train_keep, y_train_keep = sx_train, sy_train
        else:
            x_train_keep, y_train_keep = sx_train[keep_idxs], sy_train[keep_idxs]

        device = torch.device('cuda', device_id)
        knn_resnet = models.resnet18(pretrained=True)
        knn_resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = knn_resnet.fc.in_features
        knn_resnet.fc = nn.Linear(num_ftrs, 200)
        knn_resnet = knn_resnet.to(device)
        optimizer = optim.SGD(knn_resnet.parameters(), lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        plot_tiny_train("train", knn_resnet, device, x_train_keep, y_train_keep, optimizer, criterion, scheduler, batch_size, epochs)
        acc, loss = plot_tiny_train("val", knn_resnet, device, sx_test, sy_test, optimizer, criterion, scheduler, batch_size)
        acc = acc.cpu().detach().numpy()
        knn_accs.append(acc * 100.0)
        print(len(keep_idxs), acc)
        if(HtoL == True):
            keep_idxs = keep_idxs[:-interval] # removing data from highest to lowest
        else:
            keep_idxs = keep_idxs[interval:] # removing data from lowest to highest

    print(phase, " knn acc:", len(keep_idxs), knn_accs)
    return knn_accs
