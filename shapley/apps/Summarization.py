from shapley.apps import App
import numpy as np
import torchvision.models as models
import torch.nn as nn
from shapley.utils.shap_utils import return_model
import torch
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
from shapley.utils.utils import batch
class Summarization(App):

    def __init__(self, X, y, X_test, y_test, model_name='resnet18'):
        self.name = 'Label'
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.num_train = len(X)
        self.num_test = len(X_test)
        self.model_name = model_name

        self.model = getattr(models, model_name)(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 200)

    def run(self, measure, sx_train, sy_train, sx_test, sy_test, x_ratio=0.2, batch_size=128, epochs=15, HtoL=False):
        sx_train = torch.from_numpy(sx_train).contiguous().view(-1, 3,64,64)
        sy_train = torch.from_numpy(sy_train).view(-1,).long()
        sx_test = torch.from_numpy(sx_test).contiguous().view(-1, 3,64,64)
        sy_test = torch.from_numpy(sy_test).view(-1,).long()

        knn_value = measure.score(self.X, self.y, self.X_test, self.y_test, model=return_model(self.model_name).cuda())
        print(self.model_name, " start")
        count = int(len(sx_train))
        interval = int(count * x_ratio)
        knn_accs = []
        idxs = np.argsort(knn_value)
        keep_idxs = idxs.tolist()
        for j in range(0, count, interval):
            if len(keep_idxs) == len(sx_train):
                x_train_keep, y_train_keep = sx_train, sy_train
            else:
                x_train_keep, y_train_keep = sx_train[keep_idxs], sy_train[keep_idxs]

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            knn_resnet = copy.deepcopy(self.model)
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

        print(" knn acc:", len(keep_idxs), knn_accs)
        return knn_accs

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
                outputs = model(inputs)
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
