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
from sklearn.ensemble import RandomForestRegressor

class Acquisition(App):
    def __init__(self, X, y, X_test, y_test, model_name='resnet18'):
        self.name = 'Acquisition'

        self.X_test = X_test
        self.y_test = y_test
        self.num_train = 2500
        self.num_test = len(X_test)
        self.num_pre = 95000

        self.X = X[:self.num_train]
        self.y = y[:self.num_train]
        self.X_pre = X[-self.num_pre:]
        self.y_pre = y[-self.num_pre:]
        self.model_name = model_name

        self.model = getattr(models, model_name)(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 200)

    def run(self, measure, sx_train, sy_train, sx_test, sy_test, x_ratio=0.2, batch_size=128, epochs=15, HtoL=False):
        knn_value = measure.score(self.X, self.y, self.X_test, self.y_test, model=return_model(self.model_name).cuda())

        # train random forest for embeddings
        filted_embed_knn_sv_idxs = np.where(knn_value >= 0.0)[0]
        knn_value = knn_value / np.linalg.norm(knn_value)

        random_forest =  RandomForestRegressor(max_depth=100, n_estimators=50, random_state=666)
        random_forest.fit(self.X[filted_embed_knn_sv_idxs][:,:,0,0], knn_value[filted_embed_knn_sv_idxs])

        knn_pre_scores = random_forest.predict(self.X_pre[:,:, 0, 0])

        sx_pre = torch.from_numpy(sx_train[-self.num_pre:]).contiguous().view(-1, 3,64,64)
        sy_pre = torch.from_numpy(sy_train[-self.num_pre:]).view(-1,).long()
        sx_train = torch.from_numpy(sx_train[:self.num_train]).contiguous().view(-1, 3,64,64)
        sy_train = torch.from_numpy(sy_train[:self.num_train]).view(-1,).long()
        sx_test = torch.from_numpy(sx_test[:self.num_train + self.num_train]).contiguous().view(-1, 3,64,64)
        sy_test = torch.from_numpy(sy_test[:self.num_train + self.num_train]).view(-1,).long()


        print(self.model_name, " start")
        if(HtoL == True):
            print("adding data from Highest to Lowest!")
        else:
            print("adding data from Lowest to Highest!")
        accs = []
        count = int(len(sx_pre))
        interval = int(count * x_ratio)
        idxs = np.argsort(knn_pre_scores)
        keep_idxs = idxs.tolist()

        for j in range(0, count, interval):
            x_train_keep = torch.cat((sx_train, sx_pre[keep_idxs[:j]]), 0)
            y_train_keep = torch.cat((sy_train, sy_pre[keep_idxs[:j]]), 0)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            knn_resnet = copy.deepcopy(self.model)
            knn_resnet = knn_resnet.to(device)
            optimizer = optim.SGD(knn_resnet.parameters(), lr=0.005, momentum=0.9)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            criterion = nn.CrossEntropyLoss()

            plot_tiny_train("train", knn_resnet, device, x_train_keep, y_train_keep, optimizer, criterion, scheduler, batch_size, epochs)
            acc, loss = plot_tiny_train("val", knn_resnet, device, sx_test, sy_test, optimizer, criterion, scheduler, batch_size)
            acc = acc.cpu().detach().numpy()
            accs.append(acc*100.0)
            print(j, acc)
        print(self.name, " acc:", accs)
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

