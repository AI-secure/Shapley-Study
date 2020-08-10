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
from shapley.models import MNL

class Adaptation(App):
    def __init__(self, X, y, X_test, y_test, model_name='MNL', ori_dataset="mnist", dst_dataset="usps"):
        self.name = 'Adaptation'
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.num_train = 2500
        self.num_test = len(X_test)
        self.num_pre = 95000
        self.model_name = model_name

        self.model = getattr(models, model_name)(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 200)

    def run(self, measure, sx_train, sy_train, sx_test, sy_test, x_ratio=0.2, batch_size=128, epochs=15, HtoL=False):
        knn_value = measure.score(self.X, self.y, self.X_test, self.y_test, model=return_model(self.model_name).cuda())

        filted_embed_knn_sv_idxs = np.where(knn_value >= -0.0)[0]
        device = torch.device('cuda')
        model = MNL(input_size, num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        x_train = torch.from_numpy(sx_train[filted_embed_knn_sv_idxs]).contiguous().view(-1, 1, 28, 28)
        y_train = torch.from_numpy(sy_train[filted_embed_knn_sv_idxs]).view(-1,).long()

        x_test = torch.from_numpy(sx_test[:val_num]).contiguous().view(-1, 1, 28, 28)
        y_test = torch.from_numpy(sy_test[:val_num]).view(-1,).long()
        print(x_train.shape, y_train.shape)

        train(model, device, x_train, y_train, batch_size, optimizer, criterion, epochs)
        accuracy, avg_loss = evaluate(model, device, x_test, y_test, batch_size, criterion)
        print("k=",measure.k," epochs=",epochs)
        print(f'[Test] Accuracy: {100 * accuracy:5.2f}%, loss: {avg_loss:7.4f}')
        return knn_accs

def train(model, device, x_train, y_train, batch_size, optimizer, criterion, n_epochs):
    model.train()
    for epoch in tqdm_notebook(range(n_epochs), desc = 'Epochs'):
#         print("epoch model.fc.weight:")
#         print(epoch, model.fc.weight)
        for X, y in batch(x_train, y_train, batch_size):
            X, y = X.to(device).float(), y.to(device)
#             print(X.shape, y.shape)
            optimizer.zero_grad()
#             y_pred = model(X)
            *_, y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
#             for param in model.parameters():
#                 print(param.grad.data.sum())
            optimizer.step()
#         if(n_epochs > 4):
#             if(epoch % int(n_epochs/4) == 0):
#                 print(f'Train epoch {epoch}: Loss: {loss.item():7.4f}')

def evaluate(model, device, x_test, y_test, batch_size, criterion):
    model.eval()
    test_set_size = len(x_test)
    correct_answers = 0
    sum_loss = 0
    with torch.no_grad():
        for X, y in batch(x_test, y_test, batch_size):
            X, y = X.to(device).float(), y.to(device)
            *_, y_pred = model(X)
#             y_pred = model(X)

            class_pred = y_pred.argmax(dim = 1)
            correct_answers += (y == class_pred).float().sum().item()
            sum_loss += criterion(y_pred, y).item()
    accuracy = correct_answers / test_set_size
    average_loss = sum_loss / len(x_test)

    return accuracy, average_loss
