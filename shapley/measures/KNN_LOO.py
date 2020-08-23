import numpy as np
from sklearn.neural_network._base import ACTIVATIONS

import torch
import torch.nn.functional as F

from shapley.measures import Measure

class KNN_LOO(Measure):

    def __init__(self, K=10):
        self.name = 'KNN_LOO'
        self.K = K

    def score(self, X_train, y_train, X_test, y_test, model_family='', model=None):
        N = X_train.shape[0]
        M = X_test.shape[0]

        self.model = model

        if model_family == "ResNet":
            resnet = model
            X_out1 = resnet.layer1(F.relu(resnet.bn1(resnet.conv1(torch.from_numpy(X_train)))))
            X_test_out1 = resnet.layer1(F.relu(resnet.bn1(resnet.conv1(torch.from_numpy(X_test))))) # 64, 32, 32
            X_out2 = resnet.layer2(X_out1)
            X_test_out2 = resnet.layer2(X_test_out1) # 64, 32, 32
            X_out3 = resnet.layer3(X_out2)
            X_test_out3 = resnet.layer3(X_test_out2) # 64, 32, 32
            value = np.zeros(N)
            for i in range(M):
                X = X_test_out1[i]
                y = y_test[i]
                s = np.zeros(N)
                dist = []
                diff = (X_out1.detach().numpy() - X.detach().numpy()).reshape(N, -1)
                dist = np.einsum('ij, ij->i', diff, diff)
                idx = np.argsort(dist)
                ans = y_train[idx]
                s[idx[N - 1]] = float(ans[N - 1] == y) / N
                cur = N - 2
                for j in range(N):
                    if j in idx[:self.K]:
                        s[j] = float(int(ans[j] == y) - int(ans[self.K] == y)) / self.K
                    else:
                        s[j] = 0
                for i in range(N):
                    value[j] += s[j]  
            for i in range(N):
                value[i] /= M  
            return value

        if model_family == "NN":
            nn = model
            X_feature = ACTIVATIONS['relu'](np.matmul(X_train, nn.coefs_[0]) + nn.intercepts_[0])
            X_test_feature = ACTIVATIONS['relu'](np.matmul(X_test, nn.coefs_[0]) + nn.intercepts_[0])
            value = np.zeros(N)
            for i in range(M):
                X = X_test_feature[i]
                y = y_test[i]
                s = np.zeros(N)
                dist = []
                diff = (X_feature - X).reshape(N, -1)
                dist = np.einsum('ij, ij->i', diff, diff)
                idx = np.argsort(dist)
                ans = y_train[idx]
                s[idx[N - 1]] = float(ans[N - 1] == y) / N
                cur = N - 2
                for j in range(N):
                    if j in idx[:self.K]:
                        s[j] = float(int(ans[j] == y) - int(ans[self.K] == y)) / self.K
                    else:
                        s[j] = 0
                for i in range(N):
                    value[j] += s[j]  
            for i in range(N):
                value[i] /= M  
            return value

        value = np.zeros(N)
        for i in range(M):
            X = X_test[i]
            y = y_test[i]

            s = np.zeros(N)
            dist = []
            diff = (X_train - X).reshape(N, -1)
            dist = np.einsum('ij, ij->i', diff, diff)
            idx = np.argsort(dist)

            ans = y_train[idx]

            s[idx[N - 1]] = float(ans[N - 1] == y) / N

            cur = N - 2
            for j in range(N):
                if j in idx[:self.K]:
                    s[j] = float(int(ans[j] == y) - int(ans[self.K] == y)) / self.K
                else:
                    s[j] = 0
                
            for i in range(N):
                value[j] += s[j]  

        for i in range(N):
            value[i] /= M
        return value