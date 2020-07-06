import numpy as np
from sklearn.neural_network._base import ACTIVATIONS

from shapley.measures import Measure

class KNN_Shapley(Measure):

    def __init__(self, K=10):
        self.name = 'KNN_Shapley'
        self.K = K

    def score(self, X_train, y_train, X_test, y_test, model_family='', model=None):
        N = X_train.shape[0]
        M = X_test.shape[0]

        if model_family == "ResNet":
            resnet = model
            X_out1 = resnet.layer1(F.relu(resnet.bn1(resnet.conv1(torch.from_numpy(X_train)))))
            X_test_out1 = resnet.layer1(F.relu(resnet.bn1(resnet.conv1(torch.from_numpy(X_test))))) # 64, 32, 32
            X_out2 = resnet.layer2(X_out1)
            X_test_out2 = resnet.layer2(X_test_out1) # 64, 32, 32
            X_out3 = resnet.layer3(X_out2)
            X_test_out3 = resnet.layer3(X_test_out2) # 64, 32, 32
            s = np.zeros((N, M))
            for i in range(M):
                X = X_test_out1[i]
                y = y_test[i]
                dist = []
                diff = (X_out1.detach().numpy() - X.detach().numpy()).reshape(N, -1)
                dist = np.einsum('ij, ij->i', diff, diff)
                idx = np.argsort(dist)
                ans = y_train[idx]
                s[idx[N - 1]][i] = float(ans[N - 1] == y) / N
                cur = N - 2
                for j in range(N - 1):
                    s[idx[cur]][i] = s[idx[cur + 1]][i] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / self.K * (min(cur, self.K - 1) + 1) / (cur + 1)
                    cur -= 1
            return np.mean(s, axis=1)

        if model_family == "NN":
            nn = model
            X_feature = ACTIVATIONS['relu'](np.matmul(X_train, nn.coefs_[0]) + nn.intercepts_[0])
            X_test_feature = ACTIVATIONS['relu'](np.matmul(X_test, nn.coefs_[0]) + nn.intercepts_[0])
            s = np.zeros((N, M))
            for i in range(M):
                X = X_test_feature[i]
                y = y_test[i]
                dist = []
                diff = (X_feature - X).reshape(N, -1)
                dist = np.einsum('ij, ij->i', diff, diff)
                idx = np.argsort(dist)
                ans = y_train[idx]
                s[idx[N - 1]][i] = float(ans[N - 1] == y) / N
                cur = N - 2
                for j in range(N - 1):
                    s[idx[cur]][i] = s[idx[cur + 1]][i] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / self.K * (min(cur, self.K - 1) + 1) / (cur + 1)
                    cur -= 1  
            return np.mean(s, axis=1)

        value = np.zeros(N)
        s = np.zeros((N, M))
        for i in range(M):
            X = X_test[i]
            y = y_test[i]

            dist = []
            diff = (X_train - X).reshape(N, -1)
            dist = np.einsum('ij, ij->i', diff, diff)
            idx = np.argsort(dist)

            ans = y_train[idx]

            s[idx[N - 1]][i] = float(ans[N - 1] == y) / N

            cur = N - 2
            for j in range(N - 1):
                s[idx[cur]][i] = s[idx[cur + 1]][i] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / self.K * (min(cur, self.K - 1) + 1) / (cur + 1)
                cur -= 1 

            return np.mean(s, axis=1)
