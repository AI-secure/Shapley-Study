import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from pytorch_fitmodule import FitModule

# class CNN(nn.Module):
#     #train mnist
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1  , 16 , 3, padding = 1)
#         self.conv2 = nn.Conv2d(16 , 32, 3, padding = 1)
#         self.fc1   = nn.Linear(32 * 7 * 7, 32)
#         self.fc2   = nn.Linear(32        , 10)
        
#     def forward(self, x):
#         batch_size    = x.size(0)
#         out_conv1     = F.relu(self.conv1(x))
#         out_max_pool1 = F.max_pool2d(out_conv1, kernel_size = (2, 2))
#         out_conv2     = F.relu(self.conv2(out_max_pool1))
#         out_max_pool2 = F.max_pool2d(out_conv2, kernel_size = (2, 2))
#         out_view      = out_max_pool2.view(-1, 32 * 7 * 7)
#         out_fc1       = F.relu(self.fc1(out_view))
#         out_fc2       = self.fc2(out_fc1)        
#         return out_fc1, out_fc2

class CNN_svhn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x))
        x = self.fc3(x1)
        return x1, x

    
class MultiCNN(nn.Module):
    #train mnist
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1  , 32 , 3, padding = 1)
        self.conv2 = nn.Conv2d(32 , 64, 3, padding = 1)
        self.fc1   = nn.Linear(64 * 7 * 7, 64)
        self.fc2   = nn.Linear(64        , 10)
        
    def forward(self, x):
        batch_size    = x.size(0)
        out_conv1     = F.relu(self.conv1(x))
        out_max_pool1 = F.max_pool2d(out_conv1, kernel_size = (2, 2))
        out_conv2     = F.relu(self.conv2(out_max_pool1))
        out_max_pool2 = F.max_pool2d(out_conv2, kernel_size = (2, 2))
        out_view      = out_max_pool2.view(-1, 64 * 7 * 7)
        out_fc1       = F.relu(self.fc1(out_view))
        out_fc2       = self.fc2(out_fc1)        
        return out_fc1, out_fc2    

class plotCNN(nn.Module):
    #train mnist
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1  , 32 , 3, padding = 1)
        self.conv2 = nn.Conv2d(32 , 64, 3, padding = 1)
        self.fc1   = nn.Linear(64 * 7 * 7, 64)
        self.fc2   = nn.Linear(64        , 10)
        
    def forward(self, x):
        batch_size    = x.size(0)
        out_conv1     = F.relu(self.conv1(x))
        out_max_pool1 = F.max_pool2d(out_conv1, kernel_size = (2, 2))
        out_conv2     = F.relu(self.conv2(out_max_pool1))
        out_max_pool2 = F.max_pool2d(out_conv2, kernel_size = (2, 2))
        out_view      = out_max_pool2.view(-1, 64 * 7 * 7)
        out_fc1       = F.relu(self.fc1(out_view))
        out_fc2       = self.fc2(out_fc1)        
        return out_fc2      
    
    
    
class CNN_Cifar10(nn.Module):
    # train cifar10: performance is not good
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3 , 16 , 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 16 , 3, padding = 1)
        self.conv3 = nn.Conv2d(16, 32,  3, padding = 1)
        self.conv4 = nn.Conv2d(32, 32,  3, padding = 1)
        self.fc1   = nn.Linear(32 * 8 * 8, 128)
        self.fc2   = nn.Linear(128       , 10)
        
    def forward(self, x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.2, training = self.training)
        out_conv2 = F.dropout(F.relu(self.conv2(out_conv1)), 0.2, training = self.training)
        out_pool1 = F.max_pool2d(out_conv2, kernel_size = (2, 2))
        out_conv3 = F.dropout(F.relu(self.conv3(out_pool1)), 0.2, training = self.training)
        out_conv4 = F.dropout(F.relu(self.conv4(out_conv3)), 0.2, training = self.training)
        out_pool2 = F.max_pool2d(out_conv4, kernel_size = (2, 2))
        out_view  = out_pool2.view(-1, 32 * 8 * 8)
        out_fc    = F.dropout(F.relu(self.fc1(out_view)), 0.2, training = self.training)
        out       = self.fc2(out_fc)
        
        return out_conv4, out
