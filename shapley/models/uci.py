import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class UCI(nn.Module):
    #train uci
    def __init__(self):
        super().__init__()
#         print("xxx")
        self.fc1   = nn.Linear(254, 6)
        self.fc2   = nn.Linear(6, 100)
        self.fc3   = nn.Linear(100, 2)
        
    def forward(self, x):
        batch_size    = x.size(0)
        out_fc1       = torch.sigmoid(self.fc1(x))
        out_fc2       = torch.sigmoid(self.fc2(out_fc1))
        out_fc3       = self.fc3(out_fc2)
        return out_fc2, out_fc3