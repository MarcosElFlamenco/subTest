import torch
import torch.nn as nn
import torch.nn.functional as F


class Sub_model(nn.Module):
    def __init__(self):
        super(Sub_model, self).__init__()
        self.fc1 = nn.Linear(2 , 10)
        self.fc2 = nn.Linear(10 , 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
