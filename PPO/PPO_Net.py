import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rl_utils




HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600




class ActorNet(nn.Module):
    def __init__(self, state_dim,action_dim):
        #action_dim = 3,意义为steering, acceleration, brake
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.mu = nn.Linear(HIDDEN2_UNITS, action_dim)
        nn.init.normal_(self.mu.weight, 0,1e-4)
        self.log_std = nn.Linear(HIDDEN2_UNITS, action_dim)
        nn.init.normal_(self.log_std.weight, 0,1e-4)
        

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        mu = torch.tanh(self.mu(x))
        log_std = F.softplus(self.log_std(x))
        std = torch.exp(log_std)
#        print("mu------------------",mu.shape)
        return mu, std
    

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.value = nn.Linear(HIDDEN2_UNITS, 1)
        nn.init.normal_(self.value.weight, 0,1e-4)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        value = self.value(x)
#        print("value------------------",value.shape)
        return value