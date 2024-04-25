import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal






HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600


# SAC算法中的策略网络
class ActorNet(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(ActorNet,self).__init__()
        self.fc1 = nn.Linear(state_dim,HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS,HIDDEN2_UNITS)
        self.mu = nn.Linear(HIDDEN2_UNITS,action_dim)
        self.std = nn.Linear(HIDDEN2_UNITS,action_dim)



    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        std = F.softplus(self.std(x))
        dist = Normal(mu,std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob -= torch.log(1-torch.tanh(action).pow(2)+1e-7)
        return action,log_prob
    






# SAC算法中的Q网络
class QNet(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(QNet,self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim,HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS,HIDDEN2_UNITS)
        self.fc3 = nn.Linear(HIDDEN2_UNITS,1)


    def forward(self,x,a):
        cat = torch.cat([x,a],dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
