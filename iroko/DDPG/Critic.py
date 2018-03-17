import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state = 54, actions = 18, hidden1=100, hidden2 = 30):
        super(Critic, self).__init__()
        #paper claims 1st layer they do not put action through 1st layer
        self.normalize = nn.BatchNorm1d(state)
        self.normalize2 = nn.BatchNorm1d(hidden1 + actions)
        self.hidden1 = nn.Linear(state, hidden1)
        self.hidden2 = nn.Linear(hidden1 + actions, hidden2)
        self.outputs = nn.Linear(hidden2, 1)

        nn.init.xavier_normal(self.hidden1.weight.data, gain=2)
        nn.init.xavier_normal(self.hidden2.weight.data, gain=2)
        nn.init.xavier_normal(self.outputs.weight.data)



    def forward(self, state, action):
        #x = self.normalize(state)
        x = F.relu(self.hidden1(state))
        x = torch.cat((x, action),dim=-1)
        #x = self.normalize2(torch.cat(x, action))
        x = F.relu(self.hidden2(x))
        return self.outputs(x)
