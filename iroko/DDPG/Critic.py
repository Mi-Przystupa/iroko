import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state = 54, actions = 18, hidden1=100, hidden2 = 30):
        super(Critic, self).__init__()
        self.hidden1 = nn.Linear(state + actions, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.outputs = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = F.elu(self.hidden1(x))
        x = F.elu(self.hidden2(x))
        return self.outputs(x)
 
