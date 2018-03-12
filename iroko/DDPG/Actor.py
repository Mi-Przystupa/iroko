import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state = 54, actions = 18,  hidden1=81, hidden2= 40):
        super(Actor, self).__init__()
        self.hidden1 = nn.Linear(state, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.outputs = nn.Linear(hidden2, actions)

    def forward(self, x):
        x = F.elu(self.hidden1(x))
        x = F.elu(self.hidden2(x))
        return F.hardtanh(self.outputs(x))
        
