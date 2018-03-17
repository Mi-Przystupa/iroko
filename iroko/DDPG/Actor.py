import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state = 54, actions = 18,  hidden1=81, hidden2= 40, useSigmoid=False):
        super(Actor, self).__init__()
        self.normalize = nn.BatchNorm1d(state)
        self.normalize2 = nn.BatchNorm1d(hidden1)
        self.hidden1 = nn.Linear(state, hidden1)  
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.outputs = nn.Linear(hidden2, actions)

        nn.init.xavier_normal(self.hidden1.weight.data, gain=2)
        nn.init.xavier_normal(self.hidden2.weight.data, gain=2)
        nn.init.xavier_normal(self.outputs.weight.data)


    def forward(self,  state):
        #x = self.normalize(state)
        x = F.relu(self.hidden1(state))
        #x = self.normalize2(x)
        x = F.relu(self.hidden2(x))
        if (self.useSigmoid):
            return F.sigmoid(self.outputs(x))
        else:
            return F.tanh( self.outputs(x))
