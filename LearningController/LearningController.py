import torch.nn 
from  torch.autograd import Variable
import torch
import math
import random

class LearningController:
    def __init__(self, inputs, actions, numNeuron1, numNeuron2, alpha, gamma, epsilon = .9, decay=.001):
        self.model = torch.nn.Sequential(
                torch.nn.Linear(inputs + actions, numNeuron1, True ), 
                torch.nn.ReLU(),
                torch.nn.Linear(numNeuron1, numNeuron2, True),
                torch.nn.ReLU(),
                torch.nn.Linear(numNeuron2, 1))
        self.alpha = alpha 
        self.gamma = gamma
        self.inputlength = inputs
        self.actionlength = actions
        self.prevAction = torch.zeros(actions);
        self.prevState = torch.zeros(inputs);
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.01)
        self.epsilon = epsilon
        self.decay = decay

    def PerformAction(self):
        if(random.random() < self.epsilon):
            self.epsilon = max(0, self.epsilon - self.decay)
            return torch.rand( self.actionlength) >= .5 
        return self.ReturnBestAction() 

    def ReturnBestAction(self):
        bAction = torch.rand(self.actionlength)
        state = Variable(torch.cat((self.prevState, bAction),0))
        
        bVal = self.model(state)
        for i in range(0, int(math.pow(2, self.actionlength) / 8)):
            curVal = self.model(state); 

            action = torch.rand(self.actionlength) >= .5
            state = Variable(torch.cat((self.prevState, action.float()), 0))  
            if((curVal > bVal).data[0] ):
                curVal = bVal
                bAction = action

        return action

    def UpdateValueFunction(self, inputs, actions, reward):
        self.model.zero_grad()
        state = Variable(torch.cat((inputs, actions), 0))
        stateprev = Variable(torch.cat((self.prevState, self.prevAction), 0))
        #Qprev = self.model.forward(stateprev)
        Qcurr = self.model(state)
        Qprev = self.model(stateprev)
        Qupdate = Qprev.data + self.alpha * (reward + self.gamma*Qcurr.data - Qprev.data)
        Qupdate = Variable(Qupdate)
        criterion = torch.nn.MSELoss()
        loss = criterion(Qprev, Qupdate)
        loss.backward()
        self.optimizer.step()
        self.prevState = inputs
        self.prevAction = actions
        return Qupdate
     

