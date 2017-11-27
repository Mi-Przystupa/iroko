import torch.nn 
from  torch.autograd import Variable
import torch
import math
import random
import numpy as np
class LearningController:
    def __init__(self, inputs, actions, numNeuron1, numNeuron2, alpha, gamma, epsilon = .9, decay=.001, filepath = None):
        self.model = torch.nn.Sequential(
                torch.nn.Linear(inputs + actions, numNeuron1, True ), 
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(numNeuron1, numNeuron2, True),
                torch.nn.ReLU(),
                torch.nn.Linear(numNeuron2, 1))
        self.alpha = alpha 
        self.gamma = gamma
        self.inputlength = inputs
        self.actionlength = actions
        self.prevAction = torch.zeros(actions);
        self.prevState = torch.zeros(inputs);
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 1e-6)
        self.epsilon = epsilon
        self.decay = decay
        if (filepath != None):
            self.model.load_state_dict(torch.load(filepath))



    def PerformAction(self):
        if(random.random() < self.epsilon):
            # makes sure minimum is 0
            self.epsilon = max(.05, self.epsilon - self.decay)
            action = np.random.random_integers(0, self.actionlength - 1) 
            ret = torch.zeros(self.actionlength)
            ret[action] = 1.0
            return ret 
        return self.ReturnBestAction() 

    def ReturnBestAction(self):
        bAction = torch.zeros(self.actionlength)
        
        bVal = -1000
        vals = []
        for i in range(0, self.actionlength):
             
            action = torch.zeros(self.actionlength) 
            action[i] = 1
            state = Variable(torch.cat((self.prevState, action.float()), 0))  
            curVal = self.model(state);
            vals.append(curVal.data[0])
            if(curVal.data[0] > bVal ):
                bVal = curVal.data[0]
                bAction = action
        return bAction 

    def UpdateValueFunction(self, inputs, actions, reward):
        self.model.zero_grad()
        state = Variable(torch.cat((inputs, actions), 0))
        stateprev = Variable(torch.cat((self.prevState, self.prevAction), 0))
        #Qprev = self.model.forward(stateprev)
        QCurr = self.model(state)
        Qprev = self.model(stateprev)
        criterion = torch.nn.MSELoss()
        QCurr = reward + self.gamma * QCurr.data
        QCurr = Variable(QCurr)
        loss = criterion(Qprev, QCurr )
        loss.backward()
        self.optimizer.step()
        self.prevState = inputs
        self.prevAction = actions

        return self.model(Variable(torch.zeros(self.inputlength + self.actionlength))).data[0] 
     
    def saveNetwork(self):
       torch.save(self.model.state_dict(), './modelconfig') 

