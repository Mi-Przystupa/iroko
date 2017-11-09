import torch.nn 
from  torch.autograd import Variable
import torch


class LearningController:
    def __init__(self, inputs, actions, numNeuron1, numNeuron2, alpha, gamma):
        self.model = torch.nn.Sequential(
                torch.nn.Linear(inputs + actions, numNeuron1, True ), 
                torch.nn.ReLU(),
                torch.nn.Linear(numNeuron1, numNeuron2, True),
                torch.nn.ReLU(),
                torch.nn.Linear(numNeuron2, 1))
        self.alpha = alpha 
        self.gamma = gamma
        self.prevAction = torch.zeros(actions);
        self.prevState = torch.zeros(inputs);
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.01)

    def PerformAction():

        return 0

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
        return Qupdate
     

