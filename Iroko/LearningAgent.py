import torch
import torch.nn
from  torch.autograd import Variable
import random
import numpy as np

class SARSA:
    def __init__(self, inputs, actions, numNeuron1, numNeuron2, alpha, gamma, epsilon = .9, decay=.001, filepath = None):
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
        state = Variable(torch.cat((self.prevState, bAction),0))
        
        bVal = self.model(state)
        for i in range(0, self.actionlength):
             
            action = torch.zeros(self.actionlength) 
            action[i] = 1
            state = Variable(torch.cat((self.prevState, action.float()), 0))  
            curVal = self.model(state);
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
        print(Qcurr)
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
     
    def saveNetwork(self):
       torch.save(self.model.state_dict(), './modelconfig') 


class CircularList:
    def __init__(self, capacity = 15):
        self.memory = []
        self.indx = 0
        self.capacity = capacity
    def push(self, data):
        if (len(self.memory) < self.capacity):
            self.memory.append(None)
        self.memory[self.indx] = data
        self.indx = (self.indx + 1) % self.capacity

    def sample(self):
        if(len(self.memory) > 0):
            return random.sample(self.memory, 1)
        else:
            return 0

    def __len__(self):
        return len(self.memory)

class LearningAgent:
    def __init__(self,alpha = .01, gamma = .9, l = .1,  capacity= 15, globalBW = 0, defaultmax = 0):
        self.memory = CircularList(capacity)
        self.hosts = {}
        self.globalBandWidth = globalBW
        self.alpha = alpha
        self.gamma = gamma
        self.lam = l
        self.defaultmax = defaultmax 
        #inputs, actions, numNeuron1, numNeuron2, alpha, gamma, epsilon = .9, decay=.001, filepath = None
        # five actions 50% decrease, 25% decrease, 0 25% increase, 50% increase
        self.SARSALearning = SARSA(10, 3, 15, 20 , alpha, gamma, epsilon = .9, decay=.001);

    
    def addMemory(self, data):
        self.memory.push(data)

    def getSample(self):
        return self.memory.sample()
    
    def updateHostsBandwidth(self, dpid, port_no, freebandwidth):
        key = str(dpid) + "-" + str(port_no)
        if(not (key in self.hosts.keys())):
            self.hosts[key] = dict(alloctBandwidth = self.defaultmax, e =  0 )
        else:
            currhost = self.hosts[key]
            if(freebandwidth <= 0.0):
                R = currhost['alloctBandWidth'] * .1
            else:
                R = 0
            #assumes freebandwidth = allocatedbandwidth - used bandwidth
            sprime = currhost['alloctBandwidth'] - freebandwidth
            #V(s) = V(s) + alpha*e * (R * gamma* V(s') - V(s))
            delta = R + self.gamma *  sprime - currhost['alloctBandwidth'] 

            currhost['e'] += 1
            currhost['alloctBandwidth'] += self.alpha * delta * currhost['e']
            currhost['prevFreeBandwidth'] = freebandwidth

            currhost['e'] = self.gamma* self.lam * currhost['e']
            currhost['alloctBandwidth'] = max(0, currhost['alloctBandwidth'])

            #print(currhost['alloctBandwidth'])
            self.hosts[key] = currhost


    def displayAllHosts(self):
        allnames = ""
        for host in self.hosts.keys():
            allnames = allnames + " " + str(host) 
        print(allnames)
    def displayALLHostsBandwidths(self):
        allbandwidths = ""
        for host in self.hosts.keys():
            allbandwidths = allbandwidths + " " + str(self.hosts[host]['alloctBandwidth'])
        print(allbandwidths)
