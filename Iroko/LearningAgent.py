import torch
import torch.nn
from torch.autograd import Variable
import random
import numpy as np
import math


class SARSA:
    def __init__(self, inputs, actions, numNeuron1, numNeuron2, alpha, gamma, epsilon=.9, decay=.001, filepath=None):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(inputs + actions, numNeuron1, True),
            torch.nn.ReLU(),
            torch.nn.Linear(numNeuron1, numNeuron2, True),
            torch.nn.ReLU(),
            torch.nn.Linear(numNeuron2, 1))
        self.alpha = alpha
        self.gamma = gamma
        self.inputlength = inputs
        self.actionlength = actions
        self.prevAction = torch.zeros(actions)
        self.prevState = torch.zeros(inputs)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-6)
        self.epsilon = epsilon
        self.decay = decay
        if (filepath is not None):
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
        bAction[0] = 1.0
        bVal = -1000
        for i in range(0, self.actionlength):
            action = torch.zeros(self.actionlength)
            action[i] = 1
            state = Variable(torch.cat((self.prevState, action.float()), 0))
            curVal = self.model(state)
            if(curVal.data[0] > bVal):
                bVal = curVal.data[0]
                bAction = action

        return bAction

    def UpdateValueFunction(self, inputs, actions, reward):
        self.model.zero_grad()
        state = Variable(torch.cat((inputs, actions), 0))
        stateprev = Variable(torch.cat((self.prevState, self.prevAction), 0))

        Qcurr = self.model(state)
        Qprev = self.model(stateprev)
        Qupdate = Qprev.data + self.alpha * (reward + self.gamma * Qcurr.data - Qprev.data)

        Qcurr = reward + self.gamma * Qcurr.data
        Qcurr = Variable(Qcurr)
        criterion = torch.nn.MSELoss()
        loss = criterion(Qprev, Qcurr)
        loss.backward()
        self.optimizer.step()
        self.prevState = inputs
        self.prevAction = actions
        return Qupdate

    def saveNetwork(self):
        torch.save(self.model.state_dict(), './modelconfig')


class Actor:
    def __init__(self, inputs, numNeuron1, numNeuron2, learningRate=1e-6, epsilon=.5, filepath=None):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(inputs, numNeuron1, True),
            torch.nn.ReLU(),
            torch.nn.Linear(numNeuron1, numNeuron2, True),
            torch.nn.ReLU(),
            torch.nn.Linear(numNeuron2, 1),
            torch.nn.Tanh())
        self.epsilon = epsilon
        self.inputlength = inputs
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learningRate)
        self.state = torch.zeros(inputs)
        if (filepath is not None):
            self.model.load_state_dict(torch.load(filepath))

    def takeAction(self, state, asvector, actiontoPerform):
        # actionToPerform is -1, 0, or 1
        if (random.random() < self.epsilon):
            action = random.random() * actiontoPerform
            return action
        return self.model(Variable(torch.cat((state, asvector), 0))).data[0]

    def updateActions(self, state, actionVector, action, reward):
        if(reward >= 0):
            self.model.zero_grad()
            state = Variable(torch.cat((state, actionVector), 0))
            currAction = self.model(state)
            label = Variable(torch.Tensor([action]))
            criterion = torch.nn.MSELoss()
            loss = criterion(currAction, label)
            loss.backward()
            self.optimizer.step()
        # do nothing


class CircularList:
    def __init__(self, capacity=15):
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
    def __init__(self,alpha = 1e-6, gamma = .9, lam = .1,  capacity= 15, initMax = 0, defaultmax = 10e6):
        self.memory = CircularList(capacity)
        self.hosts = {}
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.defaultmax = defaultmax
        self.initmax = initMax
        # five actions 50% decrease, 25% decrease, 0 25% increase, 50% increase

        self.Actor = Actor(8, 20, 10, learningRate=alpha, epsilon=.5, filepath=None)

    def initializePorts(self, ports):
        # cleans out old ports and creates maps for new ones
        self.hosts = {}
        for key in ports.keys():
            self.hosts[key] = self.createHost()

    def createHost(self):
        host = dict(alloctBandwidth = self.initmax, e =  0, predictedAllocation = self.initmax)
        host['controller'] = self.generateController() 
        host['action'] = torch.zeros(3);
        host['modifier'] = 0
        return host

    def generateController(self):
        return SARSA(5, 3, 15, 20, self.alpha, self.gamma, epsilon=.5, decay=.001)

    def addMemory(self, data):
        self.memory.push(data)

    def getSample(self):
        return self.memory.sample()

    def updateActor(self, interface, reward):
        host = self.hosts[interface]
        self.Actor.updateActions(host['controller'].prevState, host['action'], host['modifier'], reward)

    def updateCritic(self, interface, data, reward):
        # assumes data is a pytorch.tensor
        data = data
        action = self.hosts[interface]['action']
        self.hosts[interface]['controller'].UpdateValueFunction(data, action, reward)

    def predictBandwidthOnHosts(self):
        for host in self.hosts.keys():
            self.predictBandwidthOnHost(host)

    def predictBandwidthOnHost(self, interface):
        hostsAction = self.hosts[interface]['controller'].PerformAction()
        a = 0
        if(hostsAction[0] > 0):
            a = 0
        elif (hostsAction[1] > 0):
            a = 1
        elif (hostsAction[2] > 0):
            a = -1
        else:
            print("Something is wrong")

        adjustment = self.Actor.takeAction(self.hosts[interface]['controller'].prevState, hostsAction, a)
        adjustment = max(min(adjustment, .90), -0.20)
        self.hosts[interface]['predictedAllocation'] += self.hosts[interface]['predictedAllocation'] * adjustment
        self.hosts[interface]['predictedAllocation'] = max(0, self.hosts[interface]['predictedAllocation'])
        self.hosts[interface]['predictedAllocation'] = min(self.hosts[interface]['predictedAllocation'], self.defaultmax)
        self.hosts[interface]['action'] = hostsAction
        self.hosts[interface]['modifier'] = adjustment

    def getHostsBandwidth(self, interface):
        return int(math.ceil(self.hosts[interface]['alloctBandwidth']))

    def getHostsPredictedBandwidth(self, interface):
        return int(math.ceil(self.hosts[interface]['predictedAllocation']))

    def updateHostsBandwidth(self, interface, freebandwidth):
        key = str(interface)
        if(not (key in self.hosts.keys())):
            self.hosts[key] = self.createHost()
            print("New port added: " + key)
        else:
            currhost = self.hosts[key]
            if(freebandwidth <= 0.0):
                R = currhost['alloctBandwidth'] * .15
            else:
                R =  -(freebandwidth * 0.15) 
            #assumes freebandwidth = allocatedbandwidth - used bandwidth
            sTrue = currhost['alloctBandwidth'] - freebandwidth
            #V(s) = V(s) + alpha*e * (R + gamma* V(s') - V(s))
            delta = R + self.gamma *  currhost['alloctBandwidth'] - sTrue

            currhost['e'] += 1

            currhost['alloctBandwidth'] = sTrue +   delta * currhost['e']
            currhost['e'] = self.gamma* self.lam * currhost['e']
            currhost['alloctBandwidth'] = min( max(self.defaultmax / 100, currhost['alloctBandwidth']), self.defaultmax)

            self.hosts[key] = currhost

    def displayAllHosts(self):
        allnames = ""
        for host in self.hosts.keys():
            allnames = allnames + " " + str(host)
        # print(allnames)

    def displayALLHostsBandwidths(self):
        allbandwidths = ""
        for host in self.hosts.keys():
            allbandwidths = allbandwidths + " " + str(self.hosts[host]['alloctBandwidth'])
        print("Allocated Bandwidth: " + allbandwidths)

    def displayALLHostsPredictedBandwidths(self):
        allbandwidths = ""
        for host in self.hosts.keys():
            allbandwidths = allbandwidths + " " + str(self.hosts[host]['predictedAllocation'])
        print("Predict Bandwidth: " + allbandwidths)

    def displayAdjustments(self):
        allbandwidths = ""
        for host in self.hosts.keys():
            allbandwidths = allbandwidths + " " + str(self.hosts[host]['modifier'])
        print("Bandwidth Adjustment: " + allbandwidths)
