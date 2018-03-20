import torch
import torch.nn
from torch.autograd import Variable
import random
import numpy as np
import math
from DDPG.DDPGConv import DDPGConv


class LearningAgentv3:
    def __init__(self, gamma=.99, lam=.1, memory=1000, initMax=0, defaultmax=10e6, cpath=None, apath=None, toExploit=False):
        self.hosts = {}
        self.hostcount = 0
        self.gamma = gamma
        self.lam = lam
        self.defaultmax = defaultmax
        self.initmax = initMax
        self.controller = DDPGConv(gamma, memory, 96, 16, tau=.001, criticpath=cpath, actorpath=apath, useSig=True)
        # criticpath='critic', actorpath='actor', useSig=True)
        if(toExploit):
            self.controller.exploit()
        else:
            self.controller.explore()
        #lol can you say hacking??
        self.runningState = []
        self.prevState = []
        self.frame = 0
        self.frameFill = 0
        self.totalReward = 0
        self.actionVector = []

    def initializePorts(self, ports, features=5):
        # cleans out old ports and creates maps for new ones
        self.hosts = {}
        for key in ports.keys():
            self.hosts[key] = self.createHost()
        #to make it multidimensional
        self.runningState = torch.Tensor(3,self.hostcount, features) 
        self.prevState = torch.Tensor(3, self.hostcount, features)
        self.actionVector = torch.Tensor(self.hostcount)

    def createHost(self):
        self.hostcount += 1
        host = dict(alloctBandwidth=self.initmax, e=0, predictedAllocation=self.initmax)
        # host['controller'] = self.generateController()
        host['action'] = 0.0
        # lol, what is good data abstraction really though?
        #host['state'] = torch.zeros(self.controller.state)
        # host['modifier'] = 0
        host['id'] = self.hostcount - 1
        return host

    def update(self, interface, data, reward):
        # assumes data is a pytorch.tensor
        self.prevState[(self.frame -1) % 3][self.hosts[interface]['id']] = \
                self.runningState[self.frame  % 3][self.hosts[interface]['id']] 
        self.runningState[self.frame][self.hosts[interface]['id']] = data
        self.actionVector[self.hosts[interface]['id']] = self.hosts[interface]['action']
        self.frameFill += 1

        self.totalReward += reward
        #
        if (self.frameFill == self.hostcount):
            state = self.prevState
            stateprime = self.runningState
            action = self.actionVector
            reward = torch.Tensor([self.totalReward])
            self.controller.addToMemory(state, action, reward, stateprime)
            self.frameFill = 0
            self.frame = (self.frame + 1) % 3

        if (self.controller.primedToLearn()):
            self.controller.PerformUpdate(64)
            self.controller.UpdateTargetNetworks()
            self.controller.saveActorCritic()

    def predictBandwidthOnHosts(self):
        state = self.prevState 
        action = self.controller.selectAction(state.unsqueeze(0))
        action = action.squeeze()

        for interface in self.hosts.keys():
            a = action[self.hosts[interface]['id']]
            a = np.clip(a, 0.0, 1.0)
            self.hosts[interface]['predictedAllocation'] = self.defaultmax * a
            self.hosts[interface]['action'] = a



    def getHostsPredictedBandwidth(self, interface):
        return int(math.ceil(self.hosts[interface]['predictedAllocation']))

    def displayAllHosts(self):
        allnames = ""
        for host in self.hosts.keys():
            allnames = allnames + " " + str(host)
        # print(allnames)

    def displayALLHostsBandwidths(self):
        allbandwidths = ""
        for host in self.hosts.keys():
            allbandwidths = allbandwidths + " " + str(self.hosts[host]['alloctBandwidth'])
        # print("Allocated Bandwidth: " + allbandwidths)

    def displayALLHostsPredictedBandwidths(self):
        allbandwidths = ""
        for host in self.hosts.keys():
            allbandwidths = allbandwidths + " " + str(self.hosts[host]['predictedAllocation'])
        # print("Predict Bandwidth: " + allbandwidths)

    def displayAdjustments(self):
        allbandwidths = ""
        for host in self.hosts.keys():
            allbandwidths = allbandwidths + " " + str(self.hosts[host]['action'])
        # print("Bandwidth Adjustment: " + allbandwidths)
