import torch
import torch.nn
from torch.autograd import Variable
import random
import numpy as np
import math
from DDPG.DDPGConv import DDPGConv


# the convolution agent
class LearningAgentv3:
    def __init__(self, gamma=.99, lam=.1, s=2560, memory=1000, initMax=0, defaultmax=10e6, cpath=None, apath=None, toExploit=False, frames=3, w=4):
        self.hosts = {}
        self.hostcount = 0
        self.gamma = gamma
        self.lam = lam
        self.defaultmax = defaultmax
        self.initmax = initMax
        self.controller = DDPGConv(gamma, memory, s, 16, tau=.001, criticpath=cpath, actorpath=apath, useSig=True, w=w, f=frames)
        # criticpath='critic', actorpath='actor', useSig=True)
        if(toExploit):
            self.controller.exploit()
        else:
            self.controller.explore()
        # lol can you say hacking??
        self.state = []
        self.prevState = []
        self.actionVector = []
        self.frames = 0

    def initializePorts(self, ports):
        # cleans out old ports and creates maps for new ones
        self.hosts = {}
        for key in ports.keys():
            self.hosts[key] = self.createHost()
        self.actionVector = torch.Tensor(self.hostcount)

    def initializeTrafficMatrix(self, interfaceCount=16, features=5, frames=3):
        # inputs:
            # interfaceCount: number of rows in the traffic matrix
            # features: number of metrics per interface
            # frames: number of previous frames to track
        # to make input space multi-dimensional
        self.state = torch.zeros(frames, interfaceCount, features)
        self.prevState = torch.zeros(frames, interfaceCount, features)
        self.frames = frames

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
        self.prevState = self.state
        # push back previous frames
        if (self.frames > 1):
            # might have this backwards...
            # as long as it's working like a queue it should be fine...
            # that implicity velocity stuff is kinda hand wavy anyways.
            for f in range(0, self.frames - 1):
                self.state[f] = self.state[f + 1]

        self.state[self.frames - 1] = data
        # set action
        for i in interface:
            self.actionVector[self.hosts[i]['id']] = self.hosts[i]['action']

        self.controller.addToMemory(self.prevState, self.actionVector, reward, self.state)
        if (self.controller.primedToLearn()):
            self.controller.PerformUpdate(64)
            self.controller.UpdateTargetNetworks()
            self.controller.saveActorCritic()

    def predictBandwidthOnHosts(self):
        state = self.state
        action = self.controller.selectAction(state.unsqueeze(0))
        action = action.squeeze()

        for interface in self.hosts.keys():
            a = action[self.hosts[interface]['id']]
            a = np.clip(a, 0.1, 1.0)
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
