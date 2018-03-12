import torch
import torch.nn
from torch.autograd import Variable
import random
import numpy as np
import math
from DDPG.DDPG import DDPG


class LearningAgentv2:
    def __init__(self,gamma = .9, lam = .1,  memory= 100 ,  initMax = 0, defaultmax = 10e6):
        self.hosts = {}
        self.hostcount = 0
        self.gamma = gamma
        self.lam = lam
        self.defaultmax = defaultmax
        self.initmax = initMax
        self.controller = DDPG(gamma,memory,5 + 16, 1,tau=.25, criticpath='critic', actorpath='actor') 
        


    def initializePorts(self, ports):
        # cleans out old ports and creates maps for new ones
        self.hosts = {}
        for key in ports.keys():
            self.hosts[key] = self.createHost()

    def createHost(self):
        self.hostcount += 1
        host = dict(alloctBandwidth = self.initmax, e =  0, predictedAllocation = self.initmax)
        #host['controller'] = self.generateController() 
        host['action'] = 0.0 
        #lol, what is good data abstraction really though? 
        host['state'] = torch.zeros(self.controller.state)
        #host['modifier'] = 0
        host['id'] = self.hostcount - 1
        return host

    def update(self, interface, data, reward):
        # assumes data is a pytorch.tensor
        hostvector = torch.Tensor(self.hostcount)
        hostvector[self.hosts[interface]['id']] = 1.0
        stateprime = torch.cat((data, hostvector), dim=0)
 
        state = self.hosts[interface]['state'] 
        action = self.hosts[interface]['action']
        action = torch.Tensor([action])
        reward = torch.Tensor([reward])
        self.controller.addToMemory(state, action, reward, stateprime)
        if (self.controller.primedToLearn()): 
            self.controller.PerformUpdate(32)
            self.controller.UpdateTargetNetworks()
            self.controller.saveActorCritic()
        self.hosts[interface]['state'] = stateprime

    def predictBandwidthOnHosts(self):
        for host in self.hosts.keys():
            self.predictBandwidthOnHost(host)

    def predictBandwidthOnHost(self, interface):
        state = self.hosts[interface]['state']
        adjustment = self.controller.selectAction(state)[0]
        adjustment = max(min(adjustment, .90), -0.90)
        allocation = self.hosts[interface]['predictedAllocation']
        allocation += adjustment * allocation
        allocation = max(min(allocation, self.defaultmax), 0)
        self.hosts[interface]['predictedAllocation'] = allocation 
        self.hosts[interface]['action'] = adjustment 

    def getHostsBandwidth(self, interface):
        return int(math.ceil(self.hosts[interface]['alloctBandwidth']))

    def getHostsPredictedBandwidth(self, interface):
        return int(math.ceil(self.hosts[interface]['predictedAllocation']))

    def updateHostsBandwidth(self, interface, freebandwidth, loss):
        key = str(interface)
        if(not (key in self.hosts.keys())):
            self.hosts[key] = self.createHost()
            print("New port added: " + key)
        else:
            currhost = self.hosts[key]
            #loss is increasing
            if(loss > 0.0):
                R = -1000
            else:
                R = 1000 
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
            allbandwidths = allbandwidths + " " + str(self.hosts[host]['action'])
        print("Bandwidth Adjustment: " + allbandwidths)