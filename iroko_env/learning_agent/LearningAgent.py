import torch
import torch.nn
from torch.autograd import Variable
import random
import numpy as np
import math
from DDPG import DDPG
from DDPGConv import DDPGConv


class LearningAgent:
    def __init__(self, maximumBandwidth=10e6, controller, toExploit=False):
        self.hosts = {}
        self.hostcount = 0
        self.maximumBandwidth = maximumBandwidth
        self.controller = controller
        self.min_val = 0.5
        self.max_val = 1.0
        if(toExploit):
            self.controller.exploit()
        else:
            self.controller.explore()

    def factory(type, states, actions, maxbandwidth, cpath, apath, exploit, mem=1e3, dims=None, frames=3):
        if type == 'v2':
            controller = DDPG(states, 1, memory=mem, criticpath='critic', actorpath='actor')
            return LearningAgentv2(controller toExploit=exploit)
            # LearningAgentv2(states, gamma=.99, memory=1000, maxbandwidth=10e6, \
            # cpath=None, apath=None, toExploit=False)

        if type == 'v3':
            controller = DDPG(states, actions, memory=mem, criticpath='critic', actorpath='actor', dims=dims)
            return LearningAgentv3(maxbandwidth=maxbandwidth, toExploit=exploit, frames=frames)
            # LearningAgentv3(gamma=.99, s=2560, actions=16, memory=1000,
            # maxbandwidth=10e6, cpath=None, apath=None,
            # toExploit=False, frames=3, dims=None)

        if type == 'v4':
            controller = DDPG(states, actions, criticpath='critic', actorpath='actor')
            return LearningAgentv4(controller, maxbandwidth=maxbanwidth, toExploit=toExploit)
            # LearningAgentv4(gamma=.99,  s=2560, actions=16, memory=1000,
            # maxbandwidth=10e6, cpath=None, apath=None,
            # toExploit=False)

    factory = staticmethod(factory)

    def initializePorts(self, ports):
        # cleans out old ports and creates maps for new ones
        self.hosts = {}
        for key in ports.keys():
            self.hosts[key] = self._createHost()
            self.hostcount += 1
        self.min_val = 1.0 / float(self.hostcount)

    def initializeTrafficMatrix(self, interfaceCount=16, features=5, frames=3):
        raise NotImplementedError('intializeTrafficMatrix Must be implemented')

    def _createHost(self):
        self.hostcount += 1
        host = dict(alloctBandwidth=self.initmax, e=0, predictedAllocation=self.initmax)
        # should just allocate full bandwidth at start
        host['action'] = 0.5
        # state depends on each agents representation
        host['state'] = None
        # host['modifier'] = 0
        host['id'] = self.hostcount - 1
        return host

    def _handleData(self, data, reward):
        # must be implemented in subclasses
        # subclasses make assumptions on shape of state which should be done in here
        raise NotImplementedError('_handleError must be implemented in base class')

    def update(self, interfaces, data, reward):
        # assumes data is a pytorch.tensor
        # memories should be a list of dictionaries with keys s,a,r, sp
        # newstates is dictionary corresponding to new states hosts are in
        # we always need to know condition of the states, but the memories
        # a controller learns on can vary
        memories, newstates = self._handleData(data, interfaces, reward)

        for m in memories:
            self.controller.addToMemory(m['s'], m['a'], m['r'], m['sp'])
        self._updateController()
        self._updateHostsState(interfaces, stateprimes)

    def _updateHostsState(self, interfaces, newstates):
        for interface in interfaces:
            self.hosts[interface]['state'] = newstates[interface]

    def _updateController(self):
        if (self.controller.primedToLearn()):
            self.controller.PerformUpdate(64)
            self.controller.UpdateTargetNetworks()
            self.controller.saveActorCritic()

    def predictBandwidthOnHosts(self):
        # actions should be a dictionary of specified actions
        actions = self._getActions()
        for host in self.hosts.keys():
            a = np.clip(actions[host], self.min_val, self.max_val)
            self.hosts[interface]['predictedAllocation'] = self.defaultmax * a
            self.hosts[interface]['action'] = a

    def _getActions(self):
        raise NotImplementedError('_getActions must be implemented')

    def getHostsPredictedBandwidth(self, interface):
        return int(math.ceil(self.hosts[interface]['predictedAllocation']))

    def displayAllHosts(self):
        allnames = ""
        for host in self.hosts.keys():
            allnames = allnames + " " + str(host)
        print(allnames)

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


class LearningAgentv2(LearningAgent):
    def __init__(self, controller, maxbandwidth, toExploit=False)

        super(LearningController, self).__init__(maximumBandwidth=maxbandwidth,
                                                 controller=controller, toExploit=toExploit)

    def _handleData(self, interfaces, data, reward):
        # must be implemented in subclasses
        # subclasses make assumptions on shape of state which should be done in here
        # v2 has a single action and predicts actions 1 at a time
        memories = []
        newstates = {}
        data = data.view(-1)
        for interface in interfaces:
            # assumes data is a pytorch.tensor
            hostvector = torch.zeros(self.hostcount)
            hostvector[self.hosts[interface]['id']] = 1.0
            stateprime = torch.cat((data, hostvector), dim=0)
            state = self.hosts[interface]['state']
            if (not state):
                state = torch.zeros(self.controller.state + len(hostvector))
            action = self.hosts[interface]['action']
            action = torch.Tensor([action])
            reward = torch.Tensor([reward])
            memories.append({'s': state, 'a': action, 'r': reward, 'sp': stateprime})
            newstates[interface] = stateprime
        return memories, newstates

    def initializeTrafficMatrix(self, interfaceCount=16, features=5, frames=3):
        i = 0
        # do nothing
        return

    def _getActions(self, interfaces):
        actions = {}
        for interface in interfaces:
            state = self.hosts[interface]['state']
            a = self.controller.selectAction(state.unsqueeze(0))
            actions[interface] = action.squeeze()[0]
        return actions


class LearningAgentv3(LearningAgent):
    def __init__(self, controller, maxbandwidth, toExploit=False, frames=3):

        self.state = []
        self.prevState = []
        self.actionVector = []
        self.actionVector = torch.zeros(actions)
        self.frames = frames

        super(LearningAgent, self).__init__(controller,
                                            maximumBandwidth=maxbandwidth, toExploit=toExploit)

    def initializeTrafficMatrix(self, interfaceCount=16, features=5, frames=3):
        self.state = torch.zeros(frames, interfaceCount, features)
        self.prevState = torch.zeros(frames, interfaceCount, features)
        self.frames = frames

    def _handleData(self, interfaces, data):
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
        memories = [{'s': self.prevState, 'a': self.actionVector, 'r': reward,
                     'sp': self.state}]
        newstates = []
        return memories, newstates

    def _getActions(self, interfaces):
        state = self.state
        a = self.controller.selectAction(state.unsqueeze(0))
        a = a.squeeze()
        actions = {}
        for i in interfaces:
            actions[i] = a[self.hosts[interface]['id']]
        return actions


class LearningAgentv4(LearningAgent):
    def __init__(self, controller, maxbandwidth=10e6, toExploit=False):

        self.state = []
        self.prevState = []
        self.actionVector = []
        self.actionVector = torch.zeros(actions)

        super(LearningAgent, self).__init__(controller,
                                            maximumBandwidth=maxbandwidth, toExploit=toExploit)

    def initializeTrafficMatrix(self, interfaceCount=16, features=5, frames=3):
        self.state = torch.zeros(interfaceCount * features)
        self.prevState = torch.zeros(interfaceCount * features)

    def _handleData(self, interfaces, data):
        self.prevState = self.state
        data = data.view(-1)
        self.state = data
        # set action
        for i in interface:
            self.actionVector[self.hosts[i]['id']] = self.hosts[i]['action']
        memories = [{'s': self.prevState, 'a': self.actionVector, 'r': reward,
                     'sp': self.state}]
        newstates = []
        return memories, newstates

    def _getActions(self, interfaces):
        state = self.state
        a = self.controller.selectAction(state.unsqueeze(0))
        a = a.squeeze()
        actions = {}
        for i in interfaces:
            actions[i] = a[self.hosts[interface]['id']]
        return actions
