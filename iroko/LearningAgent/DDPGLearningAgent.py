import torch
import torch.nn
from torch.autograd import Variable
import random
import numpy as np
import math
from DDPG import DDPG


def GetLearningAgentConfiguration(type, ports, num_stats, num_interfaces, bw_allow, frames):
    if type == 'v2':
        return LearningAgent(ports, num_stats, num_interfaces, bw_allow,
                             one_hot=True, use_conv=None, min_alloc=.1, max_alloc=1.0, name='v2')
    if type == 'v3':
        use_conv = {'c': frames, 'h': num_interfaces, 'w': num_stats, 'num_feature_maps': 32}
        return LearningAgent(ports, num_stats, num_interfaces, bw_allow,
                             one_hot=False, use_conv=use_conv, min_alloc=.1, max_alloc=1.0, name='v3')
    if type == 'v4':
        return LearningAgent(ports, num_stats, num_interfaces, bw_allow,
                             one_hot=False, use_conv=None, min_alloc=.1, max_alloc=1.0, name='v4')
    if type == 'v5':
        return LearningAgent(ports, num_stats, num_interfaces, bw_allow,
                             one_hot=False, use_conv=None, min_alloc=.1, max_alloc=1.0,
                             online=True, name='v5')


class LearningAgent:
    def __init__(self, ports, num_stats, num_interfaces, bw_allow,
                 one_hot, use_conv, min_alloc=.5, max_alloc=1.0, online=False, name='agent'):
        # inputs:
        # ports we want to predict on
        # allo
        assert online != one_hot or (not one_hot and not online), 'either online or one_hot not both'
        self.hosts = {}
        self.hostcount = 0
        self.min_alloc = min_alloc
        self.max_alloc = max_alloc
        self.full_bw = bw_allow
        self.onehot = one_hot
        self.use_conv = use_conv
        self.online = online

        self.num_stats = num_stats
        self.num_interfaces = num_interfaces
        self.state = []
        self.prevState = []
        self.frames = 0
        self.name = name

	self.display_action = 0
        # initialize ports last
        self.initializePorts(ports)
        # initialize DDPG with default configs, can call explicitly to alter self.controller = {}
        self.initializeController()

        # initialize internal representation for when not using one_hot
        self._initializeTrafficMatrix()
        self.actionVector = torch.zeros(self.hostcount)

    def initializePorts(self, ports):
        # cleans out old ports and creates maps for new ones
        self.hosts = {}
        for key in ports.keys():
            self.hosts[key] = self._createHost()
            self.hostcount += 1

    def _initializeTrafficMatrix(self):
        if(self.use_conv):
            self.state = torch.zeros(self.use_conv['c'], self.num_interfaces, self.num_stats)
            self.prevState = torch.zeros(self.use_conv['c'], self.num_interfaces, self.num_stats)
            self.frames = self.use_conv['c']
        elif(not self.onehot):
            self.state = torch.zeros(self.num_interfaces * self.num_stats)
            self.prevState = torch.zeros(self.num_interfaces * self.num_stats)
        # if neither of these, do not set

    def _calculateFullyConnectedSize(self):
        h = self.use_conv['h']
        w = self.use_conv['w']
        # this depends on how many convolutions we are using...magic numbering this for now
        num_conv = 3
        for _ in range(0, num_conv):
            h = np.floor((h - 1.0) / 2.0 + 1)
            w = np.floor((w - 1.0) / 1.0 + 1)
        return h * w * self.use_conv['num_feature_maps']

    def initializeController(self, tau=.001, gamma=.99, memory=1e3, learningRate=1e-3,
                             criticpath=None, actorpath=None, h1=400, h2=300, dims=None):
        if (self.use_conv):
            mod = self._calculateFullyConnectedSize()
        else:
            mod = self.num_interfaces * self.num_stats

        s = mod
        print(s)
        s = int(s)

        if (self.onehot):
            a = 1
            s = s + self.hostcount
        else:
            a = self.hostcount

        if dims:
            self.use_conv = dims

        self.controller = DDPG(s, a, dims=self.use_conv, criticpath='critic' + self.name, actorpath='actor' + self.name)

    def _createHost(self):
        host = dict(predictedAllocation=self.full_bw)
        # should just allocate full bandwidth at start
        host['action'] = 1.0
        # state depends on each agents representation
        if (self.use_conv):
            host['state'] = self.state
        else:
            host['state'] = None
        # host['modifier'] = 0
        host['id'] = self.hostcount - 1
        return host

    def _dataOneHot(self, data, reward):
        memories = []
        newstates = {}
        data = data.view(-1)
        for interface in self.hosts.keys():
            # assumes data is a pytorch.tensor
            hostvector = torch.zeros(self.hostcount)
            hostvector[self.hosts[interface]['id']] = 1.0
            sp = torch.cat((data, hostvector), dim=0)
            s = self.hosts[interface]['state']
            if (s is None):
                s = torch.zeros(self.num_interfaces * self.num_stats)
                s = torch.cat((s, hostvector))
            a = self.hosts[interface]['action']
            a = torch.Tensor([a])
            r = torch.Tensor([reward])
            memories.append({'s': s, 'a': a, 'r': r, 'sp': sp})
            newstates[interface] = sp
        return memories, newstates

    def exploit(self):
        self.controller.exploit()

    def explore(self):
        self.controller.explore()

    def _dataConv(self, data, reward):
        self.prevState = self.state
        # push back previous frames
        if self.frames > 1:
            # might have this backwards...
            # as long as it's working like a queue it should be fine...
            # that implicity velocity stuff is kinda hand wavy anyways.
            for f in range(0, self.frames - 1):
                self.state[f] = self.state[f + 1]

        self.state[self.frames - 1] = data
        # set action
        for i in self.hosts.keys():
            self.actionVector[self.hosts[i]['id']] = self.hosts[i]['action']
        memories = [{'s': self.prevState, 'a': self.actionVector, 'r': reward,
                     'sp': self.state}]
        # technically even if it's empty should be fine
        newstates = {k: self.state for k in self.hosts.keys()}
        return memories, newstates

    def _dataFlat(self, data, reward):
        self.prevState = self.state
        data = data.view(-1)
        self.state = data
        # set action
        for i in self.hosts.keys():
            self.actionVector[self.hosts[i]['id']] = self.hosts[i]['action']
        memories = [{'s': self.prevState, 'a': self.actionVector, 'r': reward,
                     'sp': self.state}]
        newstates = {k: self.state for k in self.hosts.keys()}

        return memories, newstates

    def _handleData(self, data, reward):
        if self.use_conv:
            return self._dataConv(data, reward)
        elif self.onehot:
            return self._dataOneHot(data, reward)
        else:
            return self._dataFlat(data, reward)

    def update(self, data, reward):
        # assumes data is a pytorch.tensor
        # memories should be a list of dictionaries with keys s,a,r, sp
        # newstates is dictionary corresponding to new states hosts are in
        # we always need to know condition of the states, but the memories
        # a controller learns on can vary
        memories, newstates = self._handleData(data, reward)

        for m in memories:
            self.controller.addToMemory(m['s'], m['a'], m['r'], m['sp'])
        if not self.use_conv:
            data = data.view(-1)
        self._updateController(data, reward)
        self._updateHostsState(newstates)

    def _updateHostsState(self, newstates):
        for interface in self.hosts.keys():
            self.hosts[interface]['state'] = newstates[interface]

    def _updateController(self, data, reward):
        if self.online:
            # use sample to learng right away
            self.controller.UpdateOnline(data.unsqueeze(0), reward)
            self.controller.UpdateTargetNetworks()
            self.controller.saveActorCritic()
        elif (self.controller.primedToLearn()):
            # use replay buffer to do update
            self.controller.PerformUpdate(64)
            self.controller.UpdateTargetNetworks()
            self.controller.saveActorCritic()

    def predictBandwidthOnHosts(self):
        # actions should be a dictionary of specified actions
        actions = self._getActions()
        if self.display_action % 10 == 0:
            print('curr action: {}'.format(actions))
            self.display_action = 0
        for host in self.hosts.keys():
            a = np.clip(actions[host], self.min_alloc, self.max_alloc)
            self.hosts[host]['predictedAllocation'] = self.full_bw * a
            self.hosts[host]['action'] = a

    def _getOneHotActions(self):
        actions = {}
        for interface in self.hosts.keys():
            # assumes data is a pytorch.tensor
            hostvector = torch.zeros(self.hostcount)
            hostvector[self.hosts[interface]['id']] = 1.0
            state = self.hosts[interface]['state']
            if state is None:
                state = torch.zeros(self.num_interfaces * self.num_stats)
                state = torch.cat((state, hostvector))
            a = self.controller.selectAction(state.unsqueeze(0))
            actions[interface] = a.squeeze()[0]

        return actions

    def _getFullActions(self):
        state = self.state
        a = self.controller.selectAction(state.unsqueeze(0))
        a = a.squeeze()
        actions = {}
        for interface in self.hosts.keys():
            actions[interface] = a[self.hosts[interface]['id']]
        return actions

    def _getActions(self):

        if(self.onehot):
            return self._getOneHotActions()
        else:
            return self._getFullActions()

    def getHostsPredictedBandwidth(self, interface):
        return int(self.hosts[interface]['predictedAllocation'])

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
