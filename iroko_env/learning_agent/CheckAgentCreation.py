from __future__ import division

import time
import socket
###########################################
# Stuff for learning
import signal
import torch
from DDPGLearningAgent import GetLearningAgentConfiguration, LearningAgent
i_h_map = {'3001-eth3': "192.168.10.1", '3001-eth4': "192.168.10.2", '3002-eth3': "192.168.10.3", '3002-eth4': "192.168.10.4",
           '3003-eth3': "192.168.10.5", '3003-eth4': "192.168.10.6", '3004-eth3': "192.168.10.7", '3004-eth4': "192.168.10.8",
           '3005-eth3': "192.168.10.9", '3005-eth4': "192.168.10.10", '3006-eth3': "192.168.10.11", '3006-eth4': "192.168.10.12",
           '3007-eth3': "192.168.10.13", '3007-eth4': "192.168.10.14", '3008-eth3': "192.168.10.15", '3008-eth4': "192.168.10.16", }
hosts = ["10.1.0.1", "10.1.0.2", "10.2.0.1", "10.2.0.2", "10.3.0.1", "10.3.0.2", "10.4.0.1", "10.4.0.2",
         "10.5.0.1", "10.5.0.2", "10.6.0.1", "10.6.0.2", "10.7.0.1", "10.7.0.2", "10.8.0.1", "10.8.0.2"]
i_h_map = {'1001-eth1': "192.168.10.1", '1001-eth2': "192.168.10.2",
           '1002-eth1': "192.168.10.3", '1002-eth2': "192.168.10.4"}
hosts = ["10.1.0.1", "10.1.0.2", "10.2.0.1", "10.2.0.2", "10.3.0.1"]


versions = ['C']  # ['v2', 'v3', 'v4', 'v5']
num_stats = 9
num_interfaces = 80
bw_allow = 10e6
ports = i_h_map

for v in versions:
    type = v
    print('Version: {}'.format(v))

    Agent = GetLearningAgentConfiguration(type, ports, num_stats, num_interfaces, bw_allow, 6)
    for i in range(0, 256):
        data = torch.ones(num_interfaces, num_stats)
        isEven = i % 2 == 0

        if (isEven):
            data = data * -1.0

        Agent.predictBandwidthOnHosts()
        actions = []
        for p in ports:
            a = Agent.getHostsPredictedBandwidth(p)
            if (a > .75 and isEven):
                reward = 1.0
            elif (a < .75 and not isEven):
                reward = 1.0
            else:
                reward = -1.0
            Agent.update(data, reward)
            actions.append(a)
        # print(actions)
