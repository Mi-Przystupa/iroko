from __future__ import division

import time
import socket
###########################################
# Stuff for learning
import subprocess
import signal
import torch
from argparse import ArgumentParser
from LearningAgentv2 import LearningAgentv2
from LearningAgentv3 import LearningAgentv3
from LearningAgentv4 import LearningAgentv4
from iroko_monitor import StatsCollector
from iroko_monitor import FlowCollector


MAX_CAPACITY = 10e6   # Max capacity of link
MIN_RATE = 6.25e5
EXPLOIT = False
ACTIVEAGENT = 'v2'
FEATURES = 1  # number of statistics we are using
MAX_QUEUE = 50

###########################################

I_H_MAP = {'3001-eth3': "192.168.10.1", '3001-eth4': "192.168.10.2", '3002-eth3': "192.168.10.3", '3002-eth4': "192.168.10.4",
           '3003-eth3': "192.168.10.5", '3003-eth4': "192.168.10.6", '3004-eth3': "192.168.10.7", '3004-eth4': "192.168.10.8",
           '3005-eth3': "192.168.10.9", '3005-eth4': "192.168.10.10", '3006-eth3': "192.168.10.11", '3006-eth4': "192.168.10.12",
           '3007-eth3': "192.168.10.13", '3007-eth4': "192.168.10.14", '3008-eth3': "192.168.10.15", '3008-eth4': "192.168.10.16", }
HOSTS = ["10.1.0.1", "10.1.0.2", "10.2.0.1", "10.2.0.2", "10.3.0.1", "10.3.0.2", "10.4.0.1", "10.4.0.2",
         "10.5.0.1", "10.5.0.2", "10.6.0.1", "10.6.0.2", "10.7.0.1", "10.7.0.2", "10.8.0.1", "10.8.0.2"]
I_H_MAP = {'1001-eth1': "192.168.10.1", '1001-eth2': "192.168.10.2"}
HOSTS = ["10.1.0.1", "10.1.0.2", "10.2.0.1", "10.2.0.2", "10.3.0.1"]


PARSER = ArgumentParser()
PARSER.add_argument('--agent', dest='version', default=ACTIVEAGENT, help='options are v0, v2,v3, v4')
PARSER.add_argument('--exploit', '-e', dest='exploit', default=EXPLOIT,
                    type=bool, help='flag to use explore or expoit environment')

ARGS = PARSER.parse_args()


class IrokoController():
    def __init__(self, name):
        print("Initializing controller")
        self.name = name
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_cntrl_pckt(self, interface, txrate):
        ip = "192.168.10." + I_H_MAP[interface].split('.')[-1]
        port = 20130
        pckt = str(txrate) + '\0'
        # print("interface: %s, ip: %s, rate: %s") % (interface, ip, txrate)
        self.sock.sendto(pckt, (ip, port))


class GracefulSave:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit)
        signal.signal(signal.SIGTERM, self.exit)

    def exit(self, signum, frame):
        print("Time to die...")
        self.kill_now = True


def init_agent(version, exploit, interfaces, features):
    FEATURE_MAPS = 32  # this is internal to v3 convolution filters...probably should be defined in the model
    FRAMES = 1  # number of previous matrices to use
    size = len(interfaces)
    if version == 'v2' or version == 'v0':
        Agent = LearningAgentv2(initMax=MAX_CAPACITY, memory=1000, s=size * features +
                                len(I_H_MAP), cpath='critic', apath='actor', toExploit=exploit)
    elif version == 'v3':
        Agent = LearningAgentv3(initMax=MAX_CAPACITY, memory=1000, actions=len(I_H_MAP),
                                s=(FEATURE_MAPS * size * features) / 8, cpath='critic',
                                apath='actor', toExploit=exploit, frames=FRAMES, w=features)
        Agent.initializeTrafficMatrix(len(interfaces), features=features, frames=FRAMES)
    elif version == 'v4':
        Agent = LearningAgentv4(initMax=MAX_CAPACITY, memory=1000, actions=len(I_H_MAP),
                                s=size * features, cpath='critic', apath='actor', toExploit=exploit)
        Agent.initializeTrafficMatrix(size, features=features, frames=FRAMES)
    else:
        # you had 3 options of 2 characters length and still messed up.
        # Be humbled, take a deep breath and center yourself
        raise ValueError('Invalid agent, options are v2,v3,v4')
    Agent.initializePorts(I_H_MAP)
    return Agent


if __name__ == '__main__':
    # set any configuration things
    # just incase
    ic = IrokoController("Iroko")
    ARGS.version = ARGS.version.lower()
    saver = GracefulSave()
    # Launch an asynchronous stats collector
    stats = StatsCollector()
    stats.set_interfaces()
    stats.daemon = True
    stats.start()
    interfaces = stats.iface_list
    flows = FlowCollector(HOSTS)
    flows.set_interfaces()
    flows.daemon = True
    flows.start()
    # let the monitor initialize first
    time.sleep(3)
    num_interfaces = len(interfaces)
    total_reward = 0
    total_iters = 0
    f = open('reward.txt', 'a+')
    features = FEATURES + len(HOSTS) * 2
    bws_rx = {}
    bws_tx = {}
    drops = {}
    overlimits = {}
    queues = {}
    delta_vector = stats.init_deltas()

    # initialize the Agent
    Agent = init_agent(ARGS.version, EXPLOIT, interfaces, features)

    while(1):
        # perform action
        Agent.predictBandwidthOnHosts()
        for h_iface in I_H_MAP:
            ic.send_cntrl_pckt(h_iface, Agent.getHostsPredictedBandwidth(h_iface))
        # update Agents internal representations
        time.sleep(3)
        if bws_rx:
            delta_vector = stats.get_interface_deltas(bws_rx, bws_tx, drops, overlimits, queues)
        bws_rx, bws_tx, drops, overlimits, queues = stats.get_interface_stats()
        src_flows, dst_flows = flows.get_interface_flows()
        data = torch.zeros(num_interfaces, features)
        reward = 0.0
        bw_reward = 0.0
        try:
            for i, iface in enumerate(interfaces):
                # if iface == "1001-eth3":
                #     print("iface: %s rx: %f tx: %f drops: %d over %d queues %d" %
                #           (iface, bws_rx[iface], bws_tx[iface], drops[iface], overlimits[iface], queues[iface]))
                #     print(delta_vector[iface])
                # state = [bws_rx[iface], bws_tx[iface], queues[iface]] + src_flows[iface] + dst_flows[iface]
                state = [queues[iface]] + src_flows[iface] + dst_flows[iface]
                data[i] = torch.Tensor(state)
                # if queues[iface] == 0:
                #    reward += MAX_QUEUE / 100
                #    bw_reward += (MAX_QUEUE / 1000) * float(bandwidths[iface]) / float(MAX_CAPACITY)
                # else:
                # if delta_vector[iface]["delta_q"] == 1:
                bw_reward += float(bws_rx[iface]) / float(MAX_CAPACITY)
                reward -= num_interfaces * (float(queues[iface]) / float(MAX_QUEUE))
        except Exception as e:
            print("Time to go: %s" % e)
            break
        reward += bw_reward
        print("Total Reward %f BW Reward %f " % (reward, bw_reward))

        # print("Current Reward %d" % reward)
        f.write('%f\n' % (reward))
        # if ACTIVEAGENT == 'v0':
        #     # the historic version
        #     for interface in I_H_MAP:
        #         data = torch.Tensor([bandwidths[interface], free_bandwidths[interface],
        #                              drops[interface], overlimits[interface], queues[interface]])
        #         # A supposedly more eloquent way of doing it
        #         # A supposedly more eloquent way of doing it
        #         reward = 0
        #         # if(drops[interface]  > 0.0):
        #         if(queues[interface]):
        #             reward = -1.0
        #         else:
        #             reward = 1.0
        #         Agent.update(interface, data, reward)
        if ARGS.version == 'v2':
            # fully connected agent that  uses full matrix for each action but uses current host as input
            data = data.view(-1)
            for interface in I_H_MAP:
                Agent.update(interface, data, reward)
        elif ARGS.version == 'v3':
            # just dump in traffic matrix and let a rip
            Agent.update(I_H_MAP, data, reward)
        elif ARGS.version == 'v4':
            # flatten the matrix & feed it in
            # v4 the fully connected input of v2 mixed with the single output of v3
            data = data.view(-1)
            Agent.update(I_H_MAP, data, reward)
        total_reward += reward
        total_iters += 1
        if saver.kill_now:
            break
        # update the allocated bandwidth
        # wait for update to happen

        # Agent.displayAllHosts()
        # Agent.displayALLHostsBandwidths()
        # Agent.displayALLHostsPredictedBandwidths()
        # Agent.displayAdjustments()

        # print(stats.get_interface_stats())
    f.close()
