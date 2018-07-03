from __future__ import division

import time
import socket
import math
###########################################
# Stuff for learning
import subprocess
import signal
import torch
from argparse import ArgumentParser
from learning_agent import DDPGLearningAgent

from iroko_monitor import StatsCollector
from iroko_monitor import FlowCollector
from reward_function import RewardFunction

MAX_CAPACITY = 10e6     # Max bw capacity of link in bytes
MIN_RATE = 6.25e5       # minimal possible bw of an interface in bytes
IS_EXPLOIT = False         # do we want to enable an exploit policy?
ACTIVEAGENT = 'A'       # type of the agent in use
R_FUN = 'std_dev'   # type of the reward function the agent uses
FEATURES = 5            # number of statistics we are using
MAX_QUEUE = 5000         # depth of the switch queues
WAIT = 2                # seconds the agent waits per iteration
FRAMES = 3              # number of previous matrices to use

###########################################

I_H_MAP = {'3001-eth3': "192.168.10.1", '3001-eth4': "192.168.10.2", '3002-eth3': "192.168.10.3", '3002-eth4': "192.168.10.4",
           '3003-eth3': "192.168.10.5", '3003-eth4': "192.168.10.6", '3004-eth3': "192.168.10.7", '3004-eth4': "192.168.10.8",
           '3005-eth3': "192.168.10.9", '3005-eth4': "192.168.10.10", '3006-eth3': "192.168.10.11", '3006-eth4': "192.168.10.12",
           '3007-eth3': "192.168.10.13", '3007-eth4': "192.168.10.14", '3008-eth3': "192.168.10.15", '3008-eth4': "192.168.10.16", }
HOSTS = ["10.1.0.1", "10.1.0.2", "10.2.0.1", "10.2.0.2", "10.3.0.1", "10.3.0.2", "10.4.0.1", "10.4.0.2",
         "10.5.0.1", "10.5.0.2", "10.6.0.1", "10.6.0.2", "10.7.0.1", "10.7.0.2", "10.8.0.1", "10.8.0.2"]
I_H_MAP = {'1001-eth1': "192.168.10.1", '1001-eth2': "192.168.10.2",
           '1002-eth1': "192.168.10.3", '1002-eth2': "192.168.10.4"}
HOSTS = ["10.1.0.1", "10.1.0.2", "10.2.0.1", "10.2.0.2"]


PARSER = ArgumentParser()
PARSER.add_argument('--agent', dest='version',
                    default=ACTIVEAGENT, help='options are A, B, C, D')
PARSER.add_argument('--exploit', '-e', dest='exploit', default=IS_EXPLOIT,
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


def init_agent(version, is_exploit, interfaces, num_features):
    size = len(interfaces)
    Agent = DDPGLearningAgent.GetLearningAgentConfiguration(
        version, I_H_MAP, num_features, size,
        bw_allow=MAX_CAPACITY, frames=FRAMES)
    if is_exploit:
        Agent.exploit()
    else:
        Agent.explore()
    return Agent


if __name__ == '__main__':
    # set any configuration things
    # just incase
    ic = IrokoController("Iroko")
    ARGS.version = ARGS.version.lower()
    total_reward = total_iters = 0
    saver = GracefulSave()

    # open the reward file
    file = open('reward.txt', 'a+')

    # Launch an asynchronous stats collector
    stats = StatsCollector()
    stats.daemon = True
    stats.start()

    # Launch an asynchronous flow collector
    flows = FlowCollector(HOSTS)
    flows.daemon = True
    flows.start()

    # Let the monitor threads initialize first
    time.sleep(2)
    interfaces = stats.iface_list
    print ("Running with the following interfaces:")
    print interfaces

    # count the number of interfaces after we have explored the environment
    num_interfaces = len(interfaces)

    # initialize the stats matrix
    bws_rx, bws_tx, drops, overlimits, queues = stats.get_interface_stats()

    # initialize the reward function
    dopamin = RewardFunction(
        I_H_MAP, interfaces, R_FUN, MAX_QUEUE, MAX_CAPACITY)

    # kind of a wild card, num_features depends on the input we have
    num_features = FEATURES  # + len(HOSTS) * 2 + num_delta

    # initialize the Agent
    Agent = init_agent(ARGS.version, IS_EXPLOIT, interfaces, num_features)

    # start timer
    start_time = time.time()
    while 1:
        reward = 0.0
        data = torch.zeros(num_interfaces, num_features)

        # let the agent predict bandwidth based on all previous information
        Agent.predictBandwidthOnHosts()
        # perform actions
        pred_bw = {}
        for h_iface in I_H_MAP:
            pred_bw[h_iface] = Agent.getHostsPredictedBandwidth(h_iface)
            ic.send_cntrl_pckt(h_iface, pred_bw[h_iface])
        # observe for WAIT seconds minus time needed for computation
        time.sleep(abs(round(WAIT - (time.time() - start_time), 3)))
        start_time = time.time()

        try:
            # retrieve the current deltas before updating total values
            delta_vector = stats.get_interface_deltas(
                bws_rx, bws_tx, drops, overlimits, queues)
            # get the absolute values as well as active interface flow
            bws_rx, bws_tx, drops, overlimits, queues = stats.get_interface_stats()
            src_flows, dst_flows = flows.get_interface_flows()

            # Create the data matrix for the agent based on the collected stats
            for i, iface in enumerate(interfaces):
                deltas = delta_vector[iface]
                state = [bws_rx[iface], bws_tx[iface], deltas["delta_q_abs"],
                         deltas["delta_tx_abs"], deltas["delta_rx_abs"]]
                # print("Current State %s " % iface, state)
                data[i] = torch.Tensor(state)
        except Exception as e:
            # exit gracefully in case of an error
            template = "{0} occurred. Reason:{1!r}. Time to go..."
            message = template.format(type(e).__name__, e.args)
            print message
            break

        # Compute the reward
        print bws_rx
        bw_reward, queue_reward = dopamin.get_reward(bws_rx, queues, pred_bw)
        reward = bw_reward + queue_reward
        print("Total Reward: %f BW Reward: %f Queue Reward: %f" %
              (reward, bw_reward, queue_reward))
        print("#######################################")

        # update Agents internal representations
        # majority of computation happens here
        Agent.update(data, reward)

        # write out the reward in this iteration
        # print("Current Reward %d" % reward)
        file.write('%f\n' % (reward))

        total_reward += reward
        total_iters += 1

        # catch a SIGTERM/SIGKILL and terminate gracefully
        if saver.kill_now:
            break

    file.close()
