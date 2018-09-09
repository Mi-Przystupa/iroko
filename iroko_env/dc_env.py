
from iroko_monitor import StatsCollector
from iroko_monitor import FlowCollector
from reward_function import RewardFunction

import socket
import signal
import os
import numpy as np
import time

from subprocess import Popen, PIPE
from iroko import MAX_QUEUE

import gym
from gym import error, spaces, utils
from gym.utils import seeding

MAX_CAPACITY = 10e6     # Max bw capacity of link in bytes
MIN_RATE = 6.25e5       # minimal possible bw of an interface in bytes
IS_EXPLOIT = False         # do we want to enable an exploit policy?
ACTIVEAGENT = 'A'       # type of the agent in use
R_FUN = 'std_dev'   # type of the reward function the agent uses
FEATURES = 2            # number of statistics we are using
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


class BandwidthController():
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


class DCEnv(gym.Env):

    def __init__(self, input_dir, output_dir, duration, traffic_file, algorithm, offset, epochs):
        raise NotImplementedError("Method create_filter not implemented!")

    def step(self, action):
        raise NotImplementedError("Method step not implemented!")

    def reset(self):
        raise NotImplementedError("Method reset not implemented!")

    def render(self, mode='human', close=False):
        raise NotImplementedError("Method render not implemented!")


class IrokoEnv(DCEnv):

    def __init__(self, input_dir, output_dir, duration, traffic_file, algorithm,
                 offset, epochs):
        # bench Mark calls
        self.epochs = offset
        self.tf = traffic_file
        os.system('sudo mn -c')
        self.algo = algorithm[0]
        self.conf = algorithm[1]
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.duration = duration

        self.startTraffic(20)
        # Iroko controller calls
        self.ic = BandwidthController("Iroko")

        time.sleep(2)
        self.spawnCollectors()
        self.dopamin = RewardFunction(
            I_H_MAP, self.interfaces, R_FUN, MAX_QUEUE, MAX_CAPACITY)

        self.num_features = FEATURES + len(HOSTS) * 2
        self.num_interfaces = len(self.stats.iface_list)
        self.num_actions = len(I_H_MAP)

        self.action_space = spaces.Box(low=0.0, high=1.0,
                                       shape=(self.num_actions, 1))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.num_interfaces * self.num_features, 1))

    def step(self, action):
        terminal = False
        reward = 0.0
        data = np.zeros((self.num_interfaces, self.num_features))

        if self.p.poll() is not None:
            print('Generator Finished. Simulation over')
            self.KillEnv()
            return data.reshape(self.num_interfaces * self.num_features), True, 0

        # let the agent predict bandwidth based on all previous information
        # perform actions
        pred_bw = {}
        print("Actions:")
        for i, h_iface in enumerate(I_H_MAP):
            pred_bw[h_iface] = int(action[i] * MAX_CAPACITY)
            print("%s: %3f mbit\t" %
                  (h_iface, pred_bw[h_iface] * 10 / MAX_CAPACITY))
            self.ic.send_cntrl_pckt(h_iface, pred_bw[h_iface])
        # observe for WAIT seconds minus time needed for computation
        time.sleep(abs(round(WAIT - (time.time() - self.start_time), 3)))
        self.start_time = time.time()

        try:
            # retrieve the current deltas before updating total values
            delta_vector = self.stats.get_interface_deltas(
                self.bws_rx, self.bws_tx, self.drops, self.overlimits, self.queues)
            # get the absolute values as well as active interface flow
            self.bws_rx, self.bws_tx, self.drops, self.overlimits, self.queues = self.stats.get_interface_stats()
            self.src_flows, self.dst_flows = self.flows.get_interface_flows()
            # Create the data matrix for the agent based on the collected stats
            for i, iface in enumerate(self.interfaces):
                deltas = delta_vector[iface]
                state = [deltas["delta_q_abs"], self.queues[iface]]
                state.extend(self.src_flows[iface])
                state.extend(self.dst_flows[iface])
                print("Current State %s " % iface, state)
                data[i] = np.array(state)
        except Exception as e:
            os.system('sudo chown -R $USER:$USER %s' % self.out_dir)
            # exit gracefully in case of an error
            template = "{0} occurred. Reason:{1!r}. Time to go..."
            message = template.format(type(e).__name__, e.args)

            self.stats.terminate()
            self.flows.terminate()
            print('a wild night eh')
            print message
            return data.reshape(self.num_interfaces * self.num_features), True, 0

        # Compute the reward
        bw_reward, queue_reward = self.dopamin.get_reward(
            self.bws_rx, self.bws_tx, self.queues, pred_bw)
        reward = bw_reward + queue_reward
        print("#######################################")

        return data.reshape(self.num_interfaces * self.num_features), False, reward

    def reset(self):
        print('reseting environment')
        self.KillEnv()
        self.startTraffic()
        self.spawnCollectors()
        return np.zeros(self.num_interfaces * self.num_features)

    def render(self, mode='human', close=False):
        print('nothing to draw at the moment')

    def startTraffic(self, duration=None):
        e = self.epochs
        self.pre_folder = "%s_%d" % (self.conf['pre'], e)
        self.input_file = '%s/%s/%s' % (self.input_dir,
                                        self.conf['tf'], self.tf)
        self.out_dir = '%s/%s/%s' % (self.output_dir, self.pre_folder, self.tf)

        if not duration:
            # is for initialization purposes, no sense running for full time, just long enough to set other parameters
            duration = self.duration
        self.p = Popen('sudo python iroko.py -i %s -d %s -p 0.03 -t %d --%s' %
                       (self.input_file, self.out_dir, duration, self.algo), shell=True, stdout=PIPE)
        self.epochs += 1

        # need to wait until Iroko is started for sure
        time.sleep(5)

    def CheckIfDead(self, tocheck, retry, timeoutlength):
        for i in range(0, retry):
            if tocheck.isAlive():
                tocheck.join(timeoutlength)
            else:
                break

    def KillEnv(self):
        self.stats.terminate()
        self.CheckIfDead(self.stats, 3, 1)

        self.flows.terminate()
        self.CheckIfDead(self.flows, 3, 1)

        print('wait for simulation to end')
        self.p.wait()

    def spawnCollectors(self):
        self.stats = StatsCollector()
        self.stats.daemon = True
        self.stats.start()
        # Launch an asynchronous flow collector
        self.flows = FlowCollector(HOSTS)
        self.flows.daemon = True
        self.flows.start()

        # Let the monitor threads initialize first
        time.sleep(2)
        self.interfaces = self.stats.iface_list
        print ("Running with the following interfaces:")
        print self.interfaces
        self.start_time = time.time()
        # initialize the stats matrix
        self.bws_rx, self.bws_tx, self.drops, self.overlimits, self.queues = self.stats.get_interface_stats()
