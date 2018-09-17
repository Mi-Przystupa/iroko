from iroko_monitor import StatsCollector
from iroko_monitor import FlowCollector
from reward_function import RewardFunction
import os
from mininet.log import setLogLevel, info, output, warn, error, debug
import socket
import numpy as np
import time
import signal

from topo_dumbbell import TopoConfig
from env_base import BaseEnv
from gym import error, spaces, utils
from gym.utils import seeding

R_FUN = 'std_dev'   # type of the reward function the agent uses
FEATURES = 2            # number of statistics we are using
WAIT = 2                # seconds the agent waits per iteration
###########################################


class BandwidthController():
    def __init__(self, name):
        print("Initializing controller")
        self.name = name
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_cntrl_pckt(self, ip, txrate):
        port = 20130
        pckt = str(txrate) + '\0'
        self.sock.sendto(pckt, (ip, port))


class DCEnv(BaseEnv):

    def __init__(self, offset):
        BaseEnv.__init__(self, offset)
        # Iroko controller calls
        self.ic = BandwidthController("Iroko")
        self.dopamin = RewardFunction(self.topo_conf.I_H_MAP, self.interfaces,
                                      R_FUN, self.topo_conf.MAX_QUEUE,
                                      self.topo_conf.MAX_CAPACITY)

        self.num_features = FEATURES + len(self.topo_conf.HOSTS) * 2
        self.num_actions = len(self.topo_conf.I_H_MAP)

        self.action_space = spaces.Box(low=0.0, high=1.0,
                                       shape=(self.num_actions, 1))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_interfaces * self.num_features, 1))
        self.spawn_collectors()
        self.start_time = time.time()

    def step(self, action):
        terminal = False
        reward = 0.0
        data = np.zeros((self.num_interfaces, self.num_features))

        # let the agent predict bandwidth based on all previous information
        # perform actions
        pred_bw = {}
        print("Actions:")
        for i, h_iface in enumerate(self.topo_conf.I_H_MAP):
            pred_bw[h_iface] = int(action[i] * self.topo_conf.MAX_CAPACITY)
            print("%s: %3f mbit\t" %
                  (h_iface, pred_bw[h_iface] * 10 / self.topo_conf.MAX_CAPACITY))
            self.ic.send_cntrl_pckt(
                self.topo_conf.I_H_MAP[h_iface], pred_bw[h_iface])
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

    def check_if_dead(self, tocheck, retry, timeoutlength):
        for i in range(0, retry):
            if tocheck.isAlive():
                tocheck.join(timeoutlength)
            else:
                break

    def kill_env(self):
        BaseEnv.kill_env(self)
        self.stats.terminate()
        self.check_if_dead(self.stats, 3, 1)

        self.flows.terminate()
        self.check_if_dead(self.flows, 3, 1)

    def create_and_configure_topo(self):
        topo_conf = TopoConfig(4)
        return topo_conf

    def spawn_collectors(self):
        self.stats = StatsCollector()
        self.stats.daemon = True
        self.stats.start()
        # Launch an asynchronous flow collector
        self.flows = FlowCollector(self.topo_conf.HOSTS)
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
