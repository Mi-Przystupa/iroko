from tensorforce.environments import Environment
from iroko_controller import  *

import argparser
import os
import sre_yield
import numpy as np
from iroko_plt import IrokoPlotter

from subprocess import Popen, PIPE


class Iroko_Environment(Environment):
    def __init__(self, input_dir, output_dir, duration, traffie_file, algorithm,
            traffic_file, plotter, offset, epochs):

        
        #bench Mark calls
        self.epochs = offset
        self.plotter = plotter
        self.tf = traffic_file
        os.system('sudo mn -c') 
        self.f = open("reward.txt", "a+")

        self.algo = algorithm[0]
        self.conf = algorithm[1]
        self.startIroko()

        #Iroko controller calls
        self.ic = IrokoController("Iroko")
        self.total_reward = TOTAL_ITERS = 0
       
        time.sleep(2)
        self.spawnCollectors()
        self.dopamin = RewardFunction(I_H_MAP,interfaces, R_FUN, MAX_QUEUE, MAX_CAPACITY)

        self.num_features = FEATURES
        self.num_interfaces = len(self.stats.iface_list)
        self.num_actions = len(I_H_MAP)
        # open the reward file
        self.file = open('reward.txt', 'a+')

    def startIroko(self):
        e = self.epochs
        self.pre_folder = "%s_%d" % (self.conf['pre'], e)
        self.input_file = '%s/%s/%s' % (self.input_dir, self.conf['tf'], self.tf)
        self.out_dir = '%s/%s/%s' % (self.output_dir, self.pre_folder, self.tf)
        Popen('sudo python iroko.py -i %s -d %s -p 0.03 -t %d --%s ' %
            (self.input_file, self.out_dir, self.duration, self.algo), shell=True)
        self.epochs += 1

        #Popen('sudo python iroko.py -i %s -d %s -p 0.03 -t %d --%s --agent %s' %
        #    (self.input_file, self.out_dir, self.duration, self.algo, ARGS.agent), shell=True)

    def spawnCollectors(self):
        self.stats = StatsCollector()
        self.stats.daemon=True
        self.stats.start()
         # Launch an asynchronous flow collector
        self.flows = FlowCollector(HOSTS)
        self.flows.daemon = True
        self.flows.start()

        # Let the monitor threads initialize first
        time.sleep(2)
        self.interfaces = self.stats.iface_list
        print ("Running with the following interfaces:")
        print interfaces


    @property
    def states(self):
        return {'type': 'float', 'shape': (self.num_interfaces*self.num_features, )}

    @property
    def actions(self):
        return {'type': 'float', 'num_actions': len(I_H_MAP)}

    def close(self):
        #send kill command
        self.f.close()
        print('closing')

    def reset(self):
        self.startIroko() 
        self.spawnCollectors()
        return np.zeros(18)
    def execute(self, action):
        terminal = False
        reward = 0.0

        data = torch.zeros(self.num_interfaces, self.num_features)

        # let the agent predict bandwidth based on all previous information
        # perform actions
        pred_bw = {}
        for i, h_iface in enumerate(I_H_MAP):
            pred_bw[h_iface] = action[i] 
            ic.send_cntrl_pckt(h_iface, pred_bw[h_iface])
        # observe for WAIT seconds minus time needed for computation
        time.sleep(abs(round(WAIT - (time.time() - start_time), 3)))
        start_time = time.time()

        try:
            # retrieve the current deltas before updating total values
            delta_vector = self.stats.get_interface_deltas(
                bws_rx, bws_tx, drops, overlimits, queues)
            # get the absolute values as well as active interface flow
            bws_rx, bws_tx, drops, overlimits, queues = self.stats.get_interface_stats()
            src_flows, dst_flows = self.flows.get_interface_flows()

            # Create the data matrix for the agent based on the collected stats
            for i, iface in enumerate(interfaces):
                deltas = delta_vector[iface]
                state = [deltas["delta_q_abs"], queues[iface]]
                # print("Current State %s " % iface, state)
                data[i] = torch.Tensor(state)
        except Exception as e:

            os.system('sudo chown -R $USER:$USER %s' % self.out_dir)
            self.plotter.prune_bw(self.out_dir, self.tf, self.conf['sw'])
            # exit gracefully in case of an error
            template = "{0} occurred. Reason:{1!r}. Time to go..."
            message = template.format(type(e).__name__, e.args)
            print message
            data = default
            return data, False, 0  

        # Compute the reward
        print bws_rx
        bw_reward, queue_reward = self.dopamin.get_reward(bws_rx, queues, pred_bw)
        reward = bw_reward + queue_reward
        print("Total Reward: %f BW Reward: %f Queue Reward: %f" %
              (reward, bw_reward, queue_reward))
        print("#######################################")

        # print("Current Reward %d" % reward)
        file.write('%f\n' % (reward))

        total_reward += reward
        total_iters += 1


        return state, terminal, reward

if __name__ == '__main__':
    controller = Iroko_Environment([])
    print(MAX_CAPACITY)
    print('hello')

