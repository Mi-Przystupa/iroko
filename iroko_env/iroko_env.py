from tensorforce.environments import Environment
from iroko_controller import  *

class Iroko_Environment(Environment):
    def __init__(self, args):
        self.args = args
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


    def startSimulation(self):
        print('do something')


    @property
    def states(self):
        return {'type': 'float', 'shape': (self.num_interfaces*self.num_features, )}

    @property
    def actions(self):
        return {'type': 'float', 'num_actions': len(I_H_MAP)}

    def close(self):
        #send kill command
        print('closing')

    def reset(self):
        self.startSimulation() 
        return []

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
            # exit gracefully in case of an error
            template = "{0} occurred. Reason:{1!r}. Time to go..."
            message = template.format(type(e).__name__, e.args)
            print message
            data = default
            return [], False, 0  

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

