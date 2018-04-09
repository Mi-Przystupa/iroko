import torch
import numpy



class RewardFunction:
    def __init__(self, hosts, interfaces, function_name, max_queue):
        self.interfaces = interfaces
        self.num_interfaces = len(interfaces)
        self.hosts = hosts
        self.func_name = function_name
        self.max_queue = max_queue
        self.has_congestion = set() 
    def GetReward(self, bw, queues):
        if self.func_name == 'QueueBandwidthBalance':
            return self._QueueBandwidthBalance(bw, queues)
        elif (self.func_name == 'QueuePrecision'):
            return self._QueuePrecision(queues)
    def _QueueBandwidthBalance(self,bws_rx, queues):
        bw_reward = 0.0
        queue_reward = 0.0
        for i, iface in enumerate(self.interfaces):
            bw_reward += float(bws_rx[iface]) / float(self.max_queue)
            print('{} bw reward so far: {}'.format(i,bw_reward))
            queue_reward -= self.num_interfaces * (float(queues[iface]) / float(self.max_queue))**2

        return bw_reward, queue_reward

    def _QueuePrecision(self, queues):
        bw_reward = 0.0
        queue_reward = 0.0
        for i, iface in enumerate(self.interfaces):
            if not (iface in self.hosts):
                q = float(queues[iface])
                if ( q > 0.0 and iface not in self.has_congestion):
                    self.has_congestion.add(iface)
                    queue_reward = 0.0 #don't worry about it first time around
                elif (iface in self.has_congestion):
                    if self.max_queue / 5. < q and  q <= self.max_queue / 2.:
                        queue_reward += 0.0
                    elif q <= self.max_queue/ 5.:
                        queue_reward += 0.5
                    else:
                        queue_reward -= 1.0 
        return bw_reward, queue_reward


