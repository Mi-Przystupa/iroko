
import os
import multiprocessing
from mininet.log import setLogLevel, info, output, warn, error, debug
from time import sleep
from monitor.monitor import monitor_devs_ng
from monitor.monitor import monitor_qlen
import threading

from topo_dumbbell import TopoEnv

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        self._target = target
        self._args = args
        threading.Thread.__init__(self)

    def run(self):
        self._target(*self._args)


class DCEnv(gym.Env):

    def __init__(self, input_dir, output_dir, duration, traffic_file, algorithm, offset, epochs):
        raise NotImplementedError("Method create_filter not implemented!")

    def step(self, action):
        raise NotImplementedError("Method step not implemented!")

    def reset(self):
        raise NotImplementedError("Method reset not implemented!")

    def render(self, mode='human', close=False):
        print('nothing to draw at the moment')

    def get_intf_list(self, net):
        switches = net.switches
        sw_intfs = []
        for switch in switches:
            for intf in switch.intfNames():
                if intf is not 'lo':
                    sw_intfs.append(intf)
        return sw_intfs

    def gen_traffic(self, net, out_dir, input_file, duration):
        if not os.path.exists(out_dir):
            print(out_dir)
            os.makedirs(out_dir)
        ''' Run the traffic generator and monitor all of the interfaces '''
        listen_port = 12345
        sample_period_us = 1000000
        hosts = net.hosts
        traffic_gen = 'cluster_loadgen/loadgen'
        if not os.path.isfile(traffic_gen):
            error(
                'The traffic generator doesn\'t exist. \ncd hedera/cluster_loadgen; make\n')
            return

        output('*** Starting load-generators\n %s\n' % input_file)
        for host in hosts:
            tg_cmd = ('%s -f %s -i %s -l %d -p %d 2&>1 > %s/%s.out &' %
                      (traffic_gen, input_file, host.defaultIntf(), listen_port, sample_period_us, out_dir, host.name))
            host.cmd(tg_cmd)

        sleep(1)

        output('*** Triggering load-generators\n')
        for host in hosts:
            host.cmd('nc -nzv %s %d' % (host.IP(), listen_port))
        ifaces = self.get_intf_list(net)

        monitor1 = multiprocessing.Process(
            target=monitor_devs_ng, args=('%s/rate.txt' % out_dir, 0.01))
        monitor2 = multiprocessing.Process(target=monitor_qlen, args=(
            ifaces, 1, '%s/qlen.txt' % out_dir))

        monitor1.start()
        monitor2.start()
        sleep(duration)
        output('*** Stopping monitor\n')
        monitor1.terminate()
        monitor2.terminate()

        os.system("killall bwm-ng")

        output('*** Stopping load-generators\n')
        for host in hosts:
            host.cmd('killall loadgen')
        net.stop()

    def start_traffic(self):
        self.pre_folder = "%s_%d" % (self.conf['pre'], self.epoch)
        input_file = '%s/%s/%s' % (self.input_dir,
                                   self.conf['tf'], self.tf)
        out_dir = '%s/%s/%s' % (self.output_dir, self.pre_folder, self.tf)
        self.p = FuncThread(self.gen_traffic, self.net,
                            out_dir, input_file, self.duration)
        self.p.start()
        self.epoch += 1
        # need to wait until Iroko is started for sure
        sleep(5)
