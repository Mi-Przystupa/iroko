
import os
import multiprocessing as mp
from mininet.log import setLogLevel, info, output, warn, error, debug
from time import sleep
from monitor.monitor import monitor_devs_ng
from monitor.monitor import monitor_qlen
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class BaseEnv(gym.Env):

    def __init__(self, offset):
        self.epoch = offset
        self.topo_conf = self.create_and_configure_topo()
        self.net = self.topo_conf.get_net()
        self.topo = self.topo_conf.get_topo()
        self.interfaces = self.topo_conf.get_intf_list()
        self.num_interfaces = len(self.interfaces)

    def step(self, action):
        raise NotImplementedError("Method step not implemented!")

    def reset(self):
        raise NotImplementedError("Method reset not implemented!")

    def render(self, mode='human', close=False):
        print('nothing to draw at the moment')

    def kill_env(self):
        raise NotImplementedError("Method kill_env not implemented!")

    def create_and_configure_topo(self):
        raise NotImplementedError("Method create_net_env not implemented!")

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
                'The traffic generator doesn\'t exist. \ncd cluster_loadgen; make\n')
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
        ifaces = self.topo_conf.get_intf_list()

        monitor1 = mp.Process(
            target=monitor_devs_ng, args=('%s/rate.txt' % out_dir, 0.01))
        monitor2 = mp.Process(target=monitor_qlen, args=(
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

    def start_traffic(self, conf, traffic_file, input_dir, output_dir, epoch, duration):
        self.pre_folder = "%s_%d" % (conf['pre'], epoch)
        input_file = '%s/%s/%s' % (input_dir, conf['tf'], traffic_file)
        out_dir = '%s/%s/%s' % (output_dir, self.pre_folder, traffic_file)
        self.traffic_gen = mp.Process(target=self.gen_traffic,
                                      args=(self.net, out_dir,
                                            input_file, duration))
        self.traffic_gen.start()
        # need to wait until Iroko is started for sure
        sleep(5)
