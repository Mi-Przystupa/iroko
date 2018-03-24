from __future__ import division
from operator import attrgetter

import time
import threading
import socket
import subprocess
import re

###########################################
# Stuff for learning
import numpy as np
import torch
import random
from LearningAgentv2 import LearningAgentv2
from LearningAgentv3 import LearningAgentv3

MAX_CAPACITY = 10e6   # Max capacity of link
TOSHOW = True
EXPLOIT = False
MAX_QUEUE = 50
###########################################

i_h_map = {'3001-eth3': "192.168.10.1", '3001-eth4': "192.168.10.2", '3002-eth3': "192.168.10.3", '3002-eth4': "192.168.10.4",
           '3003-eth3': "192.168.10.5", '3003-eth4': "192.168.10.6", '3004-eth3': "192.168.10.7", '3004-eth4': "192.168.10.8",
           '3005-eth3': "192.168.10.9", '3005-eth4': "192.168.10.10", '3006-eth3': "192.168.10.11", '3006-eth4': "192.168.10.12",
           '3007-eth3': "192.168.10.13", '3007-eth4': "192.168.10.14", '3008-eth3': "192.168.10.15", '3008-eth4': "192.168.10.16", }


class IrokoController(threading.Thread):
    def __init__(self, name):
        print("Initializing controller")
        threading.Thread.__init__(self)
        self.name = name
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def run(self):
        while True:
            time.sleep(1)
            for iface in i_h_map:
                txrate = random.randint(1310720, 2621440)
                self.send_cntrl_pckt(iface, txrate)

    def send_cntrl_pckt(self, interface, txrate):
        ip = "192.168.10." + i_h_map[interface].split('.')[-1]
        port = 20130
        pckt = str(txrate) + '\0'
        # print("interface: %s, ip: %s, rate: %s") % (interface, ip, txrate)
        self.sock.sendto(pckt, (ip, port))


class StatsCollector():
    """
            NetworkMonitor is a Ryu app for collecting traffic information.
    """

    def __init__(self, *args, **kwargs):
        self.name = 'Collector'
        self.prev_overlimits = 0
        self.prev_loss = 0
        self.prev_util = 0
        self.prev_overlimits_d = 0
        self.prev_loss_d = 0
        self.prev_util_d = 0
        self.iface_list = []
        self.prev_bandwidth = {}

    def _get_deltas(self, curr_loss, curr_overlimits, curr_util):
        # loss_d = max((curr_loss - self.prev_loss) - self.prev_loss_d, 0)

        # losses
        loss_d = curr_loss - self.prev_loss  # loss in epoch

        if loss_d > self.prev_loss_d:
            loss_increase = 1  # loss increasing?
        else:
            loss_increase = 0

        self.prev_loss_d = loss_d  # loss in previous epoch
        self.prev_loss = curr_loss

        # overlimits
        overlimits_d = curr_overlimits - self.prev_overlimits  # overlimints in epoch

        if overlimits_d > self.prev_overlimits_d:
            overlimits_increase = 1
        else:
            overlimits_increase = 0

        self.prev_overlimits_d = overlimits_d
        self.prev_overlimits = curr_overlimits

        # utilization
        util_d = curr_util - self.prev_util

        if util_d > 0:
            util_increase = 1
        else:
            util_increase = 0

        self.prev_util = curr_util

        # print
        # print("Deltas: Loss %d Overlimits %d Utilization: %d " % (loss_d, overlimits_d, util_d))
        # print("Increases: Loss %d Overlimits %d Utilization: %d " % (loss_increase, overlimits_increase, util_increase))
        return loss_d, overlimits_d, util_d, loss_increase, overlimits_increase, util_increase

    def _get_bandwidths(self, iface_list):
        # cmd3 = "ifstat -i %s -q 0.1 1 | awk '{if (NR==3) print $2}'" % (iface)
        bytes_old = {}
        bytes_new = {}
        for iface in iface_list:
            cmd = "awk \"/^ *%s: / \"\' { if ($1 ~ /.*:[0-9][0-9]*/) { sub(/^.*:/, \"\") ; print $1 }\
             else { print $2 } }\' /proc/net/dev" % (iface)
            try:
                output = subprocess.check_output(cmd, shell=True)
            except Exception as e:
                print("Empty Request")
                output = 0
            bytes_old[iface] = float(output) / 1024
        time.sleep(1)
        for iface in iface_list:
            cmd = "awk \"/^ *%s: / \"\' { if ($1 ~ /.*:[0-9][0-9]*/) { sub(/^.*:/, \"\") ; print $1 }\
             else { print $2 } }\' /proc/net/dev" % (iface)
            try:
                output = subprocess.check_output(cmd, shell=True)
            except Exception as e:
                print("Empty Request")
                output = 0
            bytes_new[iface] = float(output) / 1024
        curr_bandwidth = {key: bytes_new[key] - bytes_old.get(key, 0) for key in bytes_new.keys()}

        # Get bandwidth deltas

        bandwidth_d = {}
        if self.prev_bandwidth == {}:
            bandwidth_d = curr_bandwidth  # base case
        else:
            for iface in curr_bandwidth:
                bandwidth_d[iface] = curr_bandwidth[iface] - self.prev_bandwidth[iface]  # calculate delta
        self.prev_bandwidth = curr_bandwidth

        return curr_bandwidth, bandwidth_d

    def _get_free_bw(self, capacity, speed):
        # freebw: Kbit/s
        return max(capacity - speed * 8 / 1000.0, 0)

    def _get_free_bandwidths(self, bandwidths):
        free_bandwidths = {}
        for iface, bandwidth in bandwidths.iteritems():
            free_bandwidths[iface] = self._get_free_bw(MAX_CAPACITY, bandwidth)
        return free_bandwidths

    def _get_bw_deltas(self, bandwidths, bandwidths_old):

        return 0

    def _set_interfaces(self):
        cmd = "sudo ovs-vsctl list-br | xargs -L1 sudo ovs-vsctl list-ports"
        output = subprocess.check_output(cmd, shell=True)
        iface_list_temp = []
        for row in output.split('\n'):
            if row != '':
                iface_list_temp.append(row)
        return_list = [iface for iface in iface_list_temp if iface in i_h_map]  # filter against actual hosts
        self.iface_list = iface_list_temp
        return return_list

    def get_stats_sums(self, bandwidths, free_bandwidths, drops, overlimits, queues):
        bw_sum = sum(bandwidths.itervalues())
        bw_free_sum = sum(free_bandwidths.itervalues())
        loss_sum = sum(drops.itervalues())
        overlimit_sum = sum(overlimits.itervalues())
        queued_sum = sum(queues.itervalues())
        return bw_sum, bw_free_sum, loss_sum, overlimit_sum, queued_sum

    def _get_qdisc_stats(self, iface_list):
        drops = {}
        overlimits = {}
        queues = {}

        re_dropped = re.compile(r'(?<=dropped )[ 0-9]*')
        re_overlimit = re.compile(r'(?<=overlimits )[ 0-9]*')
        re_queued = re.compile(r'backlog\s[^\s]+\s([\d]+)p')
        for iface in iface_list:
            cmd = "tc -s qdisc show dev %s" % (iface)
            # cmd1 = "tc -s qdisc show dev %s | grep -ohP -m1 '(?<=dropped )[ 0-9]*'" % (iface)
            # cmd2 = "tc -s qdisc show dev %s | grep -ohP -m1 '(?<=overlimits )[ 0-9]*'" % (iface)
            # cmd2 = "tc -s qdisc show dev %s | grep -ohP -m1 '(?<=backlog )[ 0-9]*'" % (iface)
            dr = {}
            ov = {}
            qu = {}
            try:
                output = subprocess.check_output(cmd, shell=True)
                dr = re_dropped.findall(output)
                ov = re_overlimit.findall(output)
                qu = re_queued.findall(output)
            except Exception as e:
                print("Empty Request")
                dr[0] = 0
                ov[0] = 0
                qu[0] = 0
            drops[iface] = int(dr[0])
            overlimits[iface] = int(ov[0])
            queues[iface] = int(qu[0])
        return drops, overlimits, queues

    def get_interface_stats(self):
        # iface_list = self._get_interfaces()
        bandwidths, bandwidth_d = self._get_bandwidths(self.iface_list)
        free_bandwidths = self._get_free_bandwidths(bandwidths)
        drops, overlimits, queues = self._get_qdisc_stats(self.iface_list)
        return bandwidths, free_bandwidths, drops, overlimits, queues

    def show_stat(self):
        '''
                Show statistics information according to data type.
                _type: 'port' / 'flow'
        '''
        if TOSHOW is False:
            return

        #     print
        bandwidths, free_bandwidths, drops, overlimits, queues = self.get_interface_stats()
        bw_sum, bw_free_sum, loss_sum, overlimit_sum, queued_sum = self.get_stats_sums(
            bandwidths, free_bandwidths, drops, overlimits, queues)
        loss_d, overlimits_d, util_d, loss_increase, overlimits_increase, util_increase = self._get_deltas(loss_sum, overlimit_sum, bw_sum)
        print("Loss: %d Delta: %d Increase: %d" % (loss_sum, loss_d, loss_increase))
        print("Overlimits: %d Delta: %d Increase: %d" % (overlimit_sum, overlimits_d, overlimits_increase))
        print("Backlog: %d" % queued_sum)

        print("Free BW Sum: %d" % bw_free_sum)
        print("Current Util Sum: %f" % bw_sum)

        max_capacity_sum = MAX_CAPACITY * bandwidths.__len__()
        print("MAX_CAPACITY Sum: %d %d" % (max_capacity_sum, bandwidths.__len__()))
        total_util_ratio = (bw_sum) / max_capacity_sum
        total_util_avg = (bw_sum) / bandwidths.__len__()
        print("Current Average Utilization: %f" % total_util_avg)
        print("Current Ratio of Utilization: %f" % total_util_ratio)

    # sudo ovs-vsctl list-br | xargs -L1 sudo ovs-vsctl list-ports
    # sudo ovs-vsctl list-br | xargs -L1 sudo ovs-ofctl dump-ports -O Openflow13


def HandleDataCollection(self, bodys):
    for dpid in sorted(bodys.keys()):
        # you have device id 300*
        # Also port between 1 - 4 use these
        if(dpid - 3000 >= 0):
            for stat in sorted(bodys[dpid], key=attrgetter('port_no')):
                data = np.array(stat)
                if(stat.port_no < 30):
                    self.Agent.addMemory(data)
                    fb = self.free_bandwidth[dpid][stat.port_no]
                    self.Agent.updateHostsBandwidth(dpid, stat.port_no, fb)
    self.Agent.displayAllHosts()
    self.Agent.displayALLHostsBandwidths()


if __name__ == '__main__':
    # Spawning Iroko controller
    ic = IrokoController("Iroko_Thead")
    # ic.run()

    stats = StatsCollector()
    # Agent = LearningAgent(initMax=MAX_CAPACITY)
    Agent = LearningAgentv2(initMax=(MAX_CAPACITY / 2), memory=1000, cpath='critic', apath='actor', toExploit=EXPLOIT)
    # Agent = LearningAgentv3(initMax=MAX_CAPACITY, memory=1000, cpath='critic', apath='actor', toExploit=EXPLOIT)

    Agent.initializePorts(i_h_map)
    stats._set_interfaces()
    prevdrops = []
    while(1):
        # perform action
        Agent.predictBandwidthOnHosts()
        for interface in i_h_map:
            ic.send_cntrl_pckt(interface, Agent.getHostsPredictedBandwidth(interface))

        # update Agents internal representations
        bandwidths, free_bandwidths, drops, overlimits, queues = stats.get_interface_stats()
        if not prevdrops:
            prevdrops = drops
        for interface in i_h_map:
            data = torch.Tensor([bandwidths[interface], free_bandwidths[interface],
                                 drops[interface], overlimits[interface], queues[interface]])
            # A supposedly more eloquent way of doing it
            reward = MAX_QUEUE - queues[interface] - ((10 * free_bandwidths[interface]) / MAX_CAPACITY)
            Agent.update(interface, data, reward)

        # update the allocated bandwidth
        # wait for update to happen

        # Agent.displayAllHosts()
        # Agent.displayALLHostsBandwidths()
        # Agent.displayALLHostsPredictedBandwidths()
        # Agent.displayAdjustments()

        prevdrops = drops
        # print(stats.get_interface_stats())
