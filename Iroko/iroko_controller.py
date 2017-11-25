from __future__ import division
import copy
from operator import attrgetter

from ryu import cfg
from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
import sys
import time
import subprocess
sys.path.append('./')

###########################################
# Stuff for learning
import numpy as np
import torch
from LearningAgent import LearningAgent

MAX_CAPACITY = 10000   # Max capacity of link
TOSHOW = True

i_h_map = {'3001-eth3': "192.168.10.1", '3001-eth4': "192.168.10.2", '3002-eth3': "192.168.10.3", '3002-eth4': "192.168.10.4",
           '3003-eth3': "192.168.10.5", '3003-eth4': "192.168.10.6", '3004-eth3': "192.168.10.7", '3004-eth4': "192.168.10.8",
           '3005-eth3': "192.168.10.9", '3005-eth4': "192.168.10.10", '3006-eth3': "192.168.10.11", '3006-eth4': "192.168.10.12",
           '3007-eth3': "192.168.10.13", '3007-eth4': "192.168.10.14", '3008-eth3': "192.168.10.15", '3008-eth4': "192.168.10.16", }

class StatsCollector():
    """
            NetworkMonitor is a Ryu app for collecting traffic information.
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        self.name = 'monitor'
        self.datapaths = {}
        self.port_speed = {}
        self.flow_stats = {}
        self.flow_speed = {}
        self.stats = {}
        self.port_features = {}
        self.free_bandwidth = {}   # self.free_bandwidth = {dpid:{port_no:free_bw,},} unit:Kbit/s
        self.graph = None
        self.capabilities = None
        self.best_paths = None
        self.prev_overlimits = 0
        self.prev_loss = 0
        self.prev_util = 0
        self.prev_overlimits_d = 0
        self.prev_loss_d = 0
        self.prev_util_d = 0

    def _get_deltas(self, curr_loss, curr_overlimits, curr_util):
        loss_d = max((curr_loss - self.prev_loss) - self.prev_loss_d, 0)
        self.prev_loss_d = loss_d
        self.prev_loss = curr_loss
        overlimits_d = max((curr_overlimits - self.prev_overlimits) - self.prev_overlimits_d, 0)
        self.prev_overlimits_d = overlimits_d
        self.prev_overlimits = curr_overlimits
        util_d = curr_util - self.prev_util
        self.prev_util = curr_util
        print("Deltas: Loss %d Overlimits %d Utilization: %d " % (loss_d, overlimits_d, util_d))
        return loss_d, overlimits_d, util_d

    def _get_bandwidths(self, iface_list):
        #cmd3 = "ifstat -i %s -q 0.1 1 | awk '{if (NR==3) print $2}'" % (iface)
        bytes_old = {}
        bytes_new = {}
        for iface in iface_list:
            cmd = "awk \"/^ *%s: / \"\' { if ($1 ~ /.*:[0-9][0-9]*/) { sub(/^.*:/, \"\") ; print $1 } else { print $2 } }\' /proc/net/dev" % (
                iface)
            try:
                output = subprocess.check_output(cmd, shell=True)
            except:
                print("Empty Request")
                output = 0
            bytes_old[iface] = float(output) / 1024
        time.sleep(1)
        for iface in iface_list:
            cmd = "awk \"/^ *%s: / \"\' { if ($1 ~ /.*:[0-9][0-9]*/) { sub(/^.*:/, \"\") ; print $1 } else { print $2 } }\' /proc/net/dev" % (
                iface)
            try:
                output = subprocess.check_output(cmd, shell=True)
            except:
                print("Empty Request")
                output = 0
            bytes_new[iface] = float(output) / 1024
        curr_bandwidth = {key: bytes_new[key] - bytes_old.get(key, 0) for key in bytes_new.keys()}
        return curr_bandwidth

    def _get_active_ports(self):
        sw = self.port_features
        iface_list = []
        for dpid in sorted(sw.keys()):
            for port in sorted(sw[dpid]):
                if port != ofproto_v1_3.OFPP_LOCAL:
                    iface = str(dpid) + "-eth" + str(port)
                    iface_list.append(iface)
        return iface_list

    def _get_free_bw(self, capacity, speed):
        # freebw: Kbit/s
        return max(capacity - speed * 8 / 1000.0, 0)

    def show_stat(self):
        '''
                Show statistics information according to data type.
                _type: 'port' / 'flow'
        '''
        if TOSHOW is False:
            return

        #     print
        loss_sum_old = 0
        loss_sum_new = 0
        overlimits = 0
        free_bw = 0
        avg_util_new = 0

        i = 0
        cmd = "sudo ovs-vsctl list-br | xargs -L1 sudo ovs-vsctl list-ports"
        output = subprocess.check_output(cmd, shell=True)
        iface_list = []
        for row in output.split('\n'):
            if row != '':
                iface_list.append(row)

        print(iface_list)

        for iface in iface_list:
            cmd1 = "tc -s qdisc show dev %s | grep -ohP -m1 '(?<=dropped )[ 0-9]*'" % (iface)
            cmd2 = "tc -s qdisc show dev %s | grep -ohP -m1 '(?<=overlimits )[ 0-9]*'" % (iface)
            try:
                output1 = subprocess.check_output(cmd1, shell=True)
                output2 = subprocess.check_output(cmd2, shell=True)
            except:
                print("Empty Request")
                output1 = 0
                output2 = 0
            loss_sum_new += int(output1)
            overlimits += int(output2)
            i += 1
        bandwidths = self._get_bandwidths(iface_list)
        avg_util = sum(bandwidths.itervalues())
        loss_d, overlimits_d, util_d = self._get_deltas(loss_sum_new, overlimits, free_bw)
        print("Current Old Loss: %d" % loss_sum_old)
        print("Current New Loss: %d" % loss_sum_new)
        print("Overlimits: %d" % overlimits)

        print("Free BW Sum: %d" % free_bw)
        print("Current Util Sum: %f" % avg_util)

        max_capacity_sum = MAX_CAPACITY * i
        print("MAX_CAPACITY Sum: %d %d" % (max_capacity_sum, i))
        curr_avg_util = (max_capacity_sum - free_bw) / max_capacity_sum
        curr_avg_util_alt = (avg_util_new / max_capacity_sum)
        print("Current Average Utilization: %f" % curr_avg_util)
        print("Current Average Utilization ALT: %f" % curr_avg_util_alt)

    # sudo ovs-vsctl list-br | xargs -L1 sudo ovs-vsctl list-ports
    # sudo ovs-vsctl list-br | xargs -L1 sudo ovs-ofctl dump-ports -O Openflow13


if __name__ == '__main__':
    while(1):
        stats = StatsCollector()
        stats.show_stat()
