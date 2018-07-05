# Copyright (C) 2016 Huang MaChi at Chongqing University
# of Posts and Telecommunications, China.
# Copyright (C) 2016 Li Cheng at Beijing University of Posts
# and Telecommunications. www.muzixing.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.link import Link, Intf, TCLink
from mininet.topo import Topo
from mininet.node import OVSKernelSwitch, CPULimitedHost
from mininet.util import custom
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)


class NonBlocking(Topo):
    """
            Class of NonBlocking Topology.
    """
    CoreSwitchList = []
    hostList = []

    def __init__(self, k):
        self.pod = k
        self.iCoreLayerSwitch = 1
        self.iHost = k**3 / 4
        # Topo initiation
        Topo.__init__(self)

    def create_nodes(self):
        self.create_core_switch(self.iCoreLayerSwitch)
        self.create_host(self.iHost)

    def _add_switch(self, number, level, switch_list):
        """
                Create switches.
        """
        for i in range(1, number + 1):
            PREFIX = str(level) + "00"
            if i >= 10:
                PREFIX = str(level) + "0"
            switch_list.append(self.addSwitch(PREFIX + str(i)))

    def create_core_switch(self, NUMBER):
        self._add_switch(NUMBER, 1, self.CoreSwitchList)

    def create_host(self, NUMBER):
        """
                Create hosts.
        """
        for i in range(1, NUMBER + 1):
            if i >= 100:
                PREFIX = "h"
            elif i >= 10:
                PREFIX = "h0"
            else:
                PREFIX = "h00"
            self.hostList.append(self.addHost(PREFIX + str(i), cpu=1.0 / float(NUMBER)))

    def create_links(self, bw=10, max_queue=100):
        """
                Add links between switch and hosts.
        """
        for sw in self.CoreSwitchList:
            for host in self.hostList:
                self.addLink(sw, host, bw=bw, max_queue_size=max_queue)   # use_htb=False

    def set_ovs_protocol_13(self):
        """
                Set the OpenFlow version for switches.
        """
        self._set_ovs_protocol_13(self.CoreSwitchList)

    def _set_ovs_protocol_13(self, sw_list):
        for sw in sw_list:
            cmd = "sudo ovs-vsctl set bridge %s protocols=OpenFlow13" % sw
            os.system(cmd)


def set_host_ip(net, topo):
    hostList = []
    for k in range(len(topo.hostList)):
        hostList.append(net.get(topo.hostList[k]))
    i = 1
    j = 1
    # for host in hostList:
    #     host.setIP("10.0.0.%d" % i)
    #     i += 1
    for host in hostList:
        host.setIP("10.%d.0.%d" % (i, j))
        j += 1
        if j == 3:
            j = 1
            i += 1


# def install_proactive(net, topo):
#     """
#             Install proactive flow entries for the switch.
#     """
#     hostList = []
#     for k in range(len(topo.hostList)):
#         hostList.append(net.get(topo.hostList[k]))
#     for sw in topo.CoreSwitchList:
#         i = 1
#         j = 1
#         for k in range(1, topo.iHost + 1):
#             cmd = "ovs-ofctl add-flow %s -O OpenFlow13 \
#                 'table=0,idle_timeout=0,hard_timeout=0,priority=40,arp, \
#                 nw_dst=10.%d.0.%d,actions=output:%d'" % (sw, i, j, k)
#             os.system(cmd)
#             cmd = "ovs-ofctl add-flow %s -O OpenFlow13 \
#                 'table=0,idle_timeout=0,hard_timeout=0,priority=40,ip, \
#                 nw_dst=10.%d.0.%d,actions=output:%d'" % (sw, i, j, k)
#             os.system(cmd)
#             j += 1
#             if j == 3:
#                 j = 1
#                 i += 1


def config_topo(net, topo):
    # Set OVS's protocol as OF13.
    topo.set_ovs_protocol_13()
    # Set hosts IP addresses.
    set_host_ip(net, topo)
    # Install proactive flow entries
    # install_proactive(net, topo)


def create_non_block_topo(pod, cpu=-1, bw=10, max_queue=100):
    """
            Firstly, start up Mininet;
            secondly, generate traffics and test the performance of the network.
    """
    # Create Topo.
    topo = NonBlocking(pod)
    topo.create_nodes()
    topo.create_links(bw=bw, max_queue=max_queue)

    # Start Mininet
    host = custom(CPULimitedHost, cpu=cpu)
    link = custom(TCLink, max_queue=max_queue)
    net = Mininet(topo=topo, host=host, link=link, controller=RemoteController, autoSetMacs=True)
    # net.start()

    return net, topo
