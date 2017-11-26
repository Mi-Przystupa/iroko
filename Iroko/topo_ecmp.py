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

from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info, warn, error, debug
from mininet.link import Link, Intf, TCLink
from mininet.topo import Topo
from mininet.util import dumpNodeConnections

from time import sleep
from mininet.node import OVSKernelSwitch, CPULimitedHost
from mininet.util import custom


import os

MAX_QUEUE = 100
DENSITY = 2


class Fattree(Topo):
    """
        Class of Fattree Topology.
    """
    CoreSwitchList = []
    AggSwitchList = []
    EdgeSwitchList = []
    HostList = []

    def __init__(self, k):
        self.pod = k
        self.density = DENSITY
        self.iCoreLayerSwitch = (k / 2)**2
        self.iAggLayerSwitch = k * k / 2
        self.iEdgeLayerSwitch = k * k / 2
        self.iHost = self.iEdgeLayerSwitch * DENSITY

        # Init Topo
        Topo.__init__(self)

    def createNodes(self):
        self.createCoreLayerSwitch(self.iCoreLayerSwitch)
        self.createAggLayerSwitch(self.iAggLayerSwitch)
        self.createEdgeLayerSwitch(self.iEdgeLayerSwitch)
        self.createHost(self.iHost)

    # Create Switch and Host
    def _addSwitch(self, number, level, switch_list):
        """
            Create switches.
        """
        for i in range(1, number + 1):
            PREFIX = str(level) + "00"
            if i >= 10:
                PREFIX = str(level) + "0"
            switch_list.append(self.addSwitch(PREFIX + str(i)))

    def createCoreLayerSwitch(self, NUMBER):
        self._addSwitch(NUMBER, 1, self.CoreSwitchList)

    def createAggLayerSwitch(self, NUMBER):
        self._addSwitch(NUMBER, 2, self.AggSwitchList)

    def createEdgeLayerSwitch(self, NUMBER):
        self._addSwitch(NUMBER, 3, self.EdgeSwitchList)

    def createHost(self, NUMBER):
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
            self.HostList.append(self.addHost(PREFIX + str(i), cpu=1.0 / NUMBER))

    def createLinks(self, bw_c2a=10, bw_a2e=10, bw_e2h=10, dctcp=False):
        """
            Add network links.
        """
        # Core to Agg
        end = self.pod / 2
        for x in range(0, self.iAggLayerSwitch, end):
            for i in range(0, end):
                for j in range(0, end):
                    self.addLink(
                        self.CoreSwitchList[i * end + j],
                        self.AggSwitchList[x + i],
                        bw=bw_c2a, max_queue_size=MAX_QUEUE, enable_ecn=dctcp)   # use_htb=False

        # Agg to Edge
        for x in range(0, self.iAggLayerSwitch, end):
            for i in range(0, end):
                for j in range(0, end):
                    self.addLink(
                        self.AggSwitchList[x + i], self.EdgeSwitchList[x + j],
                        bw=bw_a2e, max_queue_size=MAX_QUEUE, enable_ecn=dctcp)   # use_htb=False

        # Edge to Host
        for x in range(0, self.iEdgeLayerSwitch):
            for i in range(0, self.density):
                self.addLink(
                    self.EdgeSwitchList[x],
                    self.HostList[self.density * x + i],
                    bw=bw_e2h, max_queue_size=MAX_QUEUE, enable_ecn=dctcp)   # use_htb=False

    def set_ovs_protocol(self, ovs_v):
        """
            Set the OpenFlow version for switches.
        """
        self._set_ovs_protocol(self.CoreSwitchList, ovs_v)
        self._set_ovs_protocol(self.AggSwitchList, ovs_v)
        self._set_ovs_protocol(self.EdgeSwitchList, ovs_v)

    def _set_ovs_protocol(self, sw_list, ovs_v):
        for sw in sw_list:
            cmd = "sudo ovs-vsctl set bridge %s protocols=OpenFlow%d " % (sw, ovs_v)
            os.system(cmd)


def set_host_ip(net, topo):
    hostlist = []
    for k in range(len(topo.HostList)):
        hostlist.append(net.get(topo.HostList[k]))
    i = 1
    j = 1
    for host in hostlist:
        host.setIP("10.%d.0.%d" % (i, j))
        j += 1
        if j == topo.density + 1:
            j = 1
            i += 1


def create_subnetList(topo, num):
    """
        Create the subnet list of the certain Pod.
    """
    subnetList = []
    remainder = num % (topo.pod / 2)
    if topo.pod == 4:
        if remainder == 0:
            subnetList = [num - 1, num]
        elif remainder == 1:
            subnetList = [num, num + 1]
        else:
            pass
    elif topo.pod == 8:
        if remainder == 0:
            subnetList = [num - 3, num - 2, num - 1, num]
        elif remainder == 1:
            subnetList = [num, num + 1, num + 2, num + 3]
        elif remainder == 2:
            subnetList = [num - 1, num, num + 1, num + 2]
        elif remainder == 3:
            subnetList = [num - 2, num - 1, num, num + 1]
        else:
            pass
    else:
        pass
    return subnetList


def install_proactive(net, topo, ovs_v):
    """
        Install proactive flow entries for switches.
    """
    # Edge Switch
    for sw in topo.EdgeSwitchList:
        num = int(sw[-2:])

        # Downstream.
        for i in range(1, topo.density + 1):
            cmd = "ovs-ofctl add-flow %s -O OpenFlow%d \
                'table=0,idle_timeout=0,hard_timeout=0,priority=40,arp, \
                nw_dst=10.%d.0.%d,actions=output:%d'" % (sw, ovs_v, num, i, topo.pod / 2 + i)
            os.system(cmd)
            cmd = "ovs-ofctl add-flow %s -O OpenFlow%d \
                'table=0,idle_timeout=0,hard_timeout=0,priority=40,ip, \
                nw_dst=10.%d.0.%d,actions=output:%d'" % (sw, ovs_v, num, i, topo.pod / 2 + i)
            os.system(cmd)

        # Upstream.
        if topo.pod == 4:
            cmd = "ovs-ofctl add-group %s -O OpenFlow%d \
            'group_id=1,type=select,bucket=output:1,bucket=output:2'" % (sw, ovs_v)
        elif topo.pod == 8:
            cmd = "ovs-ofctl add-group %s -O OpenFlow%d \
            'group_id=1,type=select,bucket=output:1,bucket=output:2,\
            bucket=output:3,bucket=output:4'" % (sw, ovs_v)
        else:
            pass
        os.system(cmd)
        cmd = "ovs-ofctl add-flow %s -O OpenFlow%d \
        'table=0,priority=10,arp,actions=group:1'" % (sw, ovs_v)
        os.system(cmd)
        cmd = "ovs-ofctl add-flow %s -O OpenFlow%d \
        'table=0,priority=10,ip,actions=group:1'" % (sw, ovs_v)
        os.system(cmd)

    # Aggregate Switch
    for sw in topo.AggSwitchList:
        num = int(sw[-2:])
        subnetList = create_subnetList(topo, num)

        # Downstream.
        k = 1
        for i in subnetList:
            cmd = "ovs-ofctl add-flow %s -O OpenFlow%d \
                'table=0,idle_timeout=0,hard_timeout=0,priority=40,arp, \
                nw_dst=10.%d.0.0/16, actions=output:%d'" % (sw, ovs_v, i, topo.pod / 2 + k)
            os.system(cmd)
            cmd = "ovs-ofctl add-flow %s -O OpenFlow%d \
                'table=0,idle_timeout=0,hard_timeout=0,priority=40,ip, \
                nw_dst=10.%d.0.0/16, actions=output:%d'" % (sw, ovs_v, i, topo.pod / 2 + k)
            os.system(cmd)
            k += 1

        # Upstream.
        if topo.pod == 4:
            cmd = "ovs-ofctl add-group %s -O OpenFlow%d \
            'group_id=1,type=select,bucket=output:1,bucket=output:2'" % (sw, ovs_v)
        elif topo.pod == 8:
            cmd = "ovs-ofctl add-group %s -O OpenFlow%d \
            'group_id=1,type=select,bucket=output:1,bucket=output:2,\
            bucket=output:3,bucket=output:4'" % (sw, ovs_v)
        else:
            pass
        os.system(cmd)
        cmd = "ovs-ofctl add-flow %s -O OpenFlow%d \
        'table=0,priority=10,arp,actions=group:1'" % (sw, ovs_v)
        os.system(cmd)
        cmd = "ovs-ofctl add-flow %s -O OpenFlow%d \
        'table=0,priority=10,ip,actions=group:1'" % (sw, ovs_v)
        os.system(cmd)

    # Core Switch
    for sw in topo.CoreSwitchList:
        j = 1
        k = 1
        for i in range(1, len(topo.EdgeSwitchList) + 1):
            cmd = "ovs-ofctl add-flow %s -O OpenFlow%d \
                'table=0,idle_timeout=0,hard_timeout=0,priority=10,arp, \
                nw_dst=10.%d.0.0/16, actions=output:%d'" % (sw, ovs_v, i, j)
            os.system(cmd)
            cmd = "ovs-ofctl add-flow %s -O OpenFlow%d \
                'table=0,idle_timeout=0,hard_timeout=0,priority=10,ip, \
                nw_dst=10.%d.0.0/16, actions=output:%d'" % (sw, ovs_v, i, j)
            os.system(cmd)
            k += 1
            if k == topo.pod / 2 + 1:
                j += 1
                k = 1


def configureTopo(net, topo, ovs_v, is_ecmp):
    # Set OVS's protocol as OF13.
    topo.set_ovs_protocol(ovs_v)
    # Set hosts IP addresses.
    set_host_ip(net, topo)
    # Install proactive flow entries
    if is_ecmp:
        install_proactive(net, topo, ovs_v)


def connect_controller(net, topo, controller):
    # for sw in topo.EdgeSwitchList:
    #     net.addLink(sw, controller)
    # for sw in topo.AggSwitchList:
    #     net.addLink(sw, controller)
    # for sw in topo.CoreSwitchList:
    #     net.addLink(sw, controller)
    i = 1
    for host in topo.HostList:
        # link.setIP("192.168.0.1")
        host_o = net.get(host)
        # Configure host
        net.addLink(controller, host)
        host_o.cmdPrint("ifconfig %s-eth1 192.168.10.%d" % (host, i))
        host_o.cmdPrint("route add -net 192.168.5.0/24 dev %s-eth1" % (host))
        
        # Configure controller
        # intf = controller.intfs[i - 1]
        # intf.rename("c0-%s-eth1" % host)
        controller.cmdPrint("ifconfig c0-eth%s 192.168.5.%d" % (i-1, i))
        controller.cmdPrint("route add 192.168.10.%d dev c0-eth%s" % (i, i-1))

        i += 1
        # host.setIP("10.%d.0.%d" % (i, j))


def createECMPTopo(pod, density, ip="127.0.0.1", port=6653, cpu=-1, bw_c2a=10, bw_a2e=10, bw_e2h=10, dctcp=False):
    """
        Create network topology and run the Mininet.
    """
    DENSITY = density
    # Create Topo.
    topo = Fattree(pod)
    topo.createNodes()
    topo.createLinks(bw_c2a=bw_c2a, bw_a2e=bw_a2e, bw_e2h=bw_e2h, dctcp=dctcp)
    # Start Mininet
    # CONTROLLER_IP = ip
    #CONTROLLER_PORT = port
    link = custom(TCLink, max_queue_size=MAX_QUEUE, enable_ecn=dctcp)
    host = custom(CPULimitedHost, cpu=cpu)
    net = Mininet(topo=topo, host=host, link=link, controller=None, autoSetMacs=True)
    #net.addController('controller', controller=RemoteController, ip=CONTROLLER_IP, port=CONTROLLER_PORT)

    # net.start()
    return (net, topo)
