from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.link import Link, Intf, TCLink
from mininet.topo import Topo
from mininet.node import OVSKernelSwitch, CPULimitedHost
from mininet.util import custom

from subprocess import Popen, PIPE
from time import sleep, time

import sys
import os

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)


class DumbbellTopo(Topo):
    """
            Class of Dumbbell Topology.
    """

    def __init__(self, hosts):
        self.num_hosts = hosts
        self.switch_w = None
        self.switch_e = None
        self.hostList = []
        self.switchList = []
        self.num_ifaces = 0

        # Topo initiation
        Topo.__init__(self)

    def create_nodes(self):
        self._create_switches()
        self._create_hosts(self.num_hosts)

    def _create_switches(self, ):
        self.switch_w = self.addSwitch(name="1001", failMode='standalone')
        self.switch_e = self.addSwitch(name="1002", failMode='standalone')
        self.switchList.append(self.switch_w)
        self.switchList.append(self.switch_e)

    def _create_hosts(self, NUMBER):
        """
            Create hosts.
        """
        for i in range(1, NUMBER + 1):
            self.hostList.append(self.addHost("h" + str(i), cpu=1.0 / NUMBER))

    def create_links(self, bw=10, queue=100):
        """
                Add links between switch and hosts.
        """
        for i, host in enumerate(self.hostList):
            if i < len(self.hostList) / 2:
                self.addLink(self.switch_w, host, bw=bw,
                             max_queue_size=queue)   # use_htb=False
            else:
                self.addLink(self.switch_e, host, bw=bw,
                             max_queue_size=queue)   # use_htb=False
            self.num_ifaces += 2
        self.addLink(self.switch_w, self.switch_e, bw=bw,
                     max_queue_size=queue)   # use_htb=False
        self.num_ifaces += 2

    def set_ovs_protocol(self, ovs_v):
        """
            Set the OpenFlow version for switches.
        """
        self._set_ovs_protocol(self.switchList, ovs_v)

    def _set_ovs_protocol(self, sw_list, ovs_v):
        for sw in sw_list:
            cmd = "sudo ovs-vsctl set bridge %s protocols=OpenFlow%d " % (
                sw, ovs_v)
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


def connect_controller(net, topo, controller):
    for i, host in enumerate(topo.hostList):
        host_o = net.get(host)
        # Configure host
        net.addLink(controller, host)
        host_o.cmd("ifconfig %s-eth1 192.168.10.%d" % (host, i))
        host_o.cmd("route add -net 192.168.5.0/24 dev %s-eth1" % (host))
        controller.cmd("ifconfig c0-eth%s 192.168.5.%d" % (i - 1, i))
        controller.cmd("route add 192.168.10.%d dev c0-eth%s" % (i, i - 1))


def config_topo(net, topo, ovs_v, is_ecmp):
    # Set OVS's protocol as OF13.
    topo.set_ovs_protocol(ovs_v)
    # Set hosts IP addresses.
    set_host_ip(net, topo)


def create_db_topo(hosts, cpu=-1, bw=10, max_queue=100):
    """
            Firstly, start up Mininet;
            secondly, generate traffics and test the
            performance of the network.
    """
    # Create Topo.
    topo = DumbbellTopo(hosts)
    topo.create_nodes()
    topo.create_links(bw=bw, queue=max_queue)

    # Start Mininet
    host = custom(CPULimitedHost, cpu=cpu)
    link = custom(TCLink, max_queue=max_queue)
    net = Mininet(topo=topo, host=host, link=link,
                  controller=None, autoSetMacs=True)

    return net, topo
