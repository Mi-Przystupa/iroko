from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.log import setLogLevel
from mininet.log import info, output, warn, error, debug
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.node import CPULimitedHost
from mininet.util import custom

from time import sleep
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
        self.hostlist = []
        self.switchlist = []

        # Topo initiation
        Topo.__init__(self)

    def create_nodes(self):
        self._create_switches()
        self._create_hosts(self.num_hosts)

    def _create_switches(self, ):
        self.switch_w = self.addSwitch(name="1001", failMode='standalone')
        self.switch_e = self.addSwitch(name="1002", failMode='standalone')
        self.switchlist.append(self.switch_w)
        self.switchlist.append(self.switch_e)

    def _create_hosts(self, num):
        """
            Create hosts.
        """
        for i in range(1, num + 1):
            self.hostlist.append(self.addHost("h" + str(i), cpu=1.0 / num))

    def create_links(self, bw=10, queue=100):
        """
                Add links between switch and hosts.
        """
        for i, host in enumerate(self.hostlist):
            if i < len(self.hostlist) / 2:
                self.addLink(self.switch_w, host, bw=bw,
                             max_queue_size=queue)   # use_htb=False
            else:
                self.addLink(self.switch_e, host, bw=bw,
                             max_queue_size=queue)   # use_htb=False
        self.addLink(self.switch_w, self.switch_e, bw=bw,
                     max_queue_size=queue)   # use_htb=False

    def set_ovs_protocol(self, ovs_v):
        """
            Set the OpenFlow version for switches.
        """
        self._set_ovs_protocol(self.switchlist, ovs_v)

    def _set_ovs_protocol(self, sw_list, ovs_v):
        for sw in sw_list:
            cmd = "sudo ovs-vsctl set bridge %s protocols=OpenFlow%d " % (
                sw, ovs_v)
            os.system(cmd)


class TopoEnv():

    def __init__(self, num_hosts, max_queue):
        self.num_hosts = num_hosts
        self.net, self.topo = self._create_network(
            num_hosts, max_queue=max_queue)
        self._configure_network()

    def _set_host_ip(self, net, topo):
        hostlist = []
        for k in range(len(topo.hostlist)):
            hostlist.append(net.get(topo.hostlist[k]))
        i = 1
        j = 1
        for host in hostlist:
            host.setIP("10.%d.0.%d" % (i, j))
            j += 1
            if j == 3:
                j = 1
                i += 1

    def _connect_controller(self, controller):
        i = 1
        for host in self.topo.hostlist:
            # link.setIP("192.168.0.1")
            host_o = self.net.get(host)
            # Configure host
            self.net.addLink(controller, host)
            host_o.cmd("ifconfig %s-eth1 192.168.10.%d" % (host, i))
            host_o.cmd("route add -net 192.168.5.0/24 dev %s-eth1" % (host))

            # Configure controller
            # intf = controller.intfs[i - 1]
            # intf.rename("c0-%s-eth1" % host)
            controller.cmd("ifconfig c0-eth%s 192.168.5.%d" % (i - 1, i))
            controller.cmd("route add 192.168.10.%d dev c0-eth%s" % (i, i - 1))

            i += 1
            # host.setIP("10.%d.0.%d" % (i, j))

    def _config_topo(self, ovs_v, is_ecmp):
        # Set OVS's protocol as OF13.
        self.topo.set_ovs_protocol(ovs_v)
        # Set hosts IP addresses.
        self._set_host_ip(self.net, self.topo)

    def _configure_network(self):
        ovs_v = 13  # default value
        is_ecmp = True  # default value
        c0 = RemoteController('c0', ip='127.0.0.1', port=6653)
        self.net.addController(c0)

        output('** Waiting for switches to connect to the controller\n')
        sleep(2)
        self._config_topo(ovs_v, is_ecmp)
        self._connect_controller(c0)

    def _create_network(self, num_hosts, cpu=-1, bw=10, max_queue=100):
        setLogLevel('output')
        # Create Topo
        topo = DumbbellTopo(num_hosts)
        topo.create_nodes()
        topo.create_links(bw=bw, queue=max_queue)

        # Start Mininet
        host = custom(CPULimitedHost, cpu=cpu)
        link = custom(TCLink, max_queue=max_queue)
        net = Mininet(topo=topo, host=host, link=link,
                      controller=None, autoSetMacs=True)

        net.start()

        return net, topo

    def get_net(self):
        return self.net

    def get_topo(self):
        return self.topo
