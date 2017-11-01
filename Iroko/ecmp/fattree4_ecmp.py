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
from mininet.examples.consoles import ConsoleApp

from time import sleep
import multiprocessing
from mininet.node import OVSKernelSwitch, CPULimitedHost
from mininet.util import custom


from subprocess import Popen, PIPE
from argparse import ArgumentParser
from monitor.monitor import monitor_devs_ng
from monitor.monitor import monitor_qlen

import os
import logging

parser = ArgumentParser(description="Iroko Parser")

parser.add_argument('-d', '--dir', dest='output_dir', default='log',
                    help='Output directory')

parser.add_argument('-i', '--input', dest='input_file',
                    default='../inputs/all_to_all_data',
                    help='Traffic generator input file')

parser.add_argument('-t', '--time', dest='time', type=int, default=60,
                    help='Duration (sec) to run the experiment')
args = parser.parse_args()

MAX_QUEUE = 1


class Fattree(Topo):
    """
        Class of Fattree Topology.
    """
    CoreSwitchList = []
    AggSwitchList = []
    EdgeSwitchList = []
    HostList = []

    def __init__(self, k, density):
        self.pod = k
        self.density = density
        self.iCoreLayerSwitch = (k / 2)**2
        self.iAggLayerSwitch = k * k / 2
        self.iEdgeLayerSwitch = k * k / 2
        self.iHost = self.iEdgeLayerSwitch * density

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

    def createLinks(self, bw_c2a=10, bw_a2e=10, bw_e2h=10):
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
                        bw=bw_c2a, max_queue_size=MAX_QUEUE)   # use_htb=False

        # Agg to Edge
        for x in range(0, self.iAggLayerSwitch, end):
            for i in range(0, end):
                for j in range(0, end):
                    self.addLink(
                        self.AggSwitchList[x + i], self.EdgeSwitchList[x + j],
                        bw=bw_a2e, max_queue_size=MAX_QUEUE)   # use_htb=False

        # Edge to Host
        for x in range(0, self.iEdgeLayerSwitch):
            for i in range(0, self.density):
                self.addLink(
                    self.EdgeSwitchList[x],
                    self.HostList[self.density * x + i],
                    bw=bw_e2h, max_queue_size=MAX_QUEUE)   # use_htb=False

    def set_ovs_protocol_13(self,):
        """
            Set the OpenFlow version for switches.
        """
        self._set_ovs_protocol_13(self.CoreSwitchList)
        self._set_ovs_protocol_13(self.AggSwitchList)
        self._set_ovs_protocol_13(self.EdgeSwitchList)

    def _set_ovs_protocol_13(self, sw_list):
        for sw in sw_list:
            cmd = "sudo ovs-vsctl set bridge %s protocols=OpenFlow13" % sw
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


def install_proactive(net, topo):
    """
        Install proactive flow entries for switches.
    """
    # Edge Switch
    for sw in topo.EdgeSwitchList:
        num = int(sw[-2:])

        # Downstream.
        for i in range(1, topo.density + 1):
            cmd = "ovs-ofctl add-flow %s -O OpenFlow13 \
                'table=0,idle_timeout=0,hard_timeout=0,priority=40,arp, \
                nw_dst=10.%d.0.%d,actions=output:%d'" % (sw, num, i, topo.pod / 2 + i)
            os.system(cmd)
            cmd = "ovs-ofctl add-flow %s -O OpenFlow13 \
                'table=0,idle_timeout=0,hard_timeout=0,priority=40,ip, \
                nw_dst=10.%d.0.%d,actions=output:%d'" % (sw, num, i, topo.pod / 2 + i)
            os.system(cmd)

        # Upstream.
        if topo.pod == 4:
            cmd = "ovs-ofctl add-group %s -O OpenFlow13 \
            'group_id=1,type=select,bucket=output:1,bucket=output:2'" % sw
        elif topo.pod == 8:
            cmd = "ovs-ofctl add-group %s -O OpenFlow13 \
            'group_id=1,type=select,bucket=output:1,bucket=output:2,\
            bucket=output:3,bucket=output:4'" % sw
        else:
            pass
        os.system(cmd)
        cmd = "ovs-ofctl add-flow %s -O OpenFlow13 \
        'table=0,priority=10,arp,actions=group:1'" % sw
        os.system(cmd)
        cmd = "ovs-ofctl add-flow %s -O OpenFlow13 \
        'table=0,priority=10,ip,actions=group:1'" % sw
        os.system(cmd)

    # Aggregate Switch
    for sw in topo.AggSwitchList:
        num = int(sw[-2:])
        subnetList = create_subnetList(topo, num)

        # Downstream.
        k = 1
        for i in subnetList:
            cmd = "ovs-ofctl add-flow %s -O OpenFlow13 \
                'table=0,idle_timeout=0,hard_timeout=0,priority=40,arp, \
                nw_dst=10.%d.0.0/16, actions=output:%d'" % (sw, i, topo.pod / 2 + k)
            os.system(cmd)
            cmd = "ovs-ofctl add-flow %s -O OpenFlow13 \
                'table=0,idle_timeout=0,hard_timeout=0,priority=40,ip, \
                nw_dst=10.%d.0.0/16, actions=output:%d'" % (sw, i, topo.pod / 2 + k)
            os.system(cmd)
            k += 1

        # Upstream.
        if topo.pod == 4:
            cmd = "ovs-ofctl add-group %s -O OpenFlow13 \
            'group_id=1,type=select,bucket=output:1,bucket=output:2'" % sw
        elif topo.pod == 8:
            cmd = "ovs-ofctl add-group %s -O OpenFlow13 \
            'group_id=1,type=select,bucket=output:1,bucket=output:2,\
            bucket=output:3,bucket=output:4'" % sw
        else:
            pass
        os.system(cmd)
        cmd = "ovs-ofctl add-flow %s -O OpenFlow13 \
        'table=0,priority=10,arp,actions=group:1'" % sw
        os.system(cmd)
        cmd = "ovs-ofctl add-flow %s -O OpenFlow13 \
        'table=0,priority=10,ip,actions=group:1'" % sw
        os.system(cmd)

    # Core Switch
    for sw in topo.CoreSwitchList:
        j = 1
        k = 1
        for i in range(1, len(topo.EdgeSwitchList) + 1):
            cmd = "ovs-ofctl add-flow %s -O OpenFlow13 \
                'table=0,idle_timeout=0,hard_timeout=0,priority=10,arp, \
                nw_dst=10.%d.0.0/16, actions=output:%d'" % (sw, i, j)
            os.system(cmd)
            cmd = "ovs-ofctl add-flow %s -O OpenFlow13 \
                'table=0,idle_timeout=0,hard_timeout=0,priority=10,ip, \
                nw_dst=10.%d.0.0/16, actions=output:%d'" % (sw, i, j)
            os.system(cmd)
            k += 1
            if k == topo.pod / 2 + 1:
                j += 1
                k = 1


def start_tcpprobe():
    os.system("killall -9 cat")
    ''' Install tcp_probe module and dump to file '''
    os.system("rmmod tcp_probe; modprobe tcp_probe full=1;")
    Popen("cat /proc/net/tcpprobe > ./tcp.txt", shell=True)


def stop_tcpprobe():
    os.system("killall -9 cat")


def iperfTrafficGen(args, hosts, net):
    ''' Generate traffic pattern using iperf and monitor all of thr interfaces

    input format:
    src_ip dst_ip dst_port type seed start_time stop_time flow_size r/e
    repetitions time_between_flows r/e (rpc_delay r/e)

    '''
    host_list = {}
    for h in hosts:
        print(h.IP())
        host_list[h.IP()] = h

    port = 5001

    data = open(args.input_file)

    start_tcpprobe()
    info('*** Starting iperf ...\n')
    for line in data:
        flow = line.split(' ')
        src_ip = flow[0]
        dst_ip = flow[1]
        if src_ip not in host_list:
            continue
        sleep(0.2)
        server = host_list[dst_ip]
        server.popen('iperf -s -p %s > ./server.txt' % port, shell=True)

        client = host_list[src_ip]
        client.popen('iperf -c %s -p %s -t %d > ./client.txt'
                     % (server.IP(), port, args.time), shell=True)

    monitor = multiprocessing.Process(target=monitor_devs_ng, args=('%s/rate.txt' % args.output_dir, 0.01))

    monitor.start()

    sleep(args.time)

    monitor.terminate()

    info('*** stoping iperf ...\n')
    stop_tcpprobe()

    Popen("killall iperf", shell=True).wait()


def trafficGen(args, hosts, net):
    ''' Run the traffic generator and monitor all of the interfaces '''
    listen_port = 12345
    sample_period_us = 1000000

    traffic_gen = '../cluster_loadgen/loadgen'
    if not os.path.isfile(traffic_gen):
        error('The traffic generator doesn\'t exist. \ncd hedera/cluster_loadgen; make\n')
        return

    info('*** Starting load-generators\n %s\n' % args.input_file)
    for h in hosts:
        tg_cmd = '%s -f %s -i %s -l %d -p %d 2&>1 > %s/%s.out &' % (traffic_gen,
                                                                    args.input_file, h.defaultIntf(), listen_port, sample_period_us,
                                                                    args.output_dir, h.name)
        h.cmd(tg_cmd)

    sleep(1)

    info('*** Triggering load-generators\n')
    for h in hosts:
        h.cmd('nc -nzv %s %d' % (h.IP(), listen_port))

    monitor = multiprocessing.Process(target=monitor_devs_ng, args=('%s/rate.txt' % args.output_dir, 0.01))

    monitor.start()

    sleep(args.time)

    monitor.terminate()

    info('*** Stopping load-generators\n')
    for h in hosts:
        h.cmd('killall loadgen')


def iperfTest(net, topo):
    """
        Start iperf test.
    """
    h001, h015, h016 = net.get(
        topo.HostList[0], topo.HostList[14], topo.HostList[15])
    bw = 10**6
    for edgeSwitchName in topo.EdgeSwitchList:
        switch = net.get(edgeSwitchName)
        print(switch)
        for intf in switch.intfList():
            if str(intf) != 'lo':
                switch.cmdPrint('dstat --net --time -N ' + str(intf) + '> ./tmp/dstat-' +
                                str(edgeSwitchName) + '-' + str(intf) + '.txt &')
                # os.system('ovs-vsctl -- set Port ' + str(intf) + ' qos=@newqos -- --id=@newqos create QoS type=linux-htb other-config:max-rate=' +
                #           str(bw) + ' queues=0=@q0 -- --id=@q0   create   Queue   other-config:min-rate=' + str(bw) + ' other-config:max-rate=' + str(bw))
                # os.system('ovs-vsctl set Interface ' + str(intf) + ' ingress_policing_rate=' + str(bw))

    serverPort = 5000
    clientPort = serverPort
    # for i in range(len(topo.HostList) - 1):
    #     # h001.cmdPrint('xterm  -T \"server' + str(serverPort) +
    #     #              '\" -e \"iperf3 -s -i 1 -p ' + str(serverPort) + '; bash\" &')
    #     h001.cmdPrint('iperf3 -s -D -i 1 -p ' + str(serverPort) + ' --json')  # ' > ./tmp/server_' + str(serverPort) + '')
    #     serverPort += 1

    h001.cmdPrint('xterm  -T \"server' + str(serverPort) + '\" -e \"iperf3 -s -i 1 -p 5001; bash\" &')
    h001.cmdPrint('xterm  -T \"server' + str(serverPort) + '\" -e \"iperf3 -s -i 1 -p 5002; bash\" &')
    sleep(5)
    h015.cmdPrint('iperf3 -c ' + h001.IP() + ' -u -t 40 -i 1 -b 10m -p 5001 & ')
    h016.cmdPrint('iperf3 -c ' + h001.IP() + ' -u -t 40 -i 1 -b 10m -p 5002 & ')
    # Input VPN-IDS

    monitor = multiprocessing.Process(target=monitor_qlen, args=('3001-eth2', 0.01, './log/qlen.txt'))

    monitor.start()

    sleep(50)

    monitor.terminate()
    # for hostname in topo.HostList:
    #     host = net.get(hostname)
    #     if not(host == h001):
    #         sleep(1)
    #         host.cmdPrint('iperf3 -c ' + h001.IP() + ' -t 20 -i 1 -b 10m -p ' +
    #                       str(clientPort) + ' & ')
    #         clientPort += 1
#    sleep(60)
    # iperf Server
    # h001.popen('iperf -s -u -i 1 > iperf_server_differentPod_result', shell=True)
    # iperf Server
    # h015.popen('iperf -s -u -i 1 > iperf_server_samePod_result', shell=True)
    # iperf Client
    # h016.cmdPrint('iperf -c ' + h001.IP() + ' -u -t 10 -i 1 -b 10m')
    # h016.cmdPrint('iperf -c ' + h015.IP() + ' -u -t 10 -i 1 -b 10m')


def pingTest(net):
    """
        Start ping test.
    """
    net.pingAll()


def createTopo(pod, density, ip="127.0.0.1", port=6653, bw_c2a=10, bw_a2e=10, bw_e2h=10):
    """
        Create network topology and run the Mininet.
    """
    # Create Topo.
    topo = Fattree(pod, density)
    topo.createNodes()
    topo.createLinks(bw_c2a=bw_c2a, bw_a2e=bw_a2e, bw_e2h=bw_e2h)

    link = custom(TCLink, max_queue_size=MAX_QUEUE)
    net = Mininet(topo=topo, link=link, controller=RemoteController, autoSetMacs=True)
    net.start()

    # Set OVS's protocol as OF13.
    topo.set_ovs_protocol_13()
    # Set hosts IP addresses.
    set_host_ip(net, topo)
    # Install proactive flow entries
    install_proactive(net, topo)
    # dumpNodeConnections(net.hosts)
    # pingTest(net)

    hosts = []
    for hostname in topo.HostList:
        hosts.append(net.get(hostname))
    # trafficGen(args, hosts, net)
    # app = ConsoleApp(net, width=4)
    iperfTest(net, topo)
    # app.mainloop()
    CLI(net)
    net.stop()
    clean()


def clean():
    ''' Clean any the running instances of POX '''

    p = Popen("ps aux | grep 'pox' | awk '{print $2}'",
              stdout=PIPE, shell=True)
    p.wait()
    procs = (p.communicate()[0]).split('\n')
    for pid in procs:
        try:
            pid = int(pid)
            Popen('kill %d' % pid, shell=True).wait()
        except:
            pass
    Popen('killall iperf3', shell=True).wait()
    Popen('killall xterm', shell=True).wait()


if __name__ == '__main__':
    clean()
    setLogLevel('info')
    if not os.path.exists(args.output_dir):
        print(args.output_dir)
        os.makedirs(args.output_dir)

    if os.getuid() != 0:
        logging.debug("You are NOT root")
    elif os.getuid() == 0:
        # createTopo(2, 1)
        createTopo(4, 2)
        # createTopo(8, 4)


# @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
# def port_stats_reply_handler(self, ev):
#     ports = []
#     for stat in ev.msg.body:
#         ports.append('port_no=%d '
#                      'rx_packets=%d tx_packets=%d '
#                      'rx_bytes=%d tx_bytes=%d '
#                      'rx_dropped=%d tx_dropped=%d '
#                      'rx_errors=%d tx_errors=%d '
#                      'rx_frame_err=%d rx_over_err=%d rx_crc_err=%d '
#                      'collisions=%d duration_sec=%d duration_nsec=%d' %
#                      (stat.port_no,
#                       stat.rx_packets, stat.tx_packets,
#                       stat.rx_bytes, stat.tx_bytes,
#                       stat.rx_dropped, stat.tx_dropped,
#                       stat.rx_errors, stat.tx_errors,
#                       stat.rx_frame_err, stat.rx_over_err,
#                       stat.rx_crc_err, stat.collisions,
#                       stat.duration_sec, stat.duration_nsec))
#     self.logger.debug('PortStats: %s', ports)
