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
from multiprocessing import Process
from mininet.term import makeTerm

import topo_ecmp
import topo_non_block
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


def traffic_generation(net, topo, flows_peers, duration):
    """
            Generate traffics and test the performance of the network.
    """
    print("Running iperf test")
    # 1. Start iperf. (Elephant flows)
    # Start the servers.
    serversList = set([peer[1] for peer in flows_peers])
    for server in serversList:
        filename = server[1:]
        server = net.get(server)
        server.cmd("iperf -s > %s/%s &" % (args.output_dir, 'server' + filename + '.txt'))
        # server.cmd("iperf -s > /dev/null &")   # Its statistics is useless, just throw away.
    monitor = multiprocessing.Process(target=monitor_devs_ng, args=('%s/rate.txt' % args.output_dir, 0.01))
    sleep(3)

    # Start the clients.
    for src, dest in flows_peers:
        server = net.get(dest)
        client = net.get(src)
        filename = src[1:]
        client.cmd("iperf -c %s -t %d > %s/%s &" % (server.IP(), duration, args.output_dir, 'client' + filename + '.txt'))
        # Its statistics is useless, just throw away. 1990 just means a great number.
        #client.cmd("iperf -c %s -u -t %d &" % (server.IP(), 100000))
        sleep(3)

    # Wait for the traffic to become stable.
    sleep(10)

    # 2. Start bwm-ng to monitor throughput.
    monitor = Process(target=monitor_devs_ng, args=('%s/bwmng.txt' % args.output_dir, 1.0))
    monitor.start()

    # 3. The experiment is going on.
    sleep(duration + 5)

    # 4. Shut down.
    monitor.terminate()
    os.system('killall bwm-ng')
    os.system('killall iperf')


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


# def monitor_devs_ng(fname="./txrate.txt", interval_sec=0.1):
#     """
#             Use bwm-ng tool to collect interface transmit rate statistics.
#             bwm-ng Mode: rate;
#             interval time: 1s.
#     """
#     cmd = "sleep 1; bwm-ng -t %s -o csv -u bits -T rate -C ',' > %s" % (interval_sec * 1000, fname)
#     Popen(cmd, shell=True).wait()


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
        exit(1)
    # createTopo(2, 1)
    net, topo = topo_ecmp.createECMPTopo(4, 2)
    # net, topo = createNonBlockTopo(4, 2)
    c0 = RemoteController('c0', ip='127.0.0.1', port=6653)
    makeTerm(c0, cmd="./ryu/bin/ryu-manager --observe-links --ofp-tcp-listen-port 6653 network_monitor.py")
    #c0.cmdPrint('xterm  -T \"./ryu/bin/ryu-manager --observe-links network_monitor.py\" &')
    net.addController(c0)
    sleep(2)
    topo_ecmp.configureTopo(net, topo)
    # createTopo(8, 4)
    # dumpNodeConnections(net.hosts)
    # pingTest(net)

    # hosts = []
    # for hostname in topo.HostList:
    #     hosts.append(net.get(hostname))
    # trafficGen(args, hosts, net)
    # app = ConsoleApp(net, width=4)
    # iperfTest(net, topo)
    iperf_peers = [('h011', 'h012'), ('h004', 'h006'), ('h003', 'h004'), ('h007', 'h008'), ('h008', 'h007'), ('h009', 'h010'), ('h013', 'h014'), ('h016', 'h013'),
                   ('h002', 'h003'), ('h006', 'h013'), ('h010', 'h009'), ('h012', 'h011'), ('h001', 'h002'), ('h015', 'h002'), ('h005', 'h006'), ('h014', 'h013')]
    # 2. Generate traffics and test the performance of the network.
    topo_ecmp.connect_controller(net, topo, c0)
    traffic_generation(net, topo, iperf_peers, 20)
    CLI(net)
    net.stop()
    clean()


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
