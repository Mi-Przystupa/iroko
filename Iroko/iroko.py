import os
import logging
import multiprocessing

from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI

from time import sleep
from mininet.node import OVSKernelSwitch, CPULimitedHost
from mininet.util import custom
from mininet.log import setLogLevel, info, warn, error, debug

from subprocess import Popen, PIPE
from argparse import ArgumentParser
from monitor.monitor import monitor_devs_ng
from monitor.monitor import monitor_qlen
from multiprocessing import Process
from mininet.term import makeTerm
from mininet.link import TCLink
from hedera.DCTopo import FatTreeTopo

import topo_ecmp
import topo_non_block

MAX_QUEUE = 50

parser = ArgumentParser(description="Iroko Parser")

parser.add_argument('-d', '--dir', dest='output_dir', default='log',
                    help='Output directory')

parser.add_argument('-i', '--input', dest='input_file',
                    default='../inputs/stag_prob_0_2_3_data',
                    help='Traffic generator input file')

parser.add_argument('-t', '--time', dest='time', type=int, default=60,
                    help='Duration (sec) to run the experiment')

parser.add_argument('-p', '--cpu', dest='cpu', type=float, default=-1,
                    help='cpu fraction to allocate to each host')

parser.add_argument('-n', '--nonblocking', dest='nonblocking', default=False,
                    action='store_true', help='Run the experiment on the noneblocking topo')

parser.add_argument('--iperf', dest='iperf', default=False, action='store_true',
                    help='Use iperf to generate traffics')

parser.add_argument('--hedera', dest='hedera', default=False,
                    action='store_true', help='Run the experiment with hedera GFF scheduler')

parser.add_argument('--ecmp', dest='ECMP', default=False,
                    action='store_true', help='Run the experiment with ECMP routing')

parser.add_argument('--iroko', dest='iroko', default=False,
                    action='store_true', help='Run the experiment with Iroko rate limiting')

parser.add_argument('--dctcp', dest='dctcp', default=False,
                    action='store_true', help='Run the experiment with DCTCP congestion control')

args = parser.parse_args()


def start_tcpprobe():
    os.system("killall -9 cat")
    ''' Install tcp_probe module and dump to file '''
    os.system("rmmod tcp_probe; modprobe tcp_probe full=1;")
    Popen("cat /proc/net/tcpprobe > ./tcp.txt", shell=True)


def stop_tcpprobe():
    os.system("killall -9 cat")


def enable_tcp_ecn():
    Popen("sysctl -w net.ipv4.tcp_ecn=1", shell=True).wait()


def disable_tcp_ecn():
    Popen("sysctl -w net.ipv4.tcp_ecn=0", shell=True).wait()


def enable_dctcp():
    Popen("sysctl -w net.ipv4.tcp_congestion_control=dctcp", shell=True).wait()
    enable_tcp_ecn()


def disable_dctcp():
    Popen("sysctl -w net.ipv4.tcp_congestion_control=cubic", shell=True).wait()
    disable_tcp_ecn()


def get_intf_list(net):
    switches = net.switches
    sw_intfs = []
    for sw in switches:
        for intf in sw.intfNames():
            if intf is not 'lo':
                sw_intfs.append(intf)
    return sw_intfs

def trafficGen(args, hosts, net):
    ''' Run the traffic generator and monitor all of the interfaces '''
    listen_port = 12345
    sample_period_us = 1000000

    traffic_gen = 'cluster_loadgen/loadgen'
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

    ifaces = get_intf_list(net)

    monitor1 = multiprocessing.Process(target=monitor_devs_ng, args=('%s/rate.txt' % args.output_dir, 0.01))
    monitor2 = multiprocessing.Process(target=monitor_qlen, args=(ifaces, 1, '%s/qlen.txt' % args.output_dir))

    monitor1.start()
    monitor2.start()

    sleep(args.time)
    info('*** Stopping monitor\n')
    monitor1.terminate()
    monitor2.terminate()

    os.system("killall bwm-ng")

    info('*** Stopping load-generators\n')
    for h in hosts:
        h.cmd('killall loadgen')


# def traffic_generation(net, topo, duration):
#     """
#             Generate traffics and test the performance of the network.
#     """
#     flows_peers = [('h011', 'h012'), ('h004', 'h006'), ('h003', 'h004'), ('h007', 'h008'), ('h008', 'h007'), ('h009', 'h010'), ('h013', 'h014'), ('h016', 'h013'),
#                    ('h002', 'h003'), ('h006', 'h013'), ('h010', 'h009'), ('h012', 'h011'), ('h001', 'h002'), ('h015', 'h002'), ('h005', 'h006'), ('h014', 'h013')]
#     print("Running iperf test")
#     # 1. Start iperf. (Elephant flows)
#     # Start the servers.
#     serversList = set([peer[1] for peer in flows_peers])
#     for server in serversList:
#         filename = server[1:]
#         server = net.get(server)
#         server.cmd("iperf -s > %s/%s &" % (args.output_dir, 'server' + filename + '.txt'))
#         # server.cmd("iperf -s > /dev/null &")   # Its statistics is useless, just throw away.
#     monitor = multiprocessing.Process(target=monitor_devs_ng, args=('%s/rate.txt' % args.output_dir, 0.01))
#     sleep(3)

#     # Start the clients.
#     for src, dest in flows_peers:
#         server = net.get(dest)
#         client = net.get(src)
#         filename = src[1:]
#         client.cmd("iperf -c %s -t %d > %s/%s &" % (server.IP(), duration, args.output_dir, 'client' + filename + '.txt'))
#         # Its statistics is useless, just throw away. 1990 just means a great number.
#         #client.cmd("iperf -c %s -u -t %d &" % (server.IP(), 100000))
#         sleep(3)

#     # Wait for the traffic to become stable.
#     sleep(10)

#     # 2. Start bwm-ng to monitor throughput.
#     monitor = Process(target=monitor_devs_ng, args=('%s/bwmng.txt' % args.output_dir, 1.0))
#     monitor.start()

#     # 3. The experiment is going on.
#     sleep(duration + 5)

#     # 4. Shut down.
#     monitor.terminate()
#     Popen("killall -9 top bwm-ng", shell=True).wait()
#     os.system('killall iperf')

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
    if args.dctcp:
        disable_dctcp()
    p_pox = Popen("ps aux | grep 'pox' | awk '{print $2}'",
                  stdout=PIPE, shell=True)
    p_pox.wait()
    procs = (p_pox.communicate()[0]).split('\n')
    p_ryu = Popen("ps aux | grep 'ryu' | awk '{print $2}'",
                  stdout=PIPE, shell=True)
    p_ryu.wait()
    procs.extend((p_ryu.communicate()[0]).split('\n'))
    for pid in procs:
        try:
            pid = int(pid)
            Popen('kill %d' % pid, shell=True).wait()
        except:
            pass
    Popen('killall iperf3', shell=True).wait()
    Popen('killall xterm', shell=True).wait()
    Popen('killall python2.7', shell=True).wait()


def pingTest(net):
    """
        Start ping test.
    """
    net.pingAll()


def FatTreeTest(args, controller=None):
    net, topo = topo_ecmp.createECMPTopo(pod=4, density=2, cpu=args.cpu, dctcp=args.dctcp)
    ovs_v = 13  # default value
    is_ecmp = True  # default value

    if controller is not None:
        c0 = RemoteController('c0', ip='127.0.0.1', port=6653)
        if controller == "Iroko":
            Popen("sudo python iroko_controller.py > controller.log", shell=True)
            #     #makeTerm(c0, cmd="./ryu/bin/ryu-manager --observe-links --ofp-tcp-listen-port 6653 network_monitor.py")
            #     #makeTerm(c0, cmd="sudo python iroko_controller.py")
            # elif controller == "HController":
            #     c0 = RemoteController('c0', ip='127.0.0.1', port=6653)
            #     # makeTerm(c0, cmd="hedera/pox/pox.py HController --topo=ft,4 --routing=ECMP")
            #     Popen("hedera/pox/pox.py HController --topo=ft,4 --routing=ECMP", shell=True)
            #     ovs_v = 10
            #     is_ecmp = False
        net.addController(c0)

    net.start()
    sleep(2)
    topo_ecmp.configureTopo(net, topo, ovs_v, is_ecmp)

    if controller is not None:
        topo_ecmp.connect_controller(net, topo, c0)
        info('** Waiting for switches to connect to the controller\n')
        sleep(2)
    hosts = net.hosts
    if args.dctcp:
        enable_dctcp()
    if args.dctcp:
        for host in hosts:
            host_o = net.get(host)
            host_o.cmd("sysctl -w net.ipv4.tcp_ecn=1")
            host_o.cmd("sysctl -w net.ipv4.tcp_congestion_control=dctcp")

    trafficGen(args, hosts, net)
    net.stop()

def NonBlockingTest(args):

    net, topo = topo_non_block.createNonBlockTopo(pod=4, cpu=args.cpu)
    net.start()
    topo_non_block.configureTopo(net, topo)

    info('** Waiting for switches to connect to the controller\n')
    sleep(2)

    hosts = net.hosts
    trafficGen(args, hosts, net)

    net.stop()


def HederaNet(k=4, bw=10, cpu=-1, queue=100, controller='HController'):
    ''' Create a Fat-Tree network '''

    info('*** Creating the topology')
    topo = FatTreeTopo(k)

    host = custom(CPULimitedHost, cpu=cpu)
    link = custom(TCLink, bw=bw, max_queue_size=queue)

    net = Mininet(topo, host=host, link=link, switch=OVSKernelSwitch,
                  controller=None)

    return net


def HederaTest(args):
    c0 = RemoteController('c0', ip='127.0.0.1', port=6653)
    # makeTerm(c0, cmd="hedera/pox/pox.py HController --topo=ft,4 --routing=ECMP")
    Popen("hedera/pox/pox.py HController --topo=ft,4 --routing=ECMP", shell=True)
    net = HederaNet(k=4, cpu=args.cpu, bw=10, queue=MAX_QUEUE,
                    controller=None)
    net.addController(c0)
    net.start()
    # wait for the switches to connect to the controller
    info('** Waiting for switches to connect to the controller\n')
    sleep(5)

    hosts = net.hosts

    # if args.iperf:
    #     iperfTrafficGen(args, hosts, net)
    # else:
    trafficGen(args, hosts, net)

    net.stop()


if __name__ == '__main__':
    clean()
    setLogLevel('info')
    if not os.path.exists(args.output_dir):
        print(args.output_dir)
        os.makedirs(args.output_dir)

    if os.getuid() != 0:
        logging.debug("You are NOT root")
        exit(1)
    if args.nonblocking:
        NonBlockingTest(args)
    elif args.ECMP:
        FatTreeTest(args, controller=None)
    elif args.hedera:
        HederaTest(args)
    elif args.iroko:
        FatTreeTest(args, controller='Iroko')
    else:
        info('**error** please specify either hedera, iroko, ecmp, or nonblocking\n')
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
