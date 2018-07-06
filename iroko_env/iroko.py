import os
import logging
import multiprocessing

from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI

from time import sleep
from mininet.node import OVSKernelSwitch, CPULimitedHost
from mininet.util import custom
from mininet.log import setLogLevel, info, output, warn, error, debug

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
import topo_dumbbell

MAX_QUEUE = 5000
PARSER = ArgumentParser(description="Iroko PARSER")

PARSER.add_argument('-d', '--dir', dest='output_dir', default='log',
                    help='Output directory')

PARSER.add_argument('-i', '--input', dest='input_file',
                    default='../inputs/stag_prob_0_2_3_data',
                    help='Traffic generator input file')

PARSER.add_argument('-t', '--time', dest='time', type=int, default=60,
                    help='Duration (sec) to run the experiment')

PARSER.add_argument('-p', '--cpu', dest='cpu', type=float, default=-1,
                    help='cpu fraction to allocate to each host')

PARSER.add_argument('-n', '--nonblocking', dest='nonblocking', default=False,
                    action='store_true', help='Run the experiment on the noneblocking topo')

PARSER.add_argument('--iperf', dest='iperf', default=False, action='store_true',
                    help='Use iperf to generate traffics')

PARSER.add_argument('--hedera', dest='hedera', default=False,
                    action='store_true', help='Run the experiment with hedera GFF scheduler')

PARSER.add_argument('--ecmp', dest='ECMP', default=False,
                    action='store_true', help='Run the experiment with ECMP routing')

PARSER.add_argument('--iroko', dest='iroko', default=False,
                    action='store_true', help='Run the experiment with Iroko rate limiting')

PARSER.add_argument('--dctcp', dest='dctcp', default=False,
                    action='store_true', help='Run the experiment with DCTCP congestion control')
PARSER.add_argument('--dumbbell', dest='dumbbell', default=False,
                    action='store_true', help='Run the experiment with a dumbbell topology.')

PARSER.add_argument('--agent', dest='agent', default='A',
                    help='options are A, B, C, D')

PARSER.add_argument('--dumbbell_env', dest='dumbbell_env', default='False', action='store_true', help='Just run traffic generator')
ARGS = PARSER.parse_args()


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
    for switch in switches:
        for intf in switch.intfNames():
            if intf is not 'lo':
                sw_intfs.append(intf)
    return sw_intfs


def gen_traffic(net):
    ''' Run the traffic generator and monitor all of the interfaces '''
    listen_port = 12345
    sample_period_us = 1000000
    hosts = net.hosts
    traffic_gen = 'cluster_loadgen/loadgen'
    if not os.path.isfile(traffic_gen):
        error('The traffic generator doesn\'t exist. \ncd hedera/cluster_loadgen; make\n')
        return

    output('*** Starting load-generators\n %s\n' % ARGS.input_file)
    for host in hosts:
        tg_cmd = ('%s -f %s -i %s -l %d -p %d 2&>1 > %s/%s.out &' %
                  (traffic_gen, ARGS.input_file, host.defaultIntf(), listen_port, sample_period_us, ARGS.output_dir, host.name))
        host.cmd(tg_cmd)

    sleep(1)

    output('*** Triggering load-generators\n')
    for host in hosts:
        host.cmd('nc -nzv %s %d' % (host.IP(), listen_port))
    ifaces = get_intf_list(net)

    monitor1 = multiprocessing.Process(
        target=monitor_devs_ng, args=('%s/rate.txt' % ARGS.output_dir, 0.01))
    monitor2 = multiprocessing.Process(target=monitor_qlen, args=(
        ifaces, 1, '%s/qlen.txt' % ARGS.output_dir))

    monitor1.start()
    monitor2.start()

    sleep(ARGS.time)
    output('*** Stopping monitor\n')
    monitor1.terminate()
    monitor2.terminate()

    os.system("killall bwm-ng")

    output('*** Stopping load-generators\n')
    for host in hosts:
        host.cmd('killall loadgen')


def kill_controller():
    p_pox = Popen("ps aux | grep -E 'pox|ryu|iroko_controller' | awk '{print $2}'",
                  stdout=PIPE, shell=True)
    p_pox.wait()
    procs = (p_pox.communicate()[0]).split('\n')
    # p_ryu = Popen("ps aux | grep 'ryu' | awk '{print $2}'",
    #               stdout=PIPE, shell=True)
    # p_ryu.wait()
    # procs.extend((p_ryu.communicate()[0]).split('\n'))
    # p_ryu = Popen("ps aux | grep 'ryu' | awk '{print $2}'",
    #               stdout=PIPE, shell=True)
    # p_ryu.wait()
    # procs.extend((p_ryu.communicate()[0]).split('\n'))
    for pid in procs:
        try:
            pid = int(pid)
            Popen('kill %d' % pid, shell=True).wait()
        except Exception as e:
            pass


def clean():
    ''' Clean any the running instances of POX '''
    if ARGS.dctcp:
        disable_dctcp()
    Popen('killall iperf3', shell=True).wait()
    Popen('killall xterm', shell=True).wait()
    Popen('killall python2.7', shell=True).wait()


def test_ping(net):
    """
        Start ping test.
    """
    net.pingAll()


def test_iperf(net):
    """
        Start iperf test.
        Currently uses a kind off stupid solution. Needs improvement.
    """
    hosts = net.hosts
    clients = []
    servers = []
    output('*** Starting iperf servers\n')
    for i, host in enumerate(hosts):
        if i >= len(hosts) / 2:
            cmd = "iperf -s &"
            host.cmd(cmd)
            servers.append(host)
        else:
            clients.append(host)
    sleep(1)
    output('*** Triggering iperf\n')
    for i, client in enumerate(clients):
        cmd = "iperf -c %s -u -b 10m -t %d &" % (servers[i].IP(), ARGS.time)
        client.cmdPrint(cmd)
    ifaces = get_intf_list(net)

    monitor1 = multiprocessing.Process(
        target=monitor_devs_ng, args=('%s/rate.txt' % ARGS.output_dir, 0.01))
    monitor2 = multiprocessing.Process(target=monitor_qlen, args=(
        ifaces, 1, '%s/qlen.txt' % ARGS.output_dir))

    monitor1.start()
    monitor2.start()

    sleep(ARGS.time)
    output('*** Stopping monitor\n')
    monitor1.terminate()
    monitor2.terminate()

    os.system("killall bwm-ng")
    output('*** Stopping iperf instances\n')
    for h in hosts:
        h.cmd('killall iperf')


def test_fattree(controller=None):
    net, topo = topo_ecmp.create_ecmp_topo(
        pod=4, density=2, cpu=ARGS.cpu, max_queue=MAX_QUEUE, dctcp=ARGS.dctcp)
    ovs_v = 13  # default value
    is_ecmp = True  # default value

    if controller is not None:
        c0 = RemoteController('c0', ip='127.0.0.1', port=6653)
        net.addController(c0)

    net.start()

    sleep(1)
    topo_ecmp.config_topo(net, topo, ovs_v, is_ecmp)

    if controller is not None:
        topo_ecmp.connect_controller(net, topo, c0)
        if controller == "Iroko":
            Popen("sudo python iroko_controller.py --agent %s" %
                  ARGS.agent, shell=True)
        output('** Waiting for switches to connect to the controller\n')
        sleep(1)
    if ARGS.dctcp:
        enable_dctcp()
    if ARGS.dctcp:
        for host in topo.hostList:
            host_o = net.get(host)
            host_o.cmd("sysctl -w net.ipv4.tcp_ecn=1")
            host_o.cmd("sysctl -w net.ipv4.tcp_congestion_control=dctcp")
    gen_traffic(net)
    kill_controller()
    net.stop()


def test_non_block():
    net, topo = topo_non_block.create_non_block_topo(
        pod=4, cpu=ARGS.cpu, max_queue=MAX_QUEUE, bw=10)
    net.start()
    topo_non_block.config_topo(net, topo)

    output('** Waiting for switches to connect to the controller\n')
    sleep(1)

    gen_traffic(net)
    net.stop()


def create_hedera_net(k=4, bw=10, cpu=-1, max_queue=100, controller='HController'):
    ''' Create a Fat-Tree network '''

    output('*** Creating the topology')
    topo = FatTreeTopo(k)

    host = custom(CPULimitedHost, cpu=cpu)
    link = custom(TCLink, bw=bw, max_queue_size=max_queue)

    net = Mininet(topo, host=host, link=link, switch=OVSKernelSwitch,
                  controller=None)
    return net


def test_hedera():
    c0 = RemoteController('c0', ip='127.0.0.1', port=6653)
    # makeTerm(c0, cmd="hedera/pox/pox.py HController --topo=ft,4 --routing=ECMP")
    Popen("hedera/pox/pox.py HController --topo=ft,4 --routing=ECMP", shell=True)
    net = create_hedera_net(k=4, cpu=ARGS.cpu, bw=10, max_queue=MAX_QUEUE,
                            controller=None)
    net.addController(c0)
    net.start()
    # wait for the switches to connect to the controller
    output('** Waiting for switches to connect to the controller\n')
    sleep(1)

    gen_traffic(net)
    kill_controller()
    net.stop()


def test_dumbbell():
    net, topo = topo_dumbbell.create_db_topo(
        hosts=4, cpu=ARGS.cpu, max_queue=MAX_QUEUE, bw=10)
    ovs_v = 13  # default value
    is_ecmp = True  # default value

    net.start()
    c0 = RemoteController('c0', ip='127.0.0.1', port=6653)
    net.addController(c0)

    topo_dumbbell.config_topo(net, topo, ovs_v, is_ecmp)
    topo_ecmp.connect_controller(net, topo, c0)
    Popen("sudo python iroko_controller.py --agent %s" %
          ARGS.agent, shell=True)
    #     #makeTerm(c0, cmd="./ryu/bin/ryu-manager --observe-links --ofp-tcp-listen-port 6653 network_monitor.py")
    #     #makeTerm(c0, cmd="sudo python iroko_controller.py")
    output('** Waiting for switches to connect to the controller\n')
    sleep(2)
    # iperfTest(ARGS, net)
    gen_traffic(net)

    kill_controller()
    net.stop()

def test_dumbbell_env():
    net, topo = topo_dumbbell.create_db_topo(
        hosts=4, cpu=ARGS.cpu, max_queue=MAX_QUEUE, bw=10)
    ovs_v = 13  # default value
    is_ecmp = True  # default value

    net.start()
    c0 = RemoteController('c0', ip='127.0.0.1', port=6653)
    net.addController(c0)

    topo_dumbbell.config_topo(net, topo, ovs_v, is_ecmp)
    topo_ecmp.connect_controller(net, topo, c0)
    #Popen("sudo python iroko_controller.py --agent %s" %
    #      ARGS.agent, shell=True)
    #     #makeTerm(c0, cmd="./ryu/bin/ryu-manager --observe-links --ofp-tcp-listen-port 6653 network_monitor.py")
    #     #makeTerm(c0, cmd="sudo python iroko_controller.py")
    # iperfTest(ARGS, net)
    gen_traffic(net)
    net.stop()



if __name__ == '__main__':
    kill_controller()
    clean()
    setLogLevel('output')
    if not os.path.exists(ARGS.output_dir):
        print(ARGS.output_dir)
        os.makedirs(ARGS.output_dir)
    if ARGS.dctcp:
        ARGS.ECMP = True
    if os.getuid() != 0:
        logging.error("You are NOT root")
        exit(1)
    if ARGS.nonblocking:
        test_non_block()
    elif ARGS.ECMP:
        test_fattree(controller=None)
    elif ARGS.hedera:
        test_hedera()
    elif ARGS.iroko:
        test_fattree(controller='Iroko')
    elif ARGS.dumbbell:
        test_dumbbell()
    elif ARGS.dumbbell_env:
        test_dumbbell_env()
    
	
    else:
        error('Please specify either hedera, iroko, ecmp, or nonblocking!\n')
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
