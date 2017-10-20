'''
@author: Milad Sharif (msharif@stanfor.edu)
'''

from mininet.topo import Topo
from mininet.node import Controller, RemoteController, OVSKernelSwitch, CPULimitedHost
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.util import custom
from mininet.log import setLogLevel, info, warn, error, debug

from DCTopo import FatTreeTopo, NonBlockingTopo
from DCRouting import Routing

from subprocess import Popen, PIPE
from argparse import ArgumentParser
import multiprocessing
from time import sleep
from monitor.monitor import monitor_devs_ng
import os
import sys


# Number of pods in Fat-Tree 
K = 4

# Queue Size
QUEUE_SIZE = 100

# Link capacity (Mbps)
BW = 10 

parser = ArgumentParser(description="ECMP routing")

parser.add_argument('-d', '--dir', dest='output_dir', default='log',
        help='Output directory')

parser.add_argument('-i', '--input', dest='input_file',
        default='inputs/all_to_all_data',
        help='Traffic generator input file')

parser.add_argument('-t', '--time', dest='time', type=int, default=30,
        help='Duration (sec) to run the experiment')

parser.add_argument('-p', '--cpu', dest='cpu', type=float, default=-1,
        help='cpu fraction to allocate to each host')

parser.add_argument('-n', '--nonblocking', dest='nonblocking', default=False,
        action='store_true', help='Run the experiment on the noneblocking topo')

parser.add_argument('--iperf', dest='iperf', default=False, action='store_true',
        help='Use iperf to generate traffics')

parser.add_argument('--hedera',dest='hedera', default=False,
        action='store_true', help='Run the experiment with hedera GFF scheduler')

parser.add_argument('--ecmp',dest='ECMP',default=False,
        action='store_true',help='Run the experiment with ECMP routing')

args = parser.parse_args()

def NonBlockingNet(k=4, bw=10, cpu=-1, queue=100):
    ''' Create a NonBlocking Net '''

    topo = NonBlockingTopo(k)
    host = custom(CPULimitedHost, cpu=cpu)
    link = custom(TCLink, bw=bw, max_queue_size=queue)

    net = Mininet(topo, host=host, link=link, switch=OVSKernelSwitch,
            controller=Controller)

    return net

def FatTreeNet(k=4, bw=10, cpu=-1, queue=100,controller='HController'):
    ''' Create a Fat-Tree network '''

    pox_c = Popen("./pox/pox.py %s --topo=ft,4 --routing=ECMP"%controller, shell=True)

    info('*** Creating the topology')
    topo = FatTreeTopo(k)

    host = custom(CPULimitedHost, cpu=cpu)
    link = custom(TCLink, bw=bw, max_queue_size=queue)
    
    net = Mininet(topo, host=host, link=link, switch=OVSKernelSwitch,
            controller=RemoteController)

    return net

def start_tcpprobe():
    ''' Install tcp_probe module and dump to file '''
    os.system("rmmod tcp_probe; modprobe tcp_probe full=1;")
    Popen("cat /proc/net/tcpprobe > ./tcp.txt" , shell=True)

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
        server.popen('iperf -s -p %s > ./server.txt' % port, shell = True)

        client = host_list[src_ip]
        client.popen('iperf -c %s -p %s -t %d > ./client.txt' 
                % (server.IP(), port, args.time ), shell=True)

    monitor = multiprocessing.Process(target = monitor_devs_ng, args =
                ('%s/rate.txt' % args.output_dir, 0.01))

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
    

    monitor = multiprocessing.Process(target = monitor_devs_ng, args =
        ('%s/rate.txt' % args.output_dir, 0.01))

    monitor.start()

    sleep(args.time)

    monitor.terminate()

    info('*** Stopping load-generators\n')
    for h in hosts:
        h.cmd('killall loadgen')

def FatTreeTest(args,controller):
    net = FatTreeNet( k=K, cpu=args.cpu, bw=BW, queue=QUEUE_SIZE,
            controller=controller)
    net.start()

    # wait for the switches to connect to the controller
    info('** Waiting for switches to connect to the controller\n')
    sleep(5)

    hosts = net.hosts
    
    if args.iperf:
        iperfTrafficGen(args, hosts, net)
    else:
        trafficGen(args, hosts, net)

    net.stop()

def NonBlockingTest(args):
    net = NonBlockingNet(k=K, cpu=args.cpu, bw=BW, queue=QUEUE_SIZE)
    net.start()

    info('** Waiting for switches to connect to the controller\n')
    sleep(1)

    hosts = net.hosts

    if args.iperf:
        iperfTrafficGen(args, hosts, net)
    else:
        trafficGen(args, hosts, net)

    net.stop()

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

if __name__ == '__main__':

    setLogLevel( 'info' )
    if not os.path.exists(args.output_dir):
        print args.output_dir
        os.makedirs(args.output_dir)

    clean()

    if args.nonblocking:
        NonBlockingTest(args)
    elif args.ECMP:
        FatTreeTest(args,controller='DCController')
    elif args.hedera:
        FatTreeTest(args,controller='HController')
    else:
        info('**error** please specify either hedera, ecmp, or nonblocking\n')
        
    clean()

    Popen("killall -9 top bwm-ng", shell=True).wait()
    os.system('sudo mn -c')
