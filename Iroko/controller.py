import time, thread
import socket
from random import randint

i_h_map = {'3001-eth3': "192.168.10.1", '3001-eth4': "192.168.10.2", '3002-eth3': "192.168.10.3", '3002-eth4': "192.168.10.4",
           '3003-eth3': "192.168.10.5", '3003-eth4': "192.168.10.6", '3004-eth3': "192.168.10.7", '3004-eth4': "192.168.10.8",
           '3005-eth3': "192.168.10.9", '3005-eth4': "192.168.10.10", '3006-eth3': "192.168.10.11", '3006-eth4': "192.168.10.12",
           '3007-eth3': "192.168.10.13", '3007-eth4': "192.168.10.14", '3008-eth3': "192.168.10.15", '3008-eth4': "192.168.10.16", }

class IrokoController:

    def __init__(self):
        print "Initializing controller"
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        thread.start_new_thread(self.controller_main, ());

    def controller_main(self):
        print "Starting controller"

        while True:
            time.sleep(1)
            for iface in i_h_map:
                txrate = randint(1310720, 2621440)
                self.send_cntrl_pckt(iface, txrate)

    def send_cntrl_pckt(self, interface, txrate):
        ip = "192.168.5." + i_h_map[interface].split('.')[-1]
        port = 20130
        pckt = str(txrate) + '\0'
        #print "interface: %s, ip: %s, rate: %s" % (interface, ip, txrate)
        self.sock.sendto(pckt, (ip, port))
