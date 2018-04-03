import subprocess
import re
import time
import threading

MAX_CAPACITY = 10e6   # Max capacity of link


class StatsCollector(threading.Thread):
    """
            NetworkMonitor is a Ryu app for collecting traffic information.
    """

    def __init__(self):
        threading.Thread.__init__(self)
        self.name = 'StatsCollector'
        self.prev_overlimits = 0
        self.prev_loss = 0
        self.prev_util = 0
        self.prev_overlimits_d = 0
        self.prev_loss_d = 0
        self.prev_util_d = 0
        self.iface_list = []
        self.prev_bandwidth = {}
        self.bws_rx = {}
        self.bws_tx = {}
        self.free_bandwidths = {}
        self.drops = {}
        self.overlimits = {}
        self.queues = {}
        self.terminate = False

    def run(self):
        # self._set_interfaces()
        while True:
            if self.terminate:
                self.exit()
            self._collect_stats()

    def terminate(self):
        self.terminate = True

    def set_interfaces(self):
        cmd = "sudo ovs-vsctl list-br | xargs -L1 sudo ovs-vsctl list-ports"
        output = subprocess.check_output(cmd, shell=True)
        iface_list_temp = []
        for row in output.split('\n'):
            if row != '':
                iface_list_temp.append(row)
        # return_list = [iface for iface in iface_list_temp if iface in i_h_map]  # filter against actual hosts
        self.iface_list = iface_list_temp

    def _get_deltas(self, curr_loss, curr_overlimits):
        # loss_d = max((curr_loss - self.prev_loss) - self.prev_loss_d, 0)

        # losses
        loss_d = curr_loss - self.prev_loss  # loss in epoch

        # if loss_d > self.prev_loss_d:
        #     loss_increase = 1  # loss increasing?
        # else:
        #     loss_increase = 0
        # loss in previous epoch
        self.prev_loss_d = loss_d
        self.prev_loss = curr_loss

        # overlimits
        overlimits_d = curr_overlimits - self.prev_overlimits  # overlimints in epoch

        # if overlimits_d > self.prev_overlimits_d:
        #     overlimits_increase = 1
        # else:
        #     overlimits_increase = 0

        self.prev_overlimits_d = overlimits_d
        self.prev_overlimits = curr_overlimits

        # utilization
        # util_d = curr_util - self.prev_util

        # if util_d > 0:
        #     util_increase = 1
        # else:
        #     util_increase = 0

        # self.prev_util = curr_util

        # print
        # print("Deltas: Loss %d Overlimits %d Utilization: %d " % (loss_d, overlimits_d, util_d))
        # print("Increases: Loss %d Overlimits %d Utilization: %d " % (loss_increase, overlimits_increase, util_increase))
        return loss_d, overlimits_d  # , util_d

    def _get_bandwidths(self, iface_list):
        # cmd3 = "ifstat -i %s -q 0.1 1 | awk '{if (NR==3) print $2}'" % (iface)
        processes = []
        bws_rx = {}
        bws_tx = {}
        # iface_string = ",".join(iface_list )
        for iface in iface_list:
            cmd = (" ifstat -i %s -b -q 0.1 1 | awk \'{if (NR==3) print $0}\' | \
                   awk \'{$1=$1}1\' OFS=\", \"" % (iface))  # | sed \'s/\(\([^,]*,\)\{1\}[^,]*\),/\1;/g\'
            # output = subprocess.check_output(cmd, shell=True)
            p = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
            processes.append((p, iface))

        for p, iface in processes:
            p.wait()
            output, err = p.communicate()
            bw = output.split(',')
            try:
                bws_rx[iface] = float(bw[0]) * 1000
                bws_tx[iface] = float(bw[1]) * 1000
            except Exception as e:
                # print("Empty Request %s" % e)
                bws_rx[iface] = 0
                bws_tx[iface] = 0
        return bws_rx, bws_tx

    def _get_bandwidths_old(self, iface_list):
        # cmd3 = "ifstat -i %s -q 0.1 1 | awk '{if (NR==3) print $2}'" % (iface)
        bytes_old = {}
        bytes_new = {}
        for iface in iface_list:
            cmd = "awk \"/^ *%s: / \"\' { if ($1 ~ /.*:[0-9][0-9]*/) { sub(/^.*:/, \"\") ; print $1 }\
            else { print $2 } }\' /proc/net/dev" % (iface)
            try:
                output = subprocess.check_output(cmd, shell=True)
                bytes_old[iface] = float(output)
            except Exception as e:
                # print("Empty Request %s" % e)
                output = 0
        time.sleep(1)
        for iface in iface_list:
            cmd = "awk \"/^ *%s: / \"\' { if ($1 ~ /.*:[0-9][0-9]*/) { sub(/^.*:/, \"\") ; print $1 }\
             else { print $2 } }\' /proc/net/dev" % (iface)
            try:
                output = subprocess.check_output(cmd, shell=True)
                bytes_new[iface] = float(output)
            except Exception as e:
                # print("Empty Request %s" % e)
                output = 0
        curr_bandwidth = {key: (bytes_new[key] - bytes_old.get(key, 0)) * 8 for key in bytes_new.keys()}

        # Get bandwidth deltas

        bandwidth_d = {}
        if self.prev_bandwidth == {}:
            bandwidth_d = curr_bandwidth  # base case
        else:
            for iface in curr_bandwidth:
                bandwidth_d[iface] = curr_bandwidth[iface] - self.prev_bandwidth[iface]  # calculate delta
        self.prev_bandwidth = curr_bandwidth

        return curr_bandwidth, bandwidth_d

    def _get_free_bw(self, capacity, speed):
        # freebw: Kbit/s
        return max(capacity - speed * 8 / 1000.0, 0)

    def _get_free_bandwidths(self, bandwidths):
        free_bandwidths = {}
        for iface, bandwidth in bandwidths.iteritems():
            free_bandwidths[iface] = self._get_free_bw(MAX_CAPACITY, bandwidth)
        return free_bandwidths

    # def _get_flow_stats(self, iface_list):
    #     for iface in iface_list:
    #         cmd = "sudo tcpdump -l -i %s -n -c 5 ip 2>/dev/null | " +
    #             "grep -P -o \'([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+).*? > ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\' |"+
    #             "grep -P -o \'[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\' | xargs -n 2 echo | awk '!a[$0]++\'" % (iface)
    #         output = subprocess.check_output(cmd, shell=True)
    #         print(output)

    def _get_qdisc_stats(self, iface_list):
        drops = {}
        overlimits = {}
        queues = {}

        re_dropped = re.compile(r'(?<=dropped )[ 0-9]*')
        re_overlimit = re.compile(r'(?<=overlimits )[ 0-9]*')
        re_queued = re.compile(r'backlog\s[^\s]+\s([\d]+)p')
        for iface in iface_list:
            cmd = "tc -s qdisc show dev %s" % (iface)
            # cmd1 = "tc -s qdisc show dev %s | grep -ohP -m1 '(?<=dropped )[ 0-9]*'" % (iface)
            # cmd2 = "tc -s qdisc show dev %s | grep -ohP -m1 '(?<=overlimits )[ 0-9]*'" % (iface)
            # cmd2 = "tc -s qdisc show dev %s | grep -ohP -m1 '(?<=backlog )[ 0-9]*'" % (iface)
            dr = {}
            ov = {}
            qu = {}
            try:
                output = subprocess.check_output(cmd, shell=True)
                dr = re_dropped.findall(output)
                ov = re_overlimit.findall(output)
                qu = re_queued.findall(output)
            except Exception as e:
                # print("Empty Request %s" % e)
                dr[0] = 0
                ov[0] = 0
                qu[0] = 0
            drops[iface] = int(dr[0])
            overlimits[iface] = int(ov[0])
            queues[iface] = int(qu[0])
        return drops, overlimits, queues

    def _collect_stats(self):
        # iface_list = self._get_interfaces()
        self.bws_rx, self.bws_tx = self._get_bandwidths(self.iface_list)
        self.drops, self.overlimits, self.queues = self._get_qdisc_stats(self.iface_list)
        # self.free_bandwidths = self._get_free_bandwidths(self.bandwidths)
        # self.drops, self.overlimits, self.queues = self._get_qdisc_stats(self.iface_list)

    def _compute_delta(self, iface, bw_rx, bw_tx, drops, overlimits, queues):
        deltas = {}
        if bw_rx <= self.bws_rx[iface]:
            deltas["delta_rx"] = 1
        else:
            deltas["delta_rx"] = 0

        if bw_tx <= self.bws_tx[iface]:
            deltas["delta_tx"] = 1
        else:
            deltas["delta_tx"] = 0

        if drops < self.drops[iface]:
            deltas["delta_d"] = 0
        else:
            deltas["delta_d"] = 1

        if overlimits < self.overlimits[iface]:
            deltas["delta_ov"] = 0
        else:
            deltas["delta_ov"] = 1

        if queues < self.queues[iface]:
            deltas["delta_q"] = 0
        else:
            deltas["delta_q"] = 1
        return deltas

    def init_deltas(self):
        d_vector = {}
        for iface in self.iface_list:
            d_vector[iface] = {}
            d_vector[iface]["delta_rx"] = 0
            d_vector[iface]["delta_tx"] = 0
            d_vector[iface]["delta_d"] = 0
            d_vector[iface]["delta_ov"] = 0
            d_vector[iface]["delta_q"] = 0
        return d_vector

    def get_interface_deltas(self, bws_rx, bws_tx, drops, overlimits, queues):
        d_vector = {}
        for iface in self.iface_list:
            d_vector[iface] = {}
            d_vector[iface] = self._compute_delta(iface, bws_rx[iface], bws_tx[iface], drops[iface], overlimits[iface], queues[iface])
        return d_vector

    def get_interface_stats(self):
        # self._get_flow_stats(self.iface_list)
        self.drops, self.overlimits, self.queues = self._get_qdisc_stats(self.iface_list)
        # drops_d = {}
        # overlimits_d = {}
        # for iface in self.iface_list:
        #     drops_d[iface], overlimits_d[iface] = self._get_deltas(self.drops[iface], self.overlimits[iface])
        return self.bws_rx, self.bws_tx, self.drops, self.overlimits, self.queues

    # def get_stats_sums(self, bandwidths, free_bandwidths, drops, overlimits, queues):
    #     bw_sum = sum(bandwidths.itervalues())
    #     bw_free_sum = sum(free_bandwidths.itervalues())
    #     loss_sum = sum(drops.itervalues())
    #     overlimit_sum = sum(overlimits.itervalues())
    #     queued_sum = sum(queues.itervalues())
    #     return bw_sum, bw_free_sum, loss_sum, overlimit_sum, queued_sum

    # def show_stat(self):
    #     '''
    #             Show statistics information according to data type.
    #             _type: 'port' / 'flow'
    #     '''
    #     #     print
    #     bandwidths, free_bandwidths, drops, overlimits, queues = self.get_interface_stats()
    #     bw_sum, bw_free_sum, loss_sum, overlimit_sum, queued_sum = self.get_stats_sums(
    #         bandwidths, free_bandwidths, drops, overlimits, queues)
    #     loss_d, overlimits_d, util_d, loss_increase, overlimits_increase, util_increase = self._get_deltas(
    #         loss_sum, overlimit_sum, bw_sum)
    #     print("Loss: %d Delta: %d Increase: %d" % (loss_sum, loss_d, loss_increase))
    #     print("Overlimits: %d Delta: %d Increase: %d" % (overlimit_sum, overlimits_d, overlimits_increase))
    #     print("Backlog: %d" % queued_sum)

    #     print("Free BW Sum: %d" % bw_free_sum)
    #     print("Current Util Sum: %f" % bw_sum)

    #     max_capacity_sum = MAX_CAPACITY * bandwidths.__len__()
    #     print("MAX_CAPACITY Sum: %d %d" % (max_capacity_sum, bandwidths.__len__()))
    #     total_util_ratio = (bw_sum) / max_capacity_sum
    #     total_util_avg = (bw_sum) / bandwidths.__len__()
    #     print("Current Average Utilization: %f" % total_util_avg)
    #     print("Current Ratio of Utilization: %f" % total_util_ratio)
    # sudo ovs-vsctl list-br | xargs -L1 sudo ovs-vsctl list-ports
    # sudo ovs-vsctl list-br | xargs -L1 sudo ovs-ofctl dump-ports -O Openflow13


class FlowCollector(threading.Thread):
    """
            NetworkMonitor is a Ryu app for collecting traffic information.
    """

    def __init__(self, host_ips):
        threading.Thread.__init__(self)
        self.name = 'FlowCollector'
        self.iface_list = []
        self.host_ips = host_ips
        self.terminate = False
        self.src_flows = {}
        self.dst_flows = {}
        self.set_interfaces()

    def run(self):
        # self._set_interfaces()
        while True:
            if self.terminate:
                self.exit()
            self._collect_flows()

    def terminate(self):
        self.terminate = True

    def set_interfaces(self):
        cmd = "sudo ovs-vsctl list-br | xargs -L1 sudo ovs-vsctl list-ports"
        output = subprocess.check_output(cmd, shell=True)
        iface_list_temp = []
        for row in output.split('\n'):
            if row != '':
                iface_list_temp.append(row)
                self.src_flows[row] = [0] * len(self.host_ips)
                self.dst_flows[row] = [0] * len(self.host_ips)
        # return_list = [iface for iface in iface_list_temp if iface in i_h_map]  # filter against actual hosts
        self.iface_list = iface_list_temp

    def _get_flow_stats(self, iface_list):
        processes = []
        for iface in iface_list:
            cmd = ("sudo timeout 1 tcpdump -l -i " + iface + " -n -c 5 ip 2>/dev/null | " +
                   "grep -P -o \'([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+).*? > ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\' | " +
                   "grep -P -o \'[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\' | xargs -n 2 echo | awk \'!a[$0]++\'")
            # output = subprocess.check_output(cmd, shell=True)
            p = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
            processes.append((p, iface))

        for p, iface in processes:
            p.wait()
            output, err = p.communicate()
            self.src_flows[iface] = [0] * len(self.host_ips)
            self.dst_flows[iface] = [0] * len(self.host_ips)
            for row in output.split('\n'):
                if row != '':
                    src, dst = row.split(' ')
                    for i, ip in enumerate(self.host_ips):
                        if src == ip:
                            self.src_flows[iface][i] = 1
                        if dst == ip:
                            self.dst_flows[iface][i] = 1
            # print(self.src_flows)
            # print(self.dst_flows)

    def _collect_flows(self):
        # iface_list = self._get_interfaces()
        self._get_flow_stats(self.iface_list)
        # self.free_bandwidths = self._get_free_bandwidths(self.bandwidths)
        # self.drops, self.overlimits, self.queues = self._get_qdisc_stats(self.iface_list)

    def get_interface_flows(self):
        # self._get_flow_stats(self.iface_list)
        return self.src_flows, self.dst_flows
