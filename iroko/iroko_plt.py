from math import fsum
import itertools
import json
import re
import os
import itertools  # noqa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np
from monitor.helper import stdev
from monitor.helper import avg
from monitor.helper import read_list
from monitor.helper import cdf


class IrokoPlotter():

    def __init__(self, num_ifaces, epochs):
        self.name = 'IrokoPlotter'
        self.max_bw = 10                 # Max capacity of link normalized to mbit
        self.max_queue = 50                # Max queue per link
        self.num_ifaces = num_ifaces       # Num of monitored interfaces
        self.epochs = epochs

    def get_bw_stats(self, input_file, pat_iface):
        pat_iface = re.compile(pat_iface)
        data = read_list(input_file)

        rate = {}
        column = 2
        for row in data:
            try:
                ifname = row[1]
            except Exception as e:
                break
            if pat_iface.match(ifname):
                if ifname not in rate:
                    rate[ifname] = []
                try:
                    rate[ifname].append(float(row[column]) * 8.0 / (1 << 20))
                except Exception as e:
                    break
        vals = {}
        avg_bw = []
        match = 0
        avg_bw_iface = {}
        for iface, bws in rate.items():
            rate[iface] = bws[10:-10]
            avg_bw.append(avg(bws[10:-10]))
            avg_bw_iface[iface] = avg(bws[10:-10])
            match += 1
        # Update the num of interfaces to reflect true matches
        # TODO: This is a hack. Remove.
        self.num_ifaces = match
        total_bw = list(itertools.chain.from_iterable(rate.values()))
        vals["avg_bw_iface"] = avg_bw_iface
        vals["avg_bw"] = fsum(avg_bw)
        vals["median_bw"] = np.median(total_bw)
        vals["max_bw"] = max(total_bw)
        vals["stdev_bw"] = stdev(total_bw)

        return vals, rate

    def prune_bw(self, out_dir, t_file, switch):
        print("Pruning: %s:" % out_dir)
        input_file = out_dir + '/rate.txt'
        summary_file = out_dir + '/rate_summary.json'
        pruned_file = out_dir + '/rate_filtered.json'
        vals, full_bw = self.get_bw_stats(input_file, switch)

        with open(pruned_file, 'w') as fp:
            json.dump(full_bw, fp)
            fp.close()
        with open(summary_file, 'w') as fp:
            json.dump(vals, fp)
            fp.close()
        os.remove(input_file)

    def get_bw_dict(self, file):
        with open(file, 'r') as fp:
            data = json.load(fp)
            fp.close()
            return data

    def get_qlen_stats(self, input_file, pat_iface):
        data = read_list(input_file)
        pat_iface = re.compile(pat_iface)
        qlen = {}
        for row in data:
            try:
                ifname = row[0]
            except Exception as e:
                break
            if ifname not in ['eth0', 'lo']:
                if ifname not in qlen:
                    qlen[ifname] = []
                try:
                    qlen[ifname].append(int(row[2]))
                except Exception as e:
                    break
        vals = {}
        qlens = []
        avg_qlen_iface = {}
        for k in qlen.keys():
            if pat_iface.match(k):
                qlens.append(qlen[k])
                avg_qlen_iface[k] = avg(qlen[k])
        # qlens = map(float, col(2, data))[10:-10]
        qlens = list(itertools.chain.from_iterable(qlens))
        vals["avg_qlen_iface"] = avg_qlen_iface
        vals["avg_qlen"] = avg(qlens)
        vals["median_qlen"] = np.median(qlens)
        vals["max_qlen"] = max(qlens)
        vals["stdev_qlen"] = stdev(qlens)
        vals["xcdf_qlen"], vals["ycdf_qlen"] = cdf(qlens)
        return vals

    def plot_test_bw(self, input_dir, plt_name, traffic_files, labels, algorithms):

        fbb = self.num_ifaces * self.max_bw
        num_plot = 2
        num_t = len(traffic_files)
        n_t = num_t / num_plot
        ind = np.arange(n_t)
        width = 0.15
        fig = plt.figure(1)
        fig.set_size_inches(8.5, 6.5)

        bb = {}
        for algo, conf in algorithms.iteritems():
            bb[algo] = []
            for tf in traffic_files:
                print("algo:", tf)
                input_file = input_dir + '/%s/%s/rate_summary.json' % (conf['pre'], tf)
                results = self.get_bw_dict(input_file)
                avg_bw = float(results['avg_bw'])
                print(avg_bw)
                bb[algo].append(avg_bw / fbb / 2)
        for i in range(num_plot):
            fig.set_size_inches(24, 12)
            ax = fig.add_subplot(2, 1, i + 1)
            ax.yaxis.grid()
            plt.ylim(0.0, 1.0)
            plt.xlim(0, 10)
            plt.ylabel('Normalized Average Bisection Bandwidth')
            plt.xticks(ind + 2.5 * width, labels[i * n_t:(i + 1) * n_t])
            p_bar = []
            p_legend = []
            index = 0
            for algo, conf in algorithms.iteritems():
                p = plt.bar(ind + (index + 1.5) * width, bb[algo][i * n_t:(i + 1) * n_t], width=width,
                            color=conf['color'])
                p_bar.append(p[0])
                p_legend.append(algo)
                index += 1
            plt.legend(p_bar, p_legend, loc='upper left')
            plt.savefig(plt_name)
        plt.grid(True)
        plt.gcf().clear()

    def plot_test_qlen(self, input_dir, plt_name, traffic_files, labels, algorithms, iface):
        fig = plt.figure(1)
        fig.set_size_inches(8.5, 6.5)
        for i, tf in enumerate(traffic_files):
            plt.grid(True)
            fig = plt.figure(1)
            fig.set_size_inches(40, 12)
            ax = fig.add_subplot(2, len(traffic_files) / 2, i + 1)
            ax.yaxis.grid()
            plt.ylim((0.8, 1.0))
            plt.ylabel("Fraction", fontsize=16)
            plt.xlabel(labels[i], fontsize=18)
            for algo, conf in algorithms.iteritems():
                print("%s:%s" % (algo, tf))
                input_file = input_dir + '/%s/%s/qlen.txt' % (conf['pre'], tf)
                results = self.get_qlen_stats(input_file, conf['sw'])
                plt.plot(results['xcdf_qlen'], results['ycdf_qlen'], label=labels[i], color=conf['color'], lw=2)
            plt.legend(bbox_to_anchor=(1.5, 1.22), loc='upper right', fontsize='x-large')
            plt.savefig(plt_name)
        plt.gcf().clear()

    def plot_train_bw(self, input_dir, plt_name, traffic_files, algorithm):
        plt_dir = os.path.dirname(plt_name)
        if not os.path.exists(plt_dir):
            if not plt_dir == '':
                os.makedirs(plt_dir)
        algo = algorithm[0]
        conf = algorithm[1]
        fbb = self.num_ifaces * self.max_bw
        # folders = glob.glob('%s/%s_*' % (input_dir, conf['pre']))
        bb = {}
        for tf in traffic_files:
            for e in range(self.epochs):
                bb['%s_%s' % (algo, e)] = []
                print("%s: %s" % (algo, tf))
                input_file = input_dir + '/%s_%d/%s/rate_summary.json' % (conf['pre'], e, tf)
                results = self.get_bw_dict(input_file)
                avg_bw = float(results['avg_bw'])
                print(avg_bw)
                bb['%s_%s' % (algo, e)].append(avg_bw / fbb / 2)

            p_bar = []
            p_legend = []
            for i in range(self.epochs):
                p_bar.append(bb['%s_%d' % (algo, i)][0])
                p_legend.append('Epoch %i' % i)
            print("Total Average Bandwidth: %f" % avg(p_bar))
            plt.plot(p_bar)
            x_val = list(range(self.epochs + 1))
            if self.epochs > 100:
                x_step = x_val[0::(self.epochs / 10)]
                plt.xticks(x_step)
            plt.xlabel('Epoch')
            plt.ylabel('Normalized Average Bisection Bandwidth')
            axes = plt.gca()
            axes.set_ylim([0, 1])
            plt.savefig("%s_%s" % (plt_name, tf))
            plt.gcf().clear()

    def plot_train_bw_alt(self, input_dir, plt_name, traffic_files, algorithm):
        plt_dir = os.path.dirname(plt_name)
        if not os.path.exists(plt_dir):
            if not plt_dir == '':
                os.makedirs(plt_dir)
        algo = algorithm[0]
        conf = algorithm[1]
        fbb = self.max_bw
        # folders = glob.glob('%s/%s_*' % (input_dir, conf['pre']))
        bb = {}
        for tf in traffic_files:
            for epoch in range(self.epochs):
                bb['%s_%s' % (algo, epoch)] = {}
                input_file = input_dir + '/%s_%d/%s/rate_summary.json' % (conf['pre'], epoch, tf)
                results = self.get_bw_dict(input_file)
                avg_bw = results['avg_bw_iface']
                print(avg_bw)
                for iface, bw in avg_bw.iteritems():
                    bb['%s_%s' % (algo, epoch)].setdefault(iface, []).append(float(bw) / fbb)
            for iface, bw in bb['%s_%s' % (algo, 0)].iteritems():
                p_bar = []
                p_legend = []
                for i in range(self.epochs):
                    p_bar.append(bb['%s_%d' % (algo, i)][iface][0])
                    p_legend.append('Epoch %i' % i)
                print("Total Average Bandwidth: %f" % avg(p_bar))
                plt.plot(p_bar, label=iface)
            x_val = list(range(self.epochs + 1))
            if self.epochs > 100:
                x_step = x_val[0::(self.epochs / 10)]
                plt.xticks(x_step)
            plt.xlabel('Epoch')
            plt.ylabel('Normalized Average Bisection Bandwidth')
            axes = plt.gca()
            axes.set_ylim([0, 1])
            plt.legend(loc='upper right')
            plt.savefig("%s_%s" % (plt_name, tf))
            plt.gcf().clear()

    def plot_train_qlen(self, input_dir, plt_name, traffic_files, algorithm):
        plt_dir = os.path.dirname(plt_name)
        if not os.path.exists(plt_dir):
            if not plt_dir == '':
                os.makedirs(plt_dir)
        algo = algorithm[0]
        conf = algorithm[1]
        # folders = glob.glob('%s/%s_*' % (input_dir, conf['pre']))
        bb = {}
        for tf in traffic_files:
            for e in range(self.epochs):
                bb['%s_%s' % (algo, e)] = []
                print("%s: %s" % (algo, tf))
                input_file = input_dir + '/%s_%d/%s/qlen.txt' % (conf['pre'], e, tf)
                results = self.get_qlen_stats(input_file, conf['sw'])
                avg_qlen = float(results['avg_qlen'])
                print(avg_qlen)
                bb['%s_%s' % (algo, e)].append(avg_qlen)

            p_bar = []
            p_legend = []
            for i in range(self.epochs):
                p_bar.append(bb['%s_%d' % (algo, i)][0])
                p_legend.append('Epoch %i' % i)
            print("Total Average Qlen: %f" % avg(p_bar))
            plt.plot(p_bar)
            x_val = list(range(self.epochs + 1))
            if self.epochs > 100:
                x_step = x_val[0::(self.epochs / 10)]
                plt.xticks(x_step)
            plt.xlabel('Epoch')
            plt.ylabel('Average Queue Length')
            axes = plt.gca()
            axes.set_ylim([0, self.max_queue])
            plt.savefig("%s_%s" % (plt_name, tf))
            plt.gcf().clear()

    def plot_train_qlen_alt(self, input_dir, plt_name, traffic_files, algorithm):
        plt_dir = os.path.dirname(plt_name)
        if not os.path.exists(plt_dir):
            if not plt_dir == '':
                os.makedirs(plt_dir)
        algo = algorithm[0]
        conf = algorithm[1]
        # folders = glob.glob('%s/%s_*' % (input_dir, conf['pre']))
        bb = {}
        for tf in traffic_files:
            for epoch in range(self.epochs):
                bb['%s_%s' % (algo, epoch)] = {}
                print("%s: %s" % (algo, tf))
                input_file = input_dir + '/%s_%d/%s/qlen.txt' % (conf['pre'], epoch, tf)
                results = self.get_qlen_stats(input_file, conf['sw'])
                avg_qlen = results['avg_qlen_iface']
                print(avg_qlen)
                for iface, qlen in avg_qlen.iteritems():
                    bb['%s_%s' % (algo, epoch)].setdefault(iface, []).append(float(qlen))

            for iface, bw in bb['%s_%s' % (algo, 0)].iteritems():
                p_bar = []
                p_legend = []
                for i in range(self.epochs):
                    p_bar.append(bb['%s_%d' % (algo, i)][iface][0])
                    p_legend.append('Epoch %i' % i)
                print("Total Average Qlen: %f" % avg(p_bar))
                plt.plot(p_bar, label=iface)
            x_val = list(range(self.epochs + 1))
            if self.epochs > 100:
                x_step = x_val[0::(self.epochs / 10)]
                plt.xticks(x_step)
            plt.xlabel('Epoch')
            plt.ylabel('Average Queue Length')
            axes = plt.gca()
            # axes.set_ylim([0, self.max_queue])
            plt.legend(loc='upper right')
            plt.savefig("%s_%s" % (plt_name, tf))
            plt.gcf().clear()

    def plot_reward(self, fname, pltname):
        float_rewards = []
        with open(fname) as f:
            str_rewards = f.readlines()
        str_rewards = [x.strip() for x in str_rewards]
        for reward in str_rewards:
            float_rewards.append(float(reward))
        plt.plot(float_rewards)
        plt.ylabel('Reward')
        plt.savefig(pltname)
        plt.gcf().clear()

    def moving_average(self, input_reward, n=100):
        ret = np.cumsum(input_reward, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def evolving_average(self, input_reward, n=100):
        avgreward = []
        summation = avg(input_reward[:n - 1])
        i = n
        for r in input_reward[n:]:
            summation += float(r)
            avgreward.append(summation / float(i))
            i += 1
        return avgreward

    def plot_avgreward(self, fname, pltname):
        with open(fname) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [float(x.strip()) for x in content]
        window = 100
        reward_mean = self.moving_average(content, window)
        reward_evolve = self.evolving_average(content, window)

        plt.subplot(2, 1, 1)
        plt.plot(reward_mean, label="Mean %d" % window)
        plt.legend(loc='lower left')
        plt.ylabel('Reward')
        plt.subplot(2, 1, 2)
        plt.plot(reward_evolve, label="Evolution")
        plt.ylabel('Reward')
        plt.xlabel('Iterations')
        plt.legend(loc='lower right')
        plt.savefig(pltname)
        plt.gcf().clear()
