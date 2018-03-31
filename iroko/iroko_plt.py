from monitor.helper import *
from math import fsum
import numpy as np
import itertools


def get_bw_stats(input_file, pat_iface):
    pat_iface = re.compile(pat_iface)

    data = read_list(input_file)

    rate = {}
    column = 2
    for row in data:
        try:
            ifname = row[1]
        except Exception as e:
            break
        if ifname not in ['eth0', 'lo']:
            if ifname not in rate:
                rate[ifname] = []
            try:
                rate[ifname].append(float(row[column]) * 8.0 / (1 << 20))
            except Exception as e:
                break
    vals = {}
    avg_bw = []
    full_bw = []
    for k in rate.keys():
        if pat_iface.match(k):
            t_rate = rate[k][10:-10]
            avg_bw.append(avg(t_rate))
            full_bw.append(t_rate)
    full_bw = list(itertools.chain.from_iterable(full_bw))
    vals["avg_bw"] = fsum(avg_bw)
    vals["median_bw"] = np.median(full_bw)
    vals["max_bw"] = max(full_bw)
    vals["stdev_bw"] = stdev(full_bw)
    return vals


def prune_bw(out_dir, t_file, switch):
    print("Pruning: %s:" % out_dir)
    input_file = out_dir + '/rate.txt'
    output_file = out_dir + '/rate_final.txt'
    vals = get_bw_stats(input_file, switch)
    print(vals)
    file = open(output_file, 'w')
    for key, val in vals.iteritems():
        file.write("%s:%f\n" % (key, val))
    file.close()
    os.remove(input_file)


def get_bw_dict(file):
    data = []
    for line in open(file):
        data.append(tuple(line.strip().split(':')))
    data = dict(data)
    return data


def get_qlen_stats(input_file, pat_iface):
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
    for k in qlen.keys():
        if pat_iface.match(k):
            qlens.append(qlen[k])
    # qlens = map(float, col(2, data))[10:-10]
    qlens = list(itertools.chain.from_iterable(qlens))
    vals["avg_qlen"] = avg(qlens)
    vals["median_qlen"] = np.median(qlens)
    vals["max_qlen"] = max(qlens)
    vals["stdev_qlen"] = stdev(qlens)
    vals["xcdf_qlen"], vals["ycdf_qlen"] = cdf(qlens)
    return vals


def plot_queue(files, legends, out):
    to_plot = []
    for i, f in enumerate(files):
        data = read_list(f)
        xaxis = map(float, col(1, data))
        start_time = xaxis[0]
        xaxis = map(lambda x: x - start_time, xaxis)
        qlens = map(float, col(2, data))
        to_plot.append(qlens[10:-10])

    plt.grid(True)

    for i, data in enumerate(to_plot):
        xs, ys = cdf(map(int, data))
        plt.plot(xs, ys, label=legends[i], lw=2, **get_style(i))


def plot_test_bw(input_dir, plt_name, traffic_files, labels, algorithms):

    fbb = 16. * 10  # 160 mbps
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
            input_file = input_dir + '/%s/%s/rate_final.txt' % (conf['pre'], tf)
            results = get_bw_dict(input_file)
            avg = float(results['avg_bw'])
            print(avg)
            bb[algo].append(avg / fbb / 2)
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


def plot_test_qlen(input_dir, plt_name, traffic_files, labels, algorithms, iface):
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
            results = get_qlen_stats(input_file, conf['sw'])
            plt.plot(results['xcdf_qlen'], results['ycdf_qlen'], label=labels[i], color=conf['color'], lw=2)
        plt.legend(bbox_to_anchor=(1.5, 1.22), loc='upper right', fontsize='x-large')
        plt.savefig(plt_name)
    plt.gcf().clear()


def plot_train_bw(input_dir, plt_name, traffic_files, algorithm, epochs):
    plt_dir = os.path.dirname(plt_name)
    if not os.path.exists(plt_dir):
        if not plt_dir == '':
            os.makedirs(plt_dir)
    algo = algorithm[0]
    conf = algorithm[1]
    fbb = 16. * 10  # 160 mbps
    # folders = glob.glob('%s/%s_*' % (input_dir, conf['pre']))
    bb = {}
    for tf in traffic_files:
        for e in range(epochs):
            bb['%s_%s' % (algo, e)] = []
            print("%s: %s" % (algo, tf))
            input_file = input_dir + '/%s_%d/%s/rate_final.txt' % (conf['pre'], e, tf)
            results = get_bw_dict(input_file)
            avg_bw = float(results['avg_bw'])
            print(avg_bw)
            bb['%s_%s' % (algo, e)].append(avg_bw / fbb / 2)

        p_bar = []
        p_legend = []
        for i in range(epochs):
            p_bar.append(bb['%s_%d' % (algo, i)][0])
            p_legend.append('Epoch %i' % i)
        print("Total Average Bandwidth: %f" % avg(p_bar))
        plt.plot(p_bar)
        x_val = list(range(epochs + 1))
        if epochs > 100:
            x_step = x_val[0::(epochs / 10)]
        plt.xticks(x_step)
        plt.xlabel('Epoch')
        plt.ylabel('Normalized Average Bisection Bandwidth')
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.savefig("%s_%s" % (plt_name, tf))
        plt.gcf().clear()


def plot_train_qlen(input_dir, plt_name, traffic_files, algorithm, epochs):
    plt_dir = os.path.dirname(plt_name)
    if not os.path.exists(plt_dir):
        if not plt_dir == '':
            os.makedirs(plt_dir)
    algo = algorithm[0]
    conf = algorithm[1]
    # folders = glob.glob('%s/%s_*' % (input_dir, conf['pre']))
    bb = {}
    for tf in traffic_files:
        for e in range(epochs):
            bb['%s_%s' % (algo, e)] = []
            print("%s: %s" % (algo, tf))
            input_file = input_dir + '/%s_%d/%s/qlen.txt' % (conf['pre'], e, tf)
            results = get_qlen_stats(input_file, conf['sw'])
            avg_qlen = float(results['avg_qlen'])
            print(avg_qlen)
            bb['%s_%s' % (algo, e)].append(avg_qlen)

        p_bar = []
        p_legend = []
        for i in range(epochs):
            p_bar.append(bb['%s_%d' % (algo, i)][0])
            p_legend.append('Epoch %i' % i)
        print("Total Average Qlen: %f" % avg(p_bar))
        plt.plot(p_bar)
        x_val = list(range(epochs + 1))
        if epochs > 100:
            x_step = x_val[0::(epochs / 10)]
        plt.xticks(x_step)
        plt.xlabel('Epoch')
        plt.ylabel('Average Queue Length')
        axes = plt.gca()
        axes.set_ylim([0, 50])
        plt.savefig("%s_%s" % (plt_name, tf))
        plt.gcf().clear()
