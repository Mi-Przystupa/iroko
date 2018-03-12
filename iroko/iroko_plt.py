from monitor.helper import *
from math import fsum
import numpy as np
import glob


def get_bisection_bw(input_file, pat_iface):
    pat_iface = re.compile(pat_iface)

    data = read_list(input_file)

    rate = {}
    column = 2
    for row in data:
        try:
            ifname = row[1]
        except:
            break
        if ifname not in ['eth0', 'lo']:
            if not rate.has_key(ifname):
                rate[ifname] = []
            try:
                rate[ifname].append(float(row[column]) * 8.0 / (1 << 20))
            except:
                break
    vals = []
    for k in rate.keys():
        if pat_iface.match(k):
            avg_rate = avg(rate[k][10:-10])
            vals.append(avg_rate)
    return fsum(vals)


def prune(out_dir, t_file, switch):
    print("Iroko:", t_file)
    input_file = out_dir + '/rate.txt'
    output_file = out_dir + '/rate_final.txt'
    vals = get_bisection_bw(input_file, switch)
    print(vals)
    file = open(output_file, 'w')
    file.write("avg:%f" % vals)
    file.close()
    os.remove(input_file)


def get_result_dict(file):
    data = []
    for line in open(file):
        data.append(tuple(line.strip().split(':')))
    data = dict(data)
    return data


def plot_test_results(input_dir, plt_name, traffic_files, labels, algorithms):

    fbb = 16. * 10  # 160 mbps
    num_plot = 2
    num_t = 20
    n_t = num_t / num_plot
    bb = {}
    for algo, conf in algorithms.iteritems():
        bb[algo] = []
        for tf in traffic_files:
            print("algo:", tf)
            input_file = input_dir + '/%s/%s/rate_final.txt' % (conf['pre'], tf)
            results = get_result_dict(input_file)
            avg = float(results['avg'])
            print(avg)
            bb[algo].append(avg / fbb / 2)

    ind = np.arange(n_t)
    width = 0.15
    fig = plt.figure(1)
    fig.set_size_inches(8.5, 6.5)

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
            print (conf['color'])
            p = plt.bar(ind + (index + 1.5) * width, bb[algo][i * n_t:(i + 1) * n_t], width=width,
                        color=conf['color'])
            p_bar.append(p[0])
            p_legend.append(algo)
            index += 1
        plt.legend(p_bar, p_legend, loc='upper left')
        plt.savefig(plt_name)


def plot_train_results(input_dir, plt_name, traffic_files, algorithms):
    plt_dir = os.path.dirname(plt_name)
    if not os.path.exists(plt_dir):
        if not plt_dir == '':
            os.makedirs(plt_dir)
    algo = 'iroko'
    conf = algorithms['iroko']
    fbb = 16. * 10  # 160 mbps
    folders = glob.glob('%s/%s_*' % (input_dir, conf['pre']))
    epochs = 2 #len(folders)
    bb = {}
    for tf in traffic_files:
        for e in range(epochs):
            bb['%s_%s' % (algo, e)] = []
            print("%s: %s" % (algo, tf))
            input_file = input_dir + '/%s_%d/%s/rate_final.txt' % (conf['pre'], e, tf)
            results = get_result_dict(input_file)
            avg = float(results['avg'])
            print(avg)
            bb['%s_%s' % (algo, e)].append(avg / fbb / 2)

        p_bar = []
        p_legend = []
        for i in range(epochs):
            # FatTree + Iroko
            p_bar.append(bb['iroko_%d' % i])
            p_legend.append('Epoch %i' % i)
        plt.plot(p_bar)
        x_val = list(range(epochs))
        plt.xticks(np.arange(min(x_val), max(x_val) + 1, 1.0))
        plt.xlabel('Epoch')
        plt.ylabel('Normalized Average Bisection Bandwidth')
        plt.savefig("%s_%s" % (plt_name, tf))
        plt.gcf().clear()
