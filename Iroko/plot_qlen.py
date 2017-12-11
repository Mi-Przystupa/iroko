'''
@author: Milad Sharif(msharif@stanford.edu)
'''

from monitor.helper import *
from math import fsum
import numpy as np
from os import system

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-f', dest='files', required=True, help='Input rates')

args = parser.parse_args()


''' Output of bwm-ng has the following format:
    unix_timestamp;iface_name;bytes_out;bytes_in;bytes_total;packets_out;packets_in;packets_total;errors_out;errors_in
    '''

traffics = ['stag_prob_0_2_3_data', 'stag_prob_1_2_3_data', 'stag_prob_2_2_3_data',
            'stag_prob_0_5_3_data', 'stag_prob_1_5_3_data', 'stag_prob_2_5_3_data', 'stride1_data',
            'stride2_data', 'stride4_data', 'stride8_data', 'random0_data', 'random1_data', 'random2_data',
            'random0_bij_data', 'random1_bij_data', 'random2_bij_data', 'random_2_flows_data',
            'random_3_flows_data', 'random_4_flows_data', 'hotspot_one_to_one_data']

labels = ['stag0(0.2,0.3)', 'stag1(0.2,0.3)', 'stag2(0.2,0.3)', 'stag0(0.5,0.3)',
          'stag1(0.5,0.3)', 'stag2(0.5,0.3)', 'stride1', 'stride2',
          'stride4', 'stride8', 'rand0', 'rand1', 'rand2', 'randbij0',
          'randbij1', 'randbij2', 'randx2', 'randx3', 'randx4', 'hotspot']

'''
traffics=['stag_prob_0_2_3_data']

labels=['stag0(0.2,0.3)']
'''


def get_style(i):
    if i == 0:
        return {'color': 'brown'}
    elif i == 1:
        return {'color': 'red'}
    elif i == 2:
        return {'color': 'magenta'}
    elif i == 3:
        return {'color': 'green'}
    elif i == 4:
        return {'color': 'royalblue'}
    else:
        return {'color': 'black', 'ls': '-.'}


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

    plt.legend(loc="best")


def plot_results(args):

    for i,t in enumerate(traffics):
        nb_input = args.files + '/nonblocking/%s/qlen.txt' % t
        ecmp_input = args.files + '/fattree-ecmp/%s/qlen.txt' % t
        dctcp_input = args.files + '/fattree-dctcp/%s/qlen.txt' % t
        iroko_input = args.files + '/fattree-iroko/%s/qlen.txt' % t
        hedera_input = args.files + '/fattree-hedera/%s/qlen.txt' % t

        fig = plt.figure(1)
        fig.set_size_inches(24, 12)
        ax = fig.add_subplot(2, len(traffics)/2, i + 1)
        ax.yaxis.grid()
        plt.ylim((0.5, 1.0))
        plt.ylabel("Fraction")
        plot_queue([dctcp_input, ecmp_input, iroko_input, hedera_input, nb_input],
                ["dctcp", "ecmp", "iroko", "hedera", "nonblocking"], t)

    plt.savefig("qlen.png")

plot_results(args)
