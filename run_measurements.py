from monitor.helper import *
import iroko_plt
# parser = argparse.ArgumentParser()
# parser.add_argument('--input', '-f', dest='files', required=True, help='Input rates')
# parser.add_argument('--out', '-o', dest='out', default=None,
#                     help="Output png file for the plot.")
# parser.add_argument('--epoch', '-d', dest='epoch', default=False,
#                     action='store_true', help='Generate the graphs in epoch mode to show the delta.')
# args = parser.parse_args()


''' Output of bwm-ng has the following format:
    unix_timestamp;iface_name;bytes_out;bytes_in;bytes_total;packets_out;packets_in;packets_total;errors_out;errors_in
    '''

traffic_files = ['stag_prob_0_2_3_data', 'stag_prob_1_2_3_data', 'stag_prob_2_2_3_data',
                 'stag_prob_0_5_3_data', 'stag_prob_1_5_3_data', 'stag_prob_2_5_3_data', 'stride1_data',
                 'stride2_data', 'stride4_data', 'stride8_data', 'random0_data', 'random1_data', 'random2_data',
                 'random0_bij_data', 'random1_bij_data', 'random2_bij_data', 'random_2_flows_data',
                 'random_3_flows_data', 'random_4_flows_data', 'hotspot_one_to_one_data']

traffic_files = ['stag_prob_0_2_3_data']

INPUT_DIR = 'inputs'
OUTPUT_DIR = 'results'
DURATION = 60
EPOCHS = 10
FATTREE_SW = '300[1-9]'


def train():
    os.system('sudo mn -c')
    for e in range(EPOCHS):
        for tf in traffic_files:
            input_file = '%s/iroko/%s' % (INPUT_DIR, tf)
            pref = "fattree-iroko_%d" % e
            out_dir = '%s/%s/%s' % (OUTPUT_DIR, pref, tf)
            os.system('sudo python iroko.py -i %s -d %s -p 0.03 -t %d --iroko' % (input_file, out_dir, DURATION))
            os.system('sudo chown -R $USER:$USER %s' % out_dir)
            iroko_plt.prune(out_dir, e, tf, FATTREE_SW)


if __name__ == '__main__':
    train()
    iroko_plt.plot_epoch_results('results', 'plots/training', traffic_files)
