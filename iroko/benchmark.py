from monitor.helper import *
import iroko_plt
parser = argparse.ArgumentParser()
# parser.add_argument('--input', '-f', dest='files', required=True, help='Input rates')

parser.add_argument('--train', '-tr', dest='train', default=False,
                    action='store_true', help='Train Iroko in epoch mode and measure the improvement.')
parser.add_argument('--epoch', '-e', dest='epoch', type=int, default=0,
                    help='Specify the number of epochs Iroko should be trained.')
parser.add_argument('--offset', '-o', dest='offset', type=int, default=0,
                    help='Intended to start epochs from an offset.')
parser.add_argument('--test', '-t', dest='test', default=False,
                    action='store_true', help='Run the full tests of the algorithm.')

args = parser.parse_args()


''' Output of bwm-ng has the following format:
    unix_timestamp;iface_name;bytes_out;bytes_in;bytes_total;packets_out;packets_in;packets_total;errors_out;errors_in
    '''

traffic_files = ['stag_prob_0_2_3_data', 'stag_prob_1_2_3_data', 'stag_prob_2_2_3_data',
                 'stag_prob_0_5_3_data', 'stag_prob_1_5_3_data', 'stag_prob_2_5_3_data', 'stride1_data',
                 'stride2_data', 'stride4_data', 'stride8_data', 'random0_data', 'random1_data', 'random2_data',
                 'random0_bij_data', 'random1_bij_data', 'random2_bij_data', 'random_2_flows_data',
                 'random_3_flows_data', 'random_4_flows_data', 'hotspot_one_to_one_data']
# traffic_files = ['stag_prob_0_2_3_data']

labels = ['stag0(0.2,0.3)', 'stag1(0.2,0.3)', 'stag2(0.2,0.3)', 'stag0(0.5,0.3)',
          'stag1(0.5,0.3)', 'stag2(0.5,0.3)', 'stride1', 'stride2',
          'stride4', 'stride8', 'rand0', 'rand1', 'rand2', 'randbij0',
          'randbij1', 'randbij2', 'randx2', 'randx3', 'randx4', 'hotspot']
# labels = ['stag0(0.2,0.3)']

qlen_traffics = ['stag_prob_2_2_3_data',
                 'stag_prob_2_5_3_data', 'stride1_data',
                 'stride2_data', 'random0_data',
                 'random_2_flows_data',
                 'random_3_flows_data', 'random_4_flows_data']

qlen_labels = ['stag2(0.2,0.3)',
               'stag2(0.5,0.3)', 'stride1', 'stride2',
               'rand0',
               'randx2', 'randx3', 'randx4']


# labels = ['stag0(0.2,0.3)']

INPUT_DIR = 'inputs'
OUTPUT_DIR = 'results'
DURATION = 60
NONBLOCK_SW = '1001'
HEDERA_SW = '[0-3]h[0-1]h1'
FATTREE_SW = '300[1-9]'


def train(input_dir, output_dir, duration, offset, epochs, conf):
    os.system('sudo mn -c')
    for e in range(offset, epochs + offset):
        for tf in traffic_files:
            input_file = '%s/%s/%s' % (input_dir, conf['tf'], tf)
            pre_folder = "%s_%d" % (conf['pre'], e)
            out_dir = '%s/%s/%s' % (output_dir, pre_folder, tf)
            os.system('sudo python iroko.py -i %s -d %s -p 0.03 -t %d --iroko' % (input_file, out_dir, duration))
            os.system('sudo chown -R $USER:$USER %s' % out_dir)
            iroko_plt.prune_bw(out_dir, tf, conf['sw'])


def get_test_config():
    algos = {}
    algos['nonblocking'] = {'sw': NONBLOCK_SW, 'tf': 'default', 'pre': 'nonblocking', 'color': 'royalblue'}
    algos['iroko'] = {'sw': FATTREE_SW, 'tf': 'default', 'pre': 'fattree-iroko', 'color': 'green'}
    algos['ecmp'] = {'sw': FATTREE_SW, 'tf': 'default', 'pre': 'fattree-ecmp', 'color': 'magenta'}
    algos['dctcp'] = {'sw': FATTREE_SW, 'tf': 'default', 'pre': 'fattree-dctcp', 'color': 'brown'}
    algos['hedera'] = {'sw': HEDERA_SW, 'tf': 'hedera', 'pre': 'fattree-hedera', 'color': 'red'}
    return algos


def run_tests(input_dir, output_dir, duration, traffic_files, algorithms):
    os.system('sudo mn -c')
    for algo, conf in algorithms.iteritems():
        pre_folder = conf['pre']
        for tf in traffic_files:
            input_file = '%s/%s/%s' % (input_dir, conf['tf'], tf)
            out_dir = '%s/%s/%s' % (output_dir, pre_folder, tf)
            os.system('sudo python iroko.py -i %s -d %s -p 0.03 -t %d --%s' % (input_file, out_dir, duration, algo))
            os.system('sudo chown -R $USER:$USER %s' % out_dir)
            iroko_plt.prune_bw(out_dir, tf, conf['sw'])


if __name__ == '__main__':
    algorithms = get_test_config()
    if args.train:
        if args.epoch is 0:
            print("Please specify the number of epochs you would like to train with (--epoch)!")
            exit(1)
        print("Training the Iroko agent for %d epoch(s)." % args.epoch)
        #train(INPUT_DIR, OUTPUT_DIR, DURATION, args.offset, args.epoch, algorithms['iroko'])
        iroko_plt.plot_train_bw('results', 'plots/train_bw', traffic_files, algorithms, args.epoch + args.offset)
        iroko_plt.plot_train_qlen('results', 'plots/train_qlen', traffic_files, algorithms, args.epoch + args.offset)
    if args.test:
        print("Running benchmarks for %d seconds each with input matrix at %s and output at %s"
              % (DURATION, INPUT_DIR, OUTPUT_DIR))
        run_tests(INPUT_DIR, OUTPUT_DIR, DURATION, traffic_files, algorithms)
        iroko_plt.plot_test_bw('results', 'plots/test_bw_sum', traffic_files, labels, algorithms)
        iroko_plt.plot_test_qlen('results', 'plots/test_qlen_sum', qlen_traffics, qlen_labels, algorithms)

    elif not args.train:
        print("Doing nothing...\nRun the command with --train to train the Iroko agent and/or --test to run benchmarks.")
