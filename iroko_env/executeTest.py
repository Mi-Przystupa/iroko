import argparse
import os  # noqa
import sre_yield
from iroko_plt import IrokoPlotter
from iroko_env import Iroko_Environment
from DataCenterEnv import DataCenterEnv

from tensorforce import TensorForceError
from tensorforce.agents import Agent, PPOAgent, DDPGAgent
from tensorforce.execution import Runner
import numpy as np


PARSER = argparse.ArgumentParser()
# parser.add_argument('--input', '-f', dest='files', required=True, help='Input rates')

PARSER.add_argument('--epoch', '-e', dest='epochs', type=int, default=0,
                    help='Specify the number of epochs Iroko should be trained.')
PARSER.add_argument('--offset', '-o', dest='offset', type=int, default=0,
                    help='Intended to start epochs from an offset.')

PARSER.add_argument('--plot', '-pl', dest='plot', action='store_true',
                    default='False', help='Only plot the results for training.')
PARSER.add_argument('--dumbbell', '-db', dest='dumbbell', action='store_true',
                    default='False', help='Train on a simple dumbbell topology')
PARSER.add_argument('--load', default=False, action='store_true', help='Load agent')
PARSER.add_argument('--save-dir',dest='save_dir', default='./model/', help='Model save dir')

PARSER.add_argument('--asEnv', '-env', dest='env', default=False,
                    action='store_true', help='Flag to use RL environment version')

ARGS = PARSER.parse_args()


''' Output of bwm-ng has the following format:
    unix_timestamp;iface_name;bytes_out;bytes_in;bytes_total;packets_out;packets_in;packets_total;errors_out;errors_in
    '''

TRAFFIC_FILES = ['stag_prob_0_2_3_data', 'stag_prob_1_2_3_data', 'stag_prob_2_2_3_data',
                 'stag_prob_0_5_3_data', 'stag_prob_1_5_3_data', 'stag_prob_2_5_3_data', 'stride1_data',
                 'stride2_data', 'stride4_data', 'stride8_data', 'random0_data', 'random1_data', 'random2_data',
                 'random0_bij_data', 'random1_bij_data', 'random2_bij_data', 'random_2_flows_data',
                 'random_3_flows_data', 'random_4_flows_data', 'hotspot_one_to_one_data']
TRAFFIC_FILES = ['stag_prob_0_2_3_data']

LABELS = ['stag0(0.2,0.3)', 'stag1(0.2,0.3)', 'stag2(0.2,0.3)', 'stag0(0.5,0.3)',
          'stag1(0.5,0.3)', 'stag2(0.5,0.3)', 'stride1', 'stride2',
          'stride4', 'stride8', 'rand0', 'rand1', 'rand2', 'randbij0',
          'randbij1', 'randbij2', 'randx2', 'randx3', 'randx4', 'hotspot']
LABELS = ['stag0(0.2,0.3)']

QLEN_TRAFFICS = ['stag_prob_2_2_3_data',
                 'stag_prob_2_5_3_data', 'stride1_data',
                 'stride2_data', 'random0_data',
                 'random_2_flows_data',
                 'random_3_flows_data', 'random_4_flows_data']

QLEN_LABELS = ['stag2(0.2,0.3)',
               'stag2(0.5,0.3)', 'stride1', 'stride2',
               'rand0',
               'randx2', 'randx3', 'randx4']


INPUT_DIR = 'inputs'
OUTPUT_DIR = 'results'
DURATION = 60
NONBLOCK_SW = '1001'
HEDERA_SW = '[0-3]h[0-1]h1'
FATTREE_SW = '300[1-9]'
DUMBBELL_SW = '100[1]|1002-eth3'


def get_test_config():
    algos = {}
    # algos['nonblocking'] = {'sw': NONBLOCK_SW, 'tf': 'default', 'pre': 'nonblocking', 'color': 'royalblue'}
    algos['iroko'] = {'sw': FATTREE_SW, 'tf': 'iroko',
                      'pre': 'fattree-iroko', 'color': 'green'}
    # algos['ecmp'] = {'sw': FATTREE_SW, 'tf': 'default', 'pre': 'fattree-ecmp', 'color': 'magenta'}
    # algos['dctcp'] = {'sw': FATTREE_SW, 'tf': 'default', 'pre': 'fattree-dctcp', 'color': 'brown'}
    # algos['hedera'] = {'sw': HEDERA_SW, 'tf': 'hedera', 'pre': 'fattree-hedera', 'color': 'red'}
    return algos


def prune(input_dir, output_dir, duration, traffic_files, offset, epochs, algorithm):
    conf = algorithm[1]
    algo = algorithm[0]
    for e in range(offset, epochs + offset):
        for tf in traffic_files:
            input_file = '%s/%s/%s' % (input_dir, conf['tf'], tf)
            pre_folder = "%s_%d" % (conf['pre'], e)
            out_dir = '%s/%s/%s' % (output_dir, pre_folder, tf)
            plotter.prune_bw(out_dir, tf, conf['sw'])


def get_num_interfaces(pattern):
    n = []
    for each in sre_yield.AllStrings(r'%s' % pattern):
        n.append(each)
    return len(n)


def yn_choice(message, default='y'):
    shall = 'N'
    shall = raw_input("%s (y/N) " % message).lower() == 'y'
    return shall


def end_of_episode(plotter, r):
    episode_rewards = np.array(r.episode_rewards)
    print('Average Episode Reward: {}'.format(episode_rewards.mean()))
    r.agent.save_model(directory=ARGS.save_dir)
    return True


if __name__ == '__main__':

    if yn_choice("Do you want to remove the reward file?"):
        print("Okay! Deleting...")
        try:
            os.remove("reward.txt")
        except OSError:
            pass

    algorithms = get_test_config()
    # Stupid hack, do not like.
    n = get_num_interfaces(DUMBBELL_SW)
    # Train on the dumbbell topology
    if ARGS.dumbbell is True:
        algorithms = {}
        algorithms['dumbbell'] = {
            'sw': DUMBBELL_SW, 'tf': 'dumbbell', 'pre': 'dumbbell-iroko', 'color': 'green'}
        TRAFFIC_FILES = ['incast_2']
        DURATION = 600
        # traffic_files = ['incast_4']
        # traffic_files = ['incast_8']
        LABELS = ['incast']
        #ARGS.train = True

    elif ARGS.env is True:

        algorithms = {}
        algorithms['dumbbell_env'] = {
            'sw': DUMBBELL_SW, 'tf': 'dumbbell', 'pre': 'dumbbell-iroko', 'color': 'green'}
        TRAFFIC_FILES = ['incast_2']
        DURATION = 600
        # traffic_files = ['incast_4']
        # traffic_files = ['incast_8']
        LABELS = ['incast']
        #ARGS.train = True

        traffic_file = TRAFFIC_FILES[0]
        algo, conf = algorithms.items()[0]
        if ARGS.plot is not True:
            environment = DataCenterEnv(
                INPUT_DIR, OUTPUT_DIR, DURATION, traffic_file, (algo, conf), ARGS.offset, ARGS.epochs)

            environment.reset()

            for i in range(0, 100):
                action = environment.action_space.sample()
                print(action)
                environment.step(action)


    else:
        print("Doing nothing...\nRun the command with --train to train/ the Iroko agent and/or --test to run benchmarks.")
