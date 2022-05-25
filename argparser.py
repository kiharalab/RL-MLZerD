import parser
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein',type=str, required=True,help='protein name, e.g 2AZE')
    parser.add_argument('--nofchains',type=int,required=True,help='number of chains in protein e.g 3')
    parser.add_argument('--chains',type=str,required=True, help='list of chains, e.g A,B,C')
    parser.add_argument('--path',type=str,required=True, help='path to data directory')
    parser.add_argument('--clash_threshold',type=int,default='300',help='clash threshold for atom atom clashes')
    parser.add_argument('--pair_clash_threshold',type=int,default='300',help='pairwise clash threshold for atom atom clashes')
    parser.add_argument('--lr', type=float, default='1.0', help='learning rate for training e.g 0.01')
    parser.add_argument('--eps', type=float, default='1.0', help='epsilon for agent exploration, e.g 0.3')
    parser.add_argument('--metro_threshold', type=float, default='0.6', help='Metropolic acceptance threshold, e.g 0.2')
    parser.add_argument('--episodes', type=int, default='1000', help='number of episodes to run')
    parser.add_argument('--pool_size', type=int, default='1000', help='number of decoy pool from pairwise to use')
    parser.add_argument('--out_dir', type=str, required=True, help='subdirectory in the data path to write output too')
    parser.add_argument('--not_int_pair', type=str, help='none interacting pair list e.g AB|CD')
    parser.add_argument('--int_pair', type=str, help='interacting pair list e.g A-BCD|B-A|D-ACE')
    parser.add_argument('--terminal_thres', type=int, default=500, help='Threshold for the number of time a goal state can be visited before qtable and tree is dump and everything restarts')
    parser.add_argument('--unbound',type=str, help='information about bound or unbound docking', default='no')
    parser.add_argument('--classifier',type=str, help='Which fitted classifier to use. LR|SGD', default='sgd')
    args = parser.parse_args()
    params = vars(args)
    return params
