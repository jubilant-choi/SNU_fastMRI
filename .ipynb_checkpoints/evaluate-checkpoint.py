<<<<<<< HEAD
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

import argparse
import shutil
import hashlib

from utils.learning.test_part import forward
from pathlib import Path

def parse():
    parser = argparse.ArgumentParser(description='Test on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('-n', '--net-name', type=Path, required=True, help='Name of network')
    
    parser.add_argument('-p', '--data-path', type=Path, default='/root/input/leaderboard/', help='Directory of test data')
    
    parser.add_argument('--cascade', type=int, default=2, help='Number of cascades | Should be less than 12')
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument('--input-key', type=str, default='image_input', help='Name of input key')
    
    parser.add_argument('-u', '--user', type=str, choices=['SJ','JB'], required=True, help='User name')
    parser.add_argument('-x', '--exp-name', type=str, default='test', help='Name of an experiment')
    parser.add_argument('--CV', type=int, default=None, help='Name of CV data')
    parser.add_argument('--boost', type=bool, default=False, help='enable boosting')
    parser.add_argument('--ensemble', type=bool, default=False, help='enable model ensemble')
      
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse()
    args.exp_dir = './result' / Path(args.user) / args.net_name / 'checkpoints'
    args.forward_dir = './result' / Path(args.user) / args.net_name / 'reconstructions_forward' / args.exp_name 
    if args.data_path.name == '/root/input':
        args.forward_dir = Path('/root/input/recon/kspace')
=======
import argparse
from pathlib import Path
from utils.learning.test_part import forward

def parse():
    parser = argparse.ArgumentParser(description='Test Unet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_Unet', help='Name of network')
    parser.add_argument('-p', '--data_path', type=Path, default='/root/input/leaderboard/image', help='Directory of test data')
    
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument("--input_key", type=str, default='image_input', help='Name of input key')
        
    parser.add_argument('-u', '--user', type=str, choices=['SJ','JB'], required=True, help='User name')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = './result' / Path(args.user) /  args.net_name / 'checkpoints'
    args.forward_dir = './result' / Path(args.user) / args.net_name / 'reconstructions_forward' 
>>>>>>> 1e444fb26a5b3c334a06d6b0e5bfa94f98ec8246
    print(args.forward_dir)
    forward(args)


