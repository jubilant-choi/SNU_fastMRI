import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

import argparse
import shutil
import hashlib

from utils.learning.train_part import train
from utils.common.utils import save_exp_result
from pathlib import Path

def parse():
    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=10, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, required=True, help='Name of network')
    parser.add_argument('-o', '--optim', type=str, default='Adam', help='Name of optimizer')
    parser.add_argument('-s', '--scheduler', type=str, default='Plateau', help='Name of lr scheduler')
    
    parser.add_argument('-t', '--data-path-train', type=Path, default='/root/input/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/root/input/val/', help='Directory of validation data')
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12')
    
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument('--input-key', type=str, default='image_input', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    
    parser.add_argument('--load', type=str, default='', help='Name of saved model that will be loaded')
    parser.add_argument('-u', '--user', type=str, choices=['SJ','JB'], required=True, help='User name')
    parser.add_argument('-x', '--exp-name', type=str, default='test', help='Name of an experiment')
      
    args = parser.parse_args()
#     tot_iter = 5164
#     args.report_interval = int((tot_iter/args.batch_size)/10)
    
    return args

if __name__ == '__main__':
    args = parse()
    args.exp_dir = './result' / Path(args.user) / args.net_name / 'checkpoints'
    args.val_dir = './result' / Path(args.user) / args.net_name / 'reconstructions_val'
    args.json_dir = './result' / Path(args.user) / args.net_name / 'jsons'
    args.main_dir = './result' / Path(args.user) / args.net_name / __file__

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)
    print(f"*** Experiment <{args.exp_name}> with model <{args.net_name}> starts ***")
    train(args)
    
# def run():
#     args = parse()
#     args.exp_dir = './result' / Path(args.user) / args.net_name / 'checkpoints'
#     args.val_dir = './result' / Path(args.user) / args.net_name / 'reconstructions_val'
#     args.json_dir = './result' / Path(args.user) / args.net_name / 'jsons'
#     args.main_dir = './result' / Path(args.user) / args.net_name / __file__

#     args.exp_dir.mkdir(parents=True, exist_ok=True)
#     args.val_dir.mkdir(parents=True, exist_ok=True)

#     train(args)
