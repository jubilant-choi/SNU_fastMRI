import argparse
import shutil
import hashlib
from utils.learning.train_part import train
from pathlib import Path

def parse():
    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=10, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_Unet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/root/input/train/image/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/root/input/val/image/', help='Directory of validation data')
    
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument('--input-key', type=str, default='image_input', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    
    parser.add_argument('-u', '--user', type=str, choices=['SJ,JB'], required=True, help='User name')
    parser.add_argument('-x', '--exp-name', type=str, default='test', help='Name of an experiment')
      
    args = parser.parse_args()
    
    tot_iter = 5164
    args.report_interval = int((tot_iter/args.batch_size)/10)
    
    return args

if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / Path(args.user) / args.net_name / 'checkpoints'
    args.val_dir = '../result' / Path(args.user) / args.net_name / 'reconstructions_val'
    args.json_dir = '../result' / Path(args.user) / args.net_name / 'jsons'
    args.main_dir = '../result' / Path(args.user) / args.net_name / __file__

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    result = train(args)
    save_exp_result(save_dir=args.json_dir, setting=vars(args).copy(), result=result)