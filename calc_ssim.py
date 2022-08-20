import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import random
import glob
import os
import torch
from utils.common.loss_function import SSIMLoss
import torch.nn.functional as F
import cv2 

class SSIM(SSIMLoss):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__(win_size, k1, k2)
            
    def forward(self, X, Y, data_range):
        if len(X.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(X.shape)))
        if len(Y.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(Y.shape)))
            
        X = X.unsqueeze(0).unsqueeze(0)
        Y = Y.unsqueeze(0).unsqueeze(0)
        #data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        return S.mean()
    

def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    leaderboard_data = glob.glob(os.path.join(args.leaderboard_data_path,'*.h5'))
    
    your_data = glob.glob(os.path.join(args.your_data_path,'*.h5'))
    
    ssim_total = 0
    idx = 0
    ssim_calculator = SSIM().to(device=device)
    
    fnames = []
    ssims = []
    
    with torch.no_grad():
        for i_subject in range(len(your_data)):
            print(i_subject,'calculating')
            l_fname = os.path.join(args.leaderboard_data_path, 'brain' + str(i_subject+1) + '.h5')
            y_fname = os.path.join(args.your_data_path, 'brain' + str(i_subject+1) + '.h5')
            with h5py.File(l_fname, "r") as hf:
                num_slices = hf['image_label'].shape[0]
            for i_slice in range(num_slices):
                with h5py.File(l_fname, "r") as hf:
                    target = hf['image_label'][i_slice]
                    mask = np.zeros(target.shape)
                    mask[target>5e-5] = 1
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.erode(mask, kernel, iterations=1)
                    mask = cv2.dilate(mask, kernel, iterations=15)
                    mask = cv2.erode(mask, kernel, iterations=14)
                    
                    target = torch.from_numpy(target).to(device=device)
                    mask = (torch.from_numpy(mask).to(device=device)).type(torch.float)

                    maximum = hf.attrs['max']
                    
                with h5py.File(y_fname, "r") as hf:
                    recon = hf[args.output_key][i_slice]
                    recon = torch.from_numpy(recon).to(device=device)
                    
                #ssim_total += ssim_calculator(recon, target, maximum).cpu().numpy()
                curr_ssim = ssim_calculator(recon*mask, target*mask, maximum).cpu().numpy()
                ssim_total += curr_ssim
                ssims.append(curr_ssim)
                fnames.append(y_fname)
                idx += 1
    res = pd.DataFrame({'fnames':fnames,'ssims':ssims})        
    res.to_csv('res_all.csv')
    print("Leaderboard Dataset SSIM : {:.4f}".format(ssim_total/idx))


if __name__ == '__main__':
    """
    Image Leaderboard Dataset Should Be Utilized
    For a fair comparison, Leaderboard Dataset Should Not Be Included When Training. This Is Of A Critical Issue.
    Since This Code Print SSIM To The 4th Decimal Point, You Can Use The Output Directly.
    """
    parser = argparse.ArgumentParser(description=
                                     'FastMRI challenge Leaderboard Image Evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0)
    parser.add_argument('-lp', '--leaderboard_data_path', type=str, default='/root/input/image')
    """
    Modify Path Below To Test Your Results
    """
    parser.add_argument('-yp', '--your_data_path', type=str, default='../result/test_Unet/reconstructions_forward/')
    parser.add_argument('-key', '--output_key', type=str, default='reconstruction')
    
    parser.add_argument('-n', '--net_name', type=Path, default='test_Unet', help='Name of network')
    parser.add_argument('-u', '--user', type=str, choices=['SJ','JB'], required=True, help='User name')
    parser.add_argument('-x', '--exp-name', type=str, default='test', help='Name of an experiment')
    
    args = parser.parse_args()
    if args.your_data_path == '../result/test_Unet/reconstructions_forward/':
        args.your_data_path = './result' / Path(args.user) / args.net_name / 'reconstructions_forward' / args.exp_name
    
    forward(args)
