import os
import re
import h5py
import random
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader


class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, tv=None, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.input_examples = []

        if not forward:
            image_files = list(Path(root / "image").iterdir()) if isinstance(root, Path) else make_dataset(root,'image', tv)
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]
        
        if input_key == 'kspace':
            input_files = list(Path(root / "kspace").iterdir()) if isinstance(root, Path) else make_dataset(root,'kspace', tv)
        elif input_key == 'reconstruction':
            input_files = make_CV_dataset(int(root),'recon/kspace', tv)
        elif input_key in ['image_input', 'image_grappa']:
            input_files = image_files
            
        for fname in sorted(input_files):
            num_slices = self._get_metadata(fname)

            self.input_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        input_fname, dataslice = self.input_examples[i]
        
        with h5py.File(input_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask = np.array(hf["mask"]) if self.input_key == 'kspace' else None
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        return self.transform(mask, input, target, attrs, input_fname.name, dataslice)

    
def create_data_loaders(data_path, args, tv=None, isforward=False):
    
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
        
    if args.CV != None:
        data_path = args.CV
    if args.final:
        data_path = '/root/input/'
        tv= tv+'_final'
        
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward,
        tv = tv
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        num_workers=4 # edited
    )
    return data_loader


def make_dataset(root, dtype, tv):
    data_path = f'/root/input/{dtype}/'
    if isinstance(root,int): 
        files = os.listdir(data_path)
        num_CV = root
        files.sort(key=natural_keys)
        whole_files = list(map(lambda x: Path(data_path+x), files))
        idx_first = -(num_CV * 82) if num_CV != 5 else 0
        idx_last = idx_first + 82 if num_CV != 1 else None

        if tv == 'train':
            return list(set(whole_files) - set(whole_files[idx_first:idx_last]))
        elif tv == 'val':
            return whole_files[idx_first:idx_last]
    else:
        files = list((pd.read_csv('/root/input/low_acc_files.csv')).filenames)
        whole_files = list(map(lambda x: Path(data_path+x), files))
        split = int(len(whole_files)*0.8)
        if tv == 'train_final':
            return whole_files[:split]
        else:
            return whole_files[split:]
        


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
