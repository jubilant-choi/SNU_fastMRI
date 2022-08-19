import os
import re
import h5py
import random
import numpy as np
from pathlib import Path

from utils.data.transforms import DataTransform, DataTransform_kspace
from torch.utils.data import Dataset, DataLoader

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, tv=None, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        if not forward:
            image_files = list(Path(root / "image").iterdir()) if isinstance(root, Path) else make_CV_dataset(int(root),'image', tv)
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]
        
        kspace_files = list(Path(root / "kspace").iterdir()) if isinstance(root, Path) else make_CV_dataset(int(root),'kspace',tv)
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
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
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)

    
class SliceDataUnet(Dataset):
    def __init__(self, root, transform, input_key, target_key, tv=None, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.examples = []
        self.labels = []
        
        if not self.forward:
            files = make_CV_dataset(int(root),'recon/kspace', tv)
            labels = make_CV_dataset(int(root),'image', tv)

            for fname in sorted(files):
                num_slices = self._get_metadata(fname)

                self.examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]
                name = fname.name.split('/')[-1]
                self.labels += [
                    (Path(f'/root/input/image/{name}'), slice_ind) for slice_ind in range(num_slices)
                ]
        else:
            files = list(Path(root / "image").iterdir())
            for fname in sorted(files):
                num_slices = self._get_metadata(fname)

                self.examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]
            self.labels = self.examples
            
    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf[self.input_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, dataslice = self.examples[i]
        label_fname , label_dataslice = self.labels[i]
        
        with h5py.File(fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            
        with h5py.File(label_fname, "r") as hf:
            if self.forward:
                target = -1
            else:
                target = hf[self.target_key][label_dataslice]
            attrs = dict(hf.attrs)
        return self.transform(input, target, attrs, fname.name, dataslice)

def create_data_loaders(data_path, args, tv=None, isforward=False):
    
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
     
    adaptive_VN = False
    if args.input_key == 'kspace':
        if 'AdaptiveVarNet' == args.net_name.name:
            adaptive_VN = True
#         DataTransform = DataTransform_kspace
        
    if args.CV != None:
        data_path = args.CV
    
    if args.net_name.name == 'Unet':
        SliceData = SliceDataUnet

    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_, adaptive_VN),
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

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def make_CV_dataset(num_CV, dtype, tv):
    data_path = f'/root/input/{dtype}/' 
    files = os.listdir(data_path)
    files.sort(key=natural_keys)
    whole_files = list(map(lambda x: Path(data_path+x), files))
    idx_first = -(num_CV * 82) if num_CV != 5 else 0
    idx_last = idx_first + 82 if num_CV != 1 else None
    
    if tv == 'train':
        return list(set(whole_files) - set(whole_files[idx_first:idx_last]))
    elif tv == 'val':
        return whole_files[idx_first:idx_last]
