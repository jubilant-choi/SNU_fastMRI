import os
import numpy as np
import h5py

def set_cmask():
    mask = np.ones((384,384),dtype=np.int8)
    r = 384//2
    center = np.array((r,r))

    for x in range(384):
        for y in range(384):
            if np.linalg.norm(np.array((x,y))-center) >= r-30:
                mask[x][y] = 0
    return mask

def label_cmask():
    cmask = set_cmask()
    for file in os.listdir():
        with h5py.File(file, "a") as hf:
            hf['image_label_cmask'] = hf['image_label'] * cmask
try:
    os.chdir('/root/input/train/image')
    label_cmask()
except:
    pass
try:
    os.chdir('/root/input/val/image')
    label_cmask()
except:
    pass