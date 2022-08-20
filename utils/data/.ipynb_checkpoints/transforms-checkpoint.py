import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
            
        if isinstance(mask,type(None)):
            input = to_tensor(input)
            mask = 0
        else:
            input = to_tensor((input * mask + 0.0))
            input = torch.stack((input.real, input.imag), dim=-1)
            mask = torch.from_numpy(mask.reshape(1, 1, input.shape[-2], 1).astype(np.float32)).byte()
            
        return mask, input, target, maximum, fname, slice