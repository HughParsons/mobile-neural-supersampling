import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import torch.nn.functional as F

def warp(tensor: torch.Tensor, motion: torch.Tensor):
    B, _, H, W = tensor.size()
    # Construct a grid of relative pixel positions [-1,1] = [left, right] = [top, bottom]
    xx = torch.linspace(-1, 1, W, device=tensor.device).view(1,-1).repeat(H,1)
    yy = torch.linspace(-1, 1, H, device=tensor.device).view(-1,1).repeat(1,W)

    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)

    grid = torch.cat((xx, yy),1)
    # We use HDRP motion vectors which measure the forward movement of pixels
    # measured relative to the frame, i.e. -1 means the pixel moved to the left 
    # of the frame from the right. However, we invert these in the data loader.
    #
    # To get the pixel in the previous frame that we are sampling from we add 
    # motion vector from the grid of relative pixel positions
    vgrid = grid + motion

    return F.grid_sample(tensor, vgrid.permute(0, 2, 3, 1).to(tensor.dtype), mode="bilinear", align_corners=True)

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
