import os

from base import BaseDataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as tf

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from collections import deque
from typing import Union, Tuple, List


class MNSSDataLoader(BaseDataLoader):
    """
    Generate batch of data
    `for x_batch in data_loader:`
    `x_batch` is a list of 4 tensors, meaning `view, depth, motion, view_truth`
    each size is (batch x channel x height x width)
    """
    def __init__(self,
                 data_dir: str,
                 img_dirname: str,
                 depth_dirname: str,
                 motion_dirname: str,
                 batch_size: int,
                 shuffle: bool = True,
                 validation_split: float = 0.0,
                 num_workers: int = 1,
                 downsample: Union[Tuple[int, int], List[int], int] = (2, 2),
                 num_data: Union[int,None] = None,
                 resize_factor : Union[int, None] = None,
                 num_frames: int = 5,
                 ):
        dataset = MNSSDataset(data_dir,
                              img_dirname=img_dirname,
                              depth_dirname=depth_dirname,
                              motion_dirname=motion_dirname,
                              downsample=downsample,
                              num_data=num_data,
                              resize_factor = resize_factor,
                              num_frames = num_frames,
                              )
        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         validation_split=validation_split,
                         num_workers=num_workers,
                         )


class MNSSDataset(Dataset):
    """
    Requires that corresponding view, depth and motion frames share the same name.
    """
    def __init__(self,
                 data_dir: str,
                 img_dirname: str,
                 depth_dirname: str,
                 motion_dirname: str,
                 downsample: int = 2,
                 num_data:Union[int, None] = 5,
                 resize_factor:Union[int, None] = 2,
                 num_frames: int = 10
                #  transform: nn.Module = None,
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.img_dirname = img_dirname
        self.depth_dirname = depth_dirname
        self.motion_dirname = motion_dirname

        self.resize_factor = resize_factor
        self.downsample = downsample

        # if transform is None:
        self.transform = tf.ToTensor()
        self.downscale = lambda tensor, ratio: F.interpolate(tensor.unsqueeze(0), size=(tensor.shape[1]//ratio, tensor.shape[2]//ratio), mode='nearest', antialias=True).squeeze(0)

        self.img_list = os.listdir(os.path.join(self.data_dir, self.img_dirname))
        self.img_list = sorted(self.img_list, key=lambda keys:[ord(i) for i in keys], reverse=False)
        
        if num_data is None:
            num_data = len(self.img_list)

        self.data_list = []
        # maintain a buffer for the last num_frames frames
        img_name_buffer = deque(maxlen=num_frames) 

        for i, img_name in enumerate(self.img_list):

            if(i>=num_data + num_frames - 1):
                break
                
            # handle scene change
            # TODO: more complex scene names — currently just [a-zA-Z]\d+
            if len(img_name_buffer) and img_name[0] != img_name_buffer[0][0]:
                img_name_buffer.clear()

            img_name_buffer.appendleft(img_name)

            if len(img_name_buffer) == num_frames:
                self.data_list.append(list(img_name_buffer))
                
    def __getitem__(self, index):
        data = self.data_list[index]

        view_list, depth_list, motion_list, truth_list = [], [], [], []
        # elements in the lists following the order: current frame i, pre i-1, pre i-2, pre i-3, pre i-4
        for frame in data:
            frame, _ = frame.rsplit('.', 1)
            img_path = os.path.join(self.data_dir, self.img_dirname, f"{frame}.png")
            depth_path = os.path.join(self.data_dir, self.depth_dirname, f"{frame}.exr")
            motion_path = os.path.join(self.data_dir, self.motion_dirname, f"{frame}.exr")
            
            img_view_truth = self.transform(cv2.imread(img_path))
            # depth data is in the 3rd channel (channels are BGR)
            img_depth = self.transform(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED))[2:3,:,:]
            # motion data is in the 2nd and 3rd channels (channels are BGR)
            img_motion = self.transform(cv2.imread(motion_path, cv2.IMREAD_UNCHANGED))[1:3,:,:]
            
            # resize images
            img_view_truth = self.downscale(img_view_truth, self.resize_factor)
            img_view = self.downscale(img_view_truth, self.downsample)

            img_depth = self.downscale(img_depth, self.resize_factor * self.downsample)
            img_motion = self.downscale(img_motion, self.resize_factor * self.downsample)

            # swap channels to match the order of the motion vectors
            img_motion = img_motion[[1,0],:,:]
            img_motion[1] = -img_motion[1] # flip y axis
            img_motion = img_motion * -1 # point backwards

            view_list.append(img_view)
            depth_list.append(img_depth)
            motion_list.append(img_motion)
            truth_list.append(img_view_truth)
            

        return view_list, depth_list, motion_list, truth_list

    def __len__(self) -> int:
        return len(self.data_list)
