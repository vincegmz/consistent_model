from cm.image_datasets import ImageDataset
from torch.utils.data import Dataset
import os
from pathlib import Path
from array import array
import struct
from torchvision.datasets import MNIST
import numpy as np
from PIL import Image
from cm.image_datasets import random_crop_arr,center_crop_arr
import random
class MNIST(Dataset):
    def __init__(self,data_dir = '/media/minzhe_guo/ckpt/dataset/mnist',
                shard=0,
                num_shards=1,
                random_crop=False,
                random_flip=True,
                resolution = None,
                class_cond = False,
                train = True,
                train_classifier = False,
                *args,**kwargs):
        if train:
            with open(os.path.join(data_dir,'train-images.idx3-ubyte'), "rb") as f:
                # IDX file format
                magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
                image_data = array("B", f.read())
        else:
            with open(os.path.join(data_dir,'t10k-images.idx3-ubyte'), "rb") as f:
                # IDX file format
                magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
                image_data = array("B", f.read())
        images = []
        for i in range(size):
            image = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28, 28)
            images.append(image)
        self.data = np.array(images)
        
        if train:
            with open(os.path.join(data_dir,'train-labels.idx1-ubyte'), "rb") as f:
                magic, size = struct.unpack(">II", f.read(8))
                if magic != 2049:
                    raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
                self.targets = np.array(array("B", f.read()))
        else:
            with open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'), "rb") as f:
                magic, size = struct.unpack(">II", f.read(8))
                self.targets = np.array(array("B", f.read()))

        self.class_cond = class_cond
        self.data = self.data[shard:][::num_shards]
        self.targets = None if not self.class_cond else self.targets[shard:][::num_shards]
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train_classifier = train_classifier
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
                # to return a PIL Image\
        img, target = self.data[idx], None if not self.class_cond else int(self.targets[idx])
        img = Image.fromarray(img, mode="L")
        if self.random_crop:
            arr = random_crop_arr(img, self.resolution)
        else:
            arr = center_crop_arr(img, self.resolution)
        if not self.train_classifier:
            arr = np.repeat(arr[:,:,None],3,axis = 2)
        else:
            arr = arr[:,:,None]
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        # if self.transform is not None:
        #     img = self.transform(arr)
      
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        out_dict = {}
        if target is not None:
            out_dict["y"] = np.array(target, dtype=np.int64)
        if self.train_classifier:
            out_dict = np.array(target, dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
