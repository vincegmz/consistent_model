import math
import random
import os
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from cm.image_datasets import random_crop_arr,center_crop_arr
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
import warnings
import torch
class StandardAugmentDataset(VisionDataset):

    resources = [
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz',
         '191ed53db9933bd85cc9700558847391'),
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz',
         'e11cb4d7fff76d7ec588b1134907db59')
    ]

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data
    def __init__(
        self,
        root,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        train = True,
    ):
        super(StandardAugmentDataset, self).__init__(root = root)
        self.resolution = resolution
        self.train = train
      
        imgs = []
        if self.train:
            data_file = self.training_file
            angles = [45,90,135,180,225,270,315]
            color_jitter_values = [0.5,0.75]
            hue_values = [0.25,0.5]
            updated_classes = []
            for i, path in enumerate(image_paths):
                with bf.BlobFile(path, "rb") as f:
                    pil_image = Image.open(f)
                    pil_image.load()
                pil_image = pil_image.convert("RGB")
                h_flipped_image = TF.hflip(pil_image)
                rotated_images = [TF.rotate(pil_image,angle = angle) for angle in angles]
                color_jitter_imgs = [ColorJitter(brightness=i,saturation=j,contrast=k,hue=z)(pil_image)
                                    for i in color_jitter_values for j in color_jitter_values for k in color_jitter_values for z in hue_values]
                imgs.append(pil_image)
                # imgs.append(h_flipped_image)
                # imgs.extend(rotated_images)
                # imgs.extend(color_jitter_imgs)
                # updated_classes.extend([classes[i]]*25)
            # classes = updated_classes
            if len(imgs) !=100:
                factor = int(100/len(imgs))
                imgs = imgs*factor
                updated_classes.extend([classes[i]]*100)
                classes = updated_classes
                
            # assert len(imgs) == 100
        else:
            data_file = self.test_file

        data, targets = torch.load(os.path.join(self.processed_folder, data_file))

        if self.train:
            count_dict = {digit:0 for digit in range(0,10)}
            count_dict.pop(classes[0])
        else:
            count_dict = {digit:0 for digit in range(classes[0],classes[0]+1)}
            classes = []
        # data_file = self.training_file
        # count_dict = {digit:0 for digit in range(1,10)}
        # data, targets = torch.load(os.path.join(self.processed_folder, data_file))
        index = 0
        assert type(classes) is list
        while count_dict != {}:
            num = targets[index].item()
            if self.train:
                if num == classes[0]:
                    index+=1
                    continue
            if num not in count_dict:
                index+=1
                continue
            classes.append(num)
            img = Image.fromarray(data[index].squeeze().numpy(), mode="RGB")
            imgs.append(img)
            count_dict[num]+=1
            index+=1
            if count_dict[num] == 100:
                count_dict.pop(num)
        # assert len(imgs) == 1000
        self.local_images = imgs[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        pil_image = self.local_images[idx]

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
    
    
    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNISTM', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNISTM', 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the MNIST-M data."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")