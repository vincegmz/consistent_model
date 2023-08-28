from cm.image_datasets import ImageDataset
from torch.utils.data import Dataset
import os
from PIL import Image
from cm.image_datasets import center_crop_arr, random_crop_arr
import blobfile as bf
import random
import numpy as np
import torch
class DomainNet(Dataset):
    def __init__(self,root = '/media/minzhe_guo/ckpt/dataset/domainNet',
                train = True,
                domain = 'real',
                class_cond = False,
                shard=0,
                num_shards=1,
                random_crop=False,
                random_flip=True,
                resolution = 64,
                train_classifier = False,
                transforms = None,
                ):
        
        image_paths = []
        if train:
            mode = 'train'
        else:
            mode = 'test'
        target_file = os.path.join(root,f'{mode}_{domain}_64x64.pt')
        if os.path.exists(target_file):
            self.data,self.target = torch.load(target_file)
        else:
            raise FileNotFoundError('dataset file not found')
        
        # resize_folder = root.replace('domainNet','domainNetResize')
        # if not os.path.exists(resize_folder):
        #     resize = True
        # else:
        #     resize = False
        
        # resize = not os.path.exists(os.path.join(resize_folder,domain))
        # with open(os.path.join(root,target_file),'r') as f:
        #     info = f.read().splitlines()
        #     for i, line in enumerate(info):
        #         rel_path,label = line.split()
        #         full_path = os.path.join(root,rel_path)
        #         target_path = full_path.replace('domainNet','domainNetResize')
        #         if resize:
        #             os.makedirs(os.path.join(resize_folder,domain),exist_ok=True)
        #             with bf.BlobFile(full_path, "rb") as f:
        #                 pil_image = Image.open(f)
        #                 pil_image.load()
        #             pil_image = pil_image.convert("RGB")
        #             if random_crop:
        #                 arr = random_crop_arr(pil_image, resolution)
        #             else:
        #                 arr = center_crop_arr(pil_image, resolution)
        #             resized_image = Image.fromarray(arr)
        #             os.makedirs(os.path.dirname(target_path), exist_ok=True)
        #             resized_image.save(target_path)
        #         image_paths.append(target_path)
        #         if class_cond:
        #             labels.append(label)

        self.resolution = resolution
        self.local_images = self.data[shard:][::num_shards]
        self.local_classes = None if not class_cond else self.target[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train_classifier = train_classifier
        self.class_cond = class_cond
        self.transforms = transforms
        
    def __len__(self):
        return len(self.local_images)
    
    def __getitem__(self, idx):

        arr, target = self.data[idx], None if not self.class_cond else int(self.target[idx])

        if self.train_classifier:
            assert self.transforms is not None
            pil_image = Image.fromarray(arr.numpy())
            tensor_img = self.transforms(pil_image)
            arr = tensor_img.numpy()
            assert target is not None
            out_dict= np.array(target, dtype=np.int64)
            return arr, out_dict

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        else:
            if self.random_flip and random.random() < 0.5:
                arr = arr.numpy()[:, ::-1]
            arr = arr.numpy()
            arr = arr.astype(np.float32) / 127.5 - 1
            out_dict = {}
            if target is not None:
                out_dict["y"] = np.array(target, dtype=np.int64)
            return np.transpose(arr, [2, 0, 1]), out_dict
        # if self.transform is not None:
        #     arr = self.transform(arr)
      
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

       
       
        # path = self.local_images[idx]
        # with bf.BlobFile(path, "rb") as f:
        #     pil_image = Image.open(f)
        #     pil_image.load()
        # pil_image = pil_image.convert("RGB")
        # arr = np.array(pil_image)
        # if self.random_flip and random.random() < 0.5:
        #     arr = arr[:, ::-1]

        # arr = arr.astype(np.float32) / 127.5 - 1

        # out_dict = {}
        # if self.local_classes is not None:
        #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        # if self.train_classifier:
        #     out_dict = np.array(self.local_classes[idx], dtype=np.int64)
        # return np.transpose(arr, [2, 0, 1]), out_dict
