import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset





def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    invert = False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    dataset_type = data_dir.split('/')[-1]
    if dataset_type == 'mnistm':
        from datasets.mnistm import MNISTM
        dataset = MNISTM(
        data_dir,
        train = True,
        resolution = image_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        class_cond=class_cond
        )
        
    else:
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            if invert == 'same':
                class_names = [data_dir.split('/')[-1] for _ in all_files]
            else:
                class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        dataset_type = data_dir.split('/')[-1]
        if dataset_type == 'mnist-original' or dataset_type == 'mnist':
            from datasets.mnist import MNIST
            dataset = MNIST(
            data_dir,
            resolution = image_size,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip, 
        )
        else:
            dataset = ImageDataset(
                image_size,
                all_files,
                classes=classes,
                shard=MPI.COMM_WORLD.Get_rank(),
                num_shards=MPI.COMM_WORLD.Get_size(),
                random_crop=random_crop,
                random_flip=random_flip,
            )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    return loader
    # while True:
    #     yield from loader

def create_dataset(  *,
    data_dir,
    image_size,
    class_cond=False,
    random_crop=False,
    random_flip=False,
    train = True,
    train_classifier = True,
    domain = None,
    transforms = None,
    standard_augment = False,
    class_label= 0):
    if not data_dir:
        raise ValueError("unspecified data directory")
    dataset_type = data_dir.split('/')[-1]
    if dataset_type == 'svhn':
        assert image_size == 32
        from datasets.svhn import SVHN
        if train:
            split = 'train'
        else:
            split = 'test'
        dataset = SVHN(
        data_dir,
        resolution = image_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        download=True,
        class_cond=class_cond,
        split = split,
        train_classifer=train_classifier
        )
    elif dataset_type == 'mnistm':
        assert image_size == 28
        from datasets.mnistm import MNISTM
        dataset = MNISTM(
        data_dir,
        train = train ,
        resolution = image_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        class_cond=class_cond,
        train_classifier= train_classifier,
        standard_augment = standard_augment
        )
    elif dataset_type == 'domainNet':
        if domain is None:
            raise NotImplementedError('domainNet requires domain parameter')
        assert domain in ['clipart','real','sketch','painting','infograph','quickdraw']
        from datasets.domainNet import DomainNet
        dataset = DomainNet(
            root=data_dir,
            train=train,resolution = image_size,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            domain = domain,
            class_cond = class_cond,
            train_classifier = train_classifier,
            transforms = transforms
        )
    else:
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [data_dir.split('/')[-1] for _ in all_files]
            # if standard_augment:
            #     class_names = [data_dir.split('/')[-1] for _ in all_files]
            
            # else:
            #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        dataset_type = data_dir.split('/')[-1]
        if dataset_type == 'mnist-original' or dataset_type == 'mnist':
            assert image_size == 28
            from datasets.mnist import MNIST
            dataset = MNIST(
            data_dir,
            resolution = image_size,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            class_cond = class_cond,
            train = train,
            train_classifier = train_classifier
        )
        elif not standard_augment:
            root = '/media/minzhe/ckpt/dataset/mnistm'
            from datasets.eval_inversion import EvalInversionDataset
            classes = [class_label for _ in range(len(classes))]
            dataset = EvalInversionDataset(root,image_size,
                all_files,
                classes=classes,
                shard=MPI.COMM_WORLD.Get_rank(),
                num_shards=MPI.COMM_WORLD.Get_size(),
                random_crop=random_crop,
                random_flip=random_flip,
                train=train
            )
        else:
            if standard_augment:
                root = '/media/minzhe/ckpt/dataset/mnistm'
                from datasets.standard_augment import StandardAugmentDataset
                # if data_dir.split('/')[-2] == 'color_digits':
                #     dict = {'Zero':0,'One':1,'Two':2,'Three':3,'Four':4,'Five':5,'Six':6,'Seven':7,'Eight':8,'Nine':9}
                #     classes = [dict[dataset_type] for _ in range(len(classes))]
                classes = [class_label for _ in range(len(classes))]
                dataset = StandardAugmentDataset(root,image_size,
                    all_files,
                    classes=classes,
                    shard=MPI.COMM_WORLD.Get_rank(),
                    num_shards=MPI.COMM_WORLD.Get_size(),
                    random_crop=random_crop,
                    random_flip=random_flip,
                    train=train,
                )
            else:
                dataset = ImageDataset(
                image_size,
                all_files,
                classes=classes,
                shard=MPI.COMM_WORLD.Get_rank(),
                num_shards=MPI.COMM_WORLD.Get_size(),
                random_crop=random_crop,
                random_flip=random_flip,
                )
    return dataset

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
