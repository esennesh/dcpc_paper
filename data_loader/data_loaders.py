import numpy as np
import os
from PIL import Image
import torch
from torchvision import datasets, transforms
from typing import Any, Callable, Optional, Tuple
from base import BaseDataLoader

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, idx):
        (data, target) = self._dataset[idx]
        return (data, target, np.array(idx))

    def __len__(self):
        return len(self._dataset)

class BouncingMNIST(datasets.VisionDataset):
    def __init__(self, data_dir: str, transform: Optional[Callable]=None,
                 target_transform: Optional[Callable]=None):
        super().__init__(data_dir + "/bmnist", transform=transform,
                         target_transform=target_transform)
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for file in os.listdir(self.root):
            if '.npy' not in file:
                continue
            file = os.path.join(self.root, file)
            data.append(torch.from_numpy(np.load(file)).float())
        result = torch.cat(data, dim=0)
        return result

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        imgs = self.data[index].numpy()

        if self.transform is not None:
            imgs = self.transform(imgs.swapaxes(0, -1))

        return (imgs, 0)

    def __len__(self):
        return len(self.data)

class MiniBouncingMnist(BouncingMNIST):
    def _load_data(self):
        data = []
        for file in os.listdir(self.root):
            if '.npy' not in file:
                continue
            file = os.path.join(self.root, file)
            data.append(torch.from_numpy(np.load(file)).float())
            break
        result = torch.cat(data, dim=0)
        return result

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=False, validation_split=0.0, num_workers=1, training=True, drop_last=False):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = IndexedDataset(datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, drop_last=drop_last)

class BouncingMnistDataLoader(BaseDataLoader):
    """
    Bouncing MNIST data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=False, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.mT)
        ])
        self.data_dir = data_dir
        self.dataset = IndexedDataset(BouncingMNIST(self.data_dir, transform=trsfm))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MiniBouncingMnistDataLoader(BaseDataLoader):
    """
    Bouncing MNIST data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=False, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.mT)
        ])
        self.data_dir = data_dir
        self.dataset = IndexedDataset(MiniBouncingMnist(self.data_dir, transform=trsfm))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CelebADataLoader(BaseDataLoader):
    """
    CelebA data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, img_side=64, shuffle=False, validation_split=0.0, num_workers=1, training=True, drop_last=False):
        trsfm = transforms.Compose([
            transforms.Resize(img_side),
            transforms.CenterCrop(img_side),
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = IndexedDataset(datasets.CelebA(self.data_dir, split="train" if training else "valid", download=True, transform=trsfm))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, drop_last=drop_last)

class Flowers102DataLoader(BaseDataLoader):
    """
    Oxford 102 Flower data loading use BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, img_side=128, shuffle=False, validation_split=0.0, num_workers=1, training=True, drop_last=False):
        trsfm = transforms.Compose([
            transforms.Resize(img_side),
            transforms.CenterCrop(img_side),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
            transforms.Lambda(lambda t: (t * 2) - 1),
        ])
        self.data_dir = data_dir
        self.dataset = IndexedDataset(datasets.Flowers102(self.data_dir, split="train" if training else "test", download=True, transform=trsfm))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, drop_last=drop_last)

    @property
    def reverse_transform(self):
        return transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])
