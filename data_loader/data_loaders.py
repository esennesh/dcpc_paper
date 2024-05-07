import lightning as L
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Any, Callable, Optional, Tuple

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

class MnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()
        self.dims = (1, 28, 28)

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True,
                                        transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000,
                                                                         5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False,
                                    transform=self.transform)

    def test_dataloader(self):
        return DataLoader(IndexedDataset(self.mnist_test),
                          batch_size=BATCH_SIZE)

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.mnist_train),
                          batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(IndexedDataset(self.mnist_val), batch_size=BATCH_SIZE)

class EMnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()
        self.dims = (1, 28, 28)

    def prepare_data(self):
        datasets.EMNIST(self.data_dir, split="balanced", train=True, download=True)
        datasets.EMNIST(self.data_dir, split="balanced", train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            emnist_full = datasets.EMNIST(self.data_dir, split="balanced",
                                          train=True, transform=self.transform)
            self.emnist_train, self.emnist_val = random_split(
                emnist_full, [0.9 * len(emnist_full), 0.1 * len(emnist_full)]
            )

        if stage == "test" or stage is None:
            self.emnist_test = EMNIST(self.data_dir, split="balanced",
                                      train=False, transform=self.transform)

    def test_dataloader(self):
        return DataLoader(IndexedDataset(self.emnist_test),
                          batch_size=BATCH_SIZE)

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.emnist_train),
                          batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(IndexedDataset(self.emnist_val), batch_size=BATCH_SIZE)

class FashionMnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()
        self.dims = (1, 28, 28)

    def prepare_data(self):
        datasets.FashionMNIST(self.data_dir, split="balanced", train=True, download=True)
        datasets.FashionMNIST(self.data_dir, split="balanced", train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            fashionmnist_full = datasets.FashionMNIST(self.data_dir,
                                                      split="balanced",
                                                      train=True,
                                                      transform=self.transform)
            self.fashionmnist_train, self.fashionmnist_val = random_split(
                fashionmnist_full, [0.9 * len(fashionmnist_full), 0.1 * len(fashionmnist_full)]
            )

        if stage == "test" or stage is None:
            self.fashionmnist_test = datasets.FashionMNIST(self.data_dir,
                                                           split="balanced",
                                                           train=False,
                                                           transform=self.transform)

    def test_dataloader(self):
        return DataLoader(IndexedDataset(self.fashionmnist_test),
                          batch_size=BATCH_SIZE)

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.fashionmnist_train),
                          batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(IndexedDataset(self.fashionmnist_val), batch_size=BATCH_SIZE)

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
            transforms.Lambda(lambda t: t.mT),
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
