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
        self.batch_size = batch_size
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
        return DataLoader(IndexedDataset(self.mnist_test), num_workers=2,
                          batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.mnist_train), num_workers=2,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(IndexedDataset(self.mnist_val), num_workers=2,
                          batch_size=self.batch_size)

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

class BouncingMnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.mT)
        ])

    def setup(self, stage=None):
        bouncingmnist_full = BouncingMNIST(self.data_dir, transform=self.transform)
        self.bmnist_train, self.bmnist_val = random_split(
            bouncingmnist_full, [0.9 * len(bouncingmnist_full),
                                 0.1 * len(bouncingmnist_full)]
        )

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.bmnist_train),
                          batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(IndexedDataset(self.bmnist_val),
                          batch_size=BATCH_SIZE)

class MiniBouncingMnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.mT)
        ])

    def setup(self, stage=None):
        bouncingmnist_full = MiniBouncingMNIST(self.data_dir,
                                               transform=self.transform)
        self.bmnist_train, self.bmnist_val = random_split(
            bouncingmnist_full, [0.9 * len(bouncingmnist_full),
                                 0.1 * len(bouncingmnist_full)]
        )

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.bmnist_train),
                          batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(IndexedDataset(self.bmnist_val),
                          batch_size=BATCH_SIZE)

class CelebADataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, side=64):
        super().__init__()
        self.data_dir = data_dir
        self.reverse_transform = transforms.Lambda(lambda t: t.mT)
        self.transform = transforms.Compose([
            transforms.Resize(side),
            transforms.CenterCrop(side),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.mT),
        ])
        self.dims = (3, side, side)

    def prepare_data(self):
        datasets.CelebA(self.data_dir, split="test", download=True)
        datasets.CelebA(self.data_dir, split="train", download=True)
        datasets.CelebA(self.data_dir, split="valid", download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.celeba_train = datasets.CelebA(self.data_dir, split="train",
                                                download=True,
                                                transform=self.transform)
            self.celeba_val = datasets.CelebA(self.data_dir, split="valid",
                                              download=True,
                                              transform=self.transform)

        if stage == "test" or stage is None:
            self.celeba_test = datasets.CelebA(self.data_dir, split="test",
                                               download=True,
                                               transform=self.transform)

    def test_dataloader(self):
        return DataLoader(IndexedDataset(self.celeba_test),
                          batch_size=BATCH_SIZE)

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.celeba_train),
                          batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(IndexedDataset(self.celeba_val), batch_size=BATCH_SIZE)

class Flowers102DataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, side=64):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.reverse_transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])
        self.transform = transforms.Compose([
            transforms.Resize(side),
            transforms.CenterCrop(side),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
            transforms.Lambda(lambda t: (t * 2) - 1),
        ])
        self.dims = (3, side, side)

    def prepare_data(self):
        datasets.Flowers102(self.data_dir, split="test", download=True)
        datasets.Flowers102(self.data_dir, split="train", download=True)
        datasets.Flowers102(self.data_dir, split="val", download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.flowers102_train = datasets.Flowers102(self.data_dir, split="train",
                                                        download=True,
                                                        transform=self.transform)
            self.flowers102_val = datasets.Flowers102(self.data_dir, split="val",
                                                      download=True,
                                                      transform=self.transform)

        if stage == "test" or stage is None:
            self.flowers102_test = datasets.Flowers102(self.data_dir, split="test",
                                                       download=True,
                                                       transform=self.transform)

    def test_dataloader(self):
        return DataLoader(IndexedDataset(self.flowers102_test), num_workers=2,
                          batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.flowers102_train), num_workers=2,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(IndexedDataset(self.flowers102_val), num_workers=2,
                          batch_size=self.batch_size)
