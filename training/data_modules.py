import lightning as L
import glob
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
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
        for file in glob.glob(self.root + "/ob-*.npy"):
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

class MiniBouncingMNIST(BouncingMNIST):
    def _load_data(self):
        data = []
        for file in glob.glob(self.root + "/ob-*.npy"):
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
            self.mnist_train, self.mnist_val = random_split(mnist_full,
                                                            [0.9, 0.1])

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
        self.batch_size = batch_size
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
            num_validation = int(0.1 * len(emnist_full))
            num_training = len(emnist_full) - num_validation
            self.emnist_train, self.emnist_val = random_split(emnist_full,
                                                              [num_training,
                                                              num_validation])

        if stage == "test" or stage is None:
            self.emnist_test = EMNIST(self.data_dir, split="balanced",
                                      train=False, transform=self.transform)

    def test_dataloader(self):
        return DataLoader(IndexedDataset(self.emnist_test), num_workers=2,
                          batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.emnist_train), num_workers=2,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(IndexedDataset(self.emnist_val), num_workers=2,
                          batch_size=self.batch_size)

class FashionMnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()
        self.dims = (1, 28, 28)

    def prepare_data(self):
        datasets.FashionMNIST(self.data_dir, train=True, download=True)
        datasets.FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            fashionmnist_full = datasets.FashionMNIST(self.data_dir,
                                                      train=True,
                                                      transform=self.transform)
            num_validation = int(0.1 * len(fashionmnist_full))
            num_training = len(fashionmnist_full) - num_validation
            self.fashionmnist_train, self.fashionmnist_val = random_split(
                fashionmnist_full, [num_training, num_validation]
            )

        if stage == "test" or stage is None:
            self.fashionmnist_test = datasets.FashionMNIST(self.data_dir,
                                                           train=False,
                                                           transform=self.transform)

    def test_dataloader(self):
        return DataLoader(IndexedDataset(self.fashionmnist_test), num_workers=2,
                          batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.fashionmnist_train), num_workers=2,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(IndexedDataset(self.fashionmnist_val), num_workers=2,
                          batch_size=self.batch_size)

class BouncingMnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.dims = (20, 1, 96, 96)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.mT)
        ])

    def setup(self, stage=None):
        bouncingmnist_full = BouncingMNIST(self.data_dir, transform=self.transform)
        train_length = int(0.9 * len(bouncingmnist_full))
        valid_length = len(bouncingmnist_full) - train_length
        self.bmnist_train, self.bmnist_val = random_split(
            bouncingmnist_full, [train_length, valid_length]
        )

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.bmnist_train), num_workers=2,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(IndexedDataset(self.bmnist_val), num_workers=2,
                          batch_size=self.batch_size)

class MiniBouncingMnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.dims = (20, 1, 96, 96)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.mT)
        ])

    def setup(self, stage=None):
        bouncingmnist_full = MiniBouncingMNIST(self.data_dir,
                                               transform=self.transform)
        train_length = int(0.9 * len(bouncingmnist_full))
        valid_length = len(bouncingmnist_full) - train_length
        self.bmnist_train, self.bmnist_val = random_split(
            bouncingmnist_full, [train_length, valid_length]
        )

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.bmnist_train), num_workers=2,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(IndexedDataset(self.bmnist_val), num_workers=2,
                          batch_size=self.batch_size)

class CelebADataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, side=64):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.reverse_transform = transforms.Normalize((-1, -1, -1), (2, 2, 2))
        self.transform = transforms.Compose([
            transforms.Resize(side),
            transforms.CenterCrop(side),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.mT),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        return DataLoader(IndexedDataset(self.celeba_test), num_workers=2,
                          batch_size=self.batch_size, pin_memory=True)

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.celeba_train), num_workers=2,
                          batch_size=self.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(IndexedDataset(self.celeba_val), num_workers=2,
                          batch_size=self.batch_size, pin_memory=True)

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

class CifarMemoryDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_seqs=5, seq_len=10, side=32):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_seqs = num_seqs
        self.transform = transforms.Compose([
            transforms.Resize(side),
            transforms.ToTensor()
        ])
        self.dims = (seq_len, 3, side, side)

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True,
                         transform=self.transform)

    def _sample_seqs(self, imgs):
        seqs = []
        for i in range(self.num_seqs):
            indices = torch.randint(0, len(imgs), (self.dims[0],))
            seq = [imgs[idx][0] for idx in indices]
            seqs.append(torch.stack(seq, dim=0))
        return torch.stack(seqs, dim=0), torch.zeros(len(seqs), 1)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.cifar10_train = datasets.CIFAR10(self.data_dir, train=True,
                                                  download=True,
                                                  transform=self.transform)
            seqs, targets = self._sample_seqs(self.cifar10_train)
            self.train_sequences = TensorDataset(seqs, targets)

        if stage == "test" or stage is None:
            self.cifar10_test = datasets.CIFAR10(self.data_dir, train=False,
                                                  download=True,
                                                  transform=self.transform)
            seqs, targets = self._sample_seqs(self.cifar10_test)
            self.test_sequences = TensorDataset(seqs, targets)

    def train_dataloader(self):
        return DataLoader(IndexedDataset(self.train_sequences), num_workers=2,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return self.train_dataloader()
