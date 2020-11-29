import torchvision
from torchvision import transforms
from dataloaders.wrapper import CacheClassLabel
import torchvision.transforms.functional as TF
import torch

from sklearn.model_selection import train_test_split
import numpy as np
from typing import List


def get_subset(dataset, targets: List, ratio, seed: int = 42):
    split_indices, _, split_targets, _ = train_test_split(np.arange(len(dataset)), dataset,
                                                          stratify=targets,
                                                          train_size=ratio,
                                                          random_state=seed)
    subset = torch.utils.data.dataset.Subset(dataset, split_indices)
    subset.root = dataset.root
    return subset


class RotationTransform:
    """Rotate by one of the given angles."""
    
    def __init__(self, angle):
        self.angle = angle
    
    def __call__(self, x):
        return TF.rotate(x, self.angle)


def MNIST(dataroot, train_aug=False, angle=0, subset_size=None):
    # Add padding to make 32x32
    # normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28
    normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32
    rotate = RotationTransform(angle=angle)
    
    val_transform = transforms.Compose([
        transforms.Pad(2, fill=0, padding_mode='constant'),
        rotate,
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            rotate,
            transforms.ToTensor(),
            normalize,
        ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    
    if subset_size is not None and subset_size < 1:
        train_dataset = get_subset(dataset=train_dataset,
                                   targets=train_dataset.targets.tolist(),
                                   ratio=subset_size)
    
    train_dataset = CacheClassLabel(train_dataset)
    
    val_dataset = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    if subset_size is not None and subset_size < 1:
        val_dataset = get_subset(dataset=val_dataset,
                                 targets=val_dataset.targets.tolist(),
                                 ratio=subset_size)
    val_dataset = CacheClassLabel(val_dataset)
    assert val_dataset.number_classes == train_dataset.number_classes
    return train_dataset, val_dataset



def CIFAR10(dataroot, train_aug=False, angle=0, subset_size=None):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    rotate = RotationTransform(angle=angle)
    
    val_transform = transforms.Compose([
        rotate,
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            rotate,
            transforms.ToTensor(),
            normalize,
        ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    if subset_size is not None and subset_size < 1:
        train_dataset = get_subset(dataset=train_dataset,
                                   targets=train_dataset.targets.tolist(),
                                   ratio=subset_size)
    train_dataset = CacheClassLabel(train_dataset)
    
    val_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    if subset_size is not None and subset_size < 1:
        val_dataset = get_subset(dataset=val_dataset,
                                 targets=val_dataset.targets.tolist(),
                                 ratio=subset_size)
    val_dataset = CacheClassLabel(val_dataset)
    assert val_dataset.number_classes == train_dataset.number_classes
    return train_dataset, val_dataset


def CIFAR100(dataroot, train_aug=False, angle=0, subset_size=None):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    if subset_size is not None and subset_size < 1:
        train_dataset = get_subset(dataset=train_dataset,
                                   targets=train_dataset.targets,
                                   ratio=subset_size)
    train_dataset = CacheClassLabel(train_dataset)
    
    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    if subset_size is not None and subset_size < 1:
        val_dataset = get_subset(dataset=val_dataset,
                                 targets=val_dataset.targets,
                                 ratio=subset_size)
    val_dataset = CacheClassLabel(val_dataset)
    
    assert val_dataset.number_classes == train_dataset.number_classes
    return train_dataset, val_dataset


if __name__ == '__main__':
    train_dataset, val_dataset = CIFAR100(dataroot='/tmp/datasets', subset_size=0.25)
    print(f'len(train_dataset) : {len(train_dataset)}')
    print(f'len(val_dataset) : {len(val_dataset)}')
    assert val_dataset.number_classes == train_dataset.number_classes
